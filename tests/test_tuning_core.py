"""Real-model experiment-D replay, invalid settings and full-fold aggregation."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models import DriverMONI
from src.tuning import aggregate_tuning_reports, run_tuning_fold, tuning_candidates
from src.utils import load_config
from tests.test_validation_training import make_dataset


GRID = {"learning_rates": [0.001, 0.0003, 0.0001],
        "dropout_pairs": [[0.7, 0.6], [0.5, 0.3]], "selection_metric": "auc"}


def small_config():
    config = load_config(Path(__file__).parents[1] / "configs/brca_validation_train_mode.yaml")
    config["device"] = "cpu"
    config["training"]["epochs"] = 2
    config["training"]["batch_size"] = 4
    return config


@pytest.mark.parametrize("changes", [
    {"selection_metric": "f1"}, {"learning_rates": []}, {"dropout_pairs": []},
    {"learning_rates": [0.001, float("nan")]}, {"learning_rates": [0.001, 0]},
    {"learning_rates": [0.001, True]}, {"learning_rates": [0.001, 0.001]},
    {"dropout_pairs": [[0.7, 0.6], [0.5]]}, {"dropout_pairs": [[0.7, 0.6], [1, 0.3]]},
    {"dropout_pairs": [[0.7, 0.6], [-0.1, 0.3]]},
    {"dropout_pairs": [[0.7, 0.6], [0.7, 0.6]]}, {"learning_rates": [0.0003]},
    {"unknown": "unused"},
])
def test_invalid_search_rejected(changes):
    with pytest.raises(ValueError):
        tuning_candidates(dict(deepcopy(GRID), **changes))


@pytest.mark.parametrize("section,key,value", [
    ("training", "preserve_eval_mode_after_first_evaluation", True),
    ("evaluation", "probability_threshold", 0.4),
])
def test_incompatible_protocol_rejected_before_output(tmp_path, section, key, value):
    config = small_config()
    config[section][key] = value
    directory = tmp_path / "experiment"
    with pytest.raises(ValueError):
        run_tuning_fold(make_dataset(), config, GRID, 1, directory)
    assert not directory.exists()


def test_real_model_search_replay_defaults_and_aggregation(tmp_path):
    """Run all six choices through actual graph attention, optimizer and dropout."""
    dataset, config = make_dataset(), small_config()
    pristine_config = deepcopy(config)
    folds = [run_tuning_fold(dataset, config, GRID, fold, tmp_path / f"fold_{fold}", log=lambda _: None)
             for fold in (1, 2, 3)]
    assert config == pristine_config
    baseline = DriverMONI()
    assert (baseline.dnn.dropout1.p, baseline.dnn.dropout2.p) == (0.7, 0.6)
    for fold in folds:
        assert fold["status"] == "complete"
        assert len(fold["candidate_results"]) == 6
        assert fold["seeds"]["model"] == 42 + fold["fold"] - 1
        best_auc = max(item["selected_validation_metrics"]["auc"] for item in fold["candidate_results"])
        expected = next(item for item in fold["candidate_results"]
                        if item["selected_validation_metrics"]["auc"] == best_auc)
        assert fold["selected_candidate_id"] == expected["candidate_id"]
        expected_val_ids = set(fold["splits"]["validation_gene_ids"])
        for item in fold["candidate_results"]:
            assert {row["gene_id"] for row in item["validation_predictions"]} == expected_val_ids
            assert all(epoch["training_mode_during_updates"] for epoch in item["epoch_history"])
            state = torch.load(tmp_path / f"fold_{fold['fold']}" / f"{item['candidate_id']}.pt",
                               weights_only=True, map_location="cpu")
            baseline.load_state_dict(state, strict=True)
            # Full persistent running statistics are retained, not just parameters.
            assert state["dnn.batch_norm1.num_batches_tracked"].item() == 4 * item["selected_epoch"]
        test_ids = set(fold["splits"]["test_gene_ids"])
        assert not test_ids & expected_val_ids
        for name in ("tuned_test", "fixed_c_test"):
            assert {row["gene_id"] for row in fold[name]["predictions"]} == test_ids

    replay = run_tuning_fold(dataset, config, GRID, 1, tmp_path / "replay", log=lambda _: None)
    for first, second in zip(folds[0]["candidate_results"], replay["candidate_results"], strict=True):
        for key in ("selected_epoch", "selected_validation_metrics", "validation_predictions"):
            assert first[key] == second[key]
        assert [epoch["mean_batch_loss"] for epoch in first["epoch_history"]] == [
            epoch["mean_batch_loss"] for epoch in second["epoch_history"]]
    assert folds[0]["tuned_test"] == replay["tuned_test"]
    assert folds[0]["fixed_c_test"] == replay["fixed_c_test"]
    resumed = run_tuning_fold(dataset, config, GRID, 1, tmp_path / "fold_1", resume=True, log=lambda _: None)
    assert resumed == folds[0]
    with pytest.raises(FileExistsError):
        run_tuning_fold(dataset, config, GRID, 1, tmp_path / "fold_1", log=lambda _: None)

    aggregate = aggregate_tuning_reports(folds)
    for name, result in (("tuned_summary", "tuned_test"), ("fixed_c_summary", "fixed_c_test")):
        for metric, summary in aggregate[name].items():
            values = [fold[result]["metrics"][metric] for fold in folds]
            assert summary == {"mean": float(np.mean(values)), "std": float(np.std(values))}
    with pytest.raises(ValueError, match="exactly once"):
        aggregate_tuning_reports([folds[0], folds[1], folds[1]])
    changed = deepcopy(folds)
    changed[2]["status"] = "failed"
    with pytest.raises(ValueError, match="complete"):
        aggregate_tuning_reports(changed)
    changed = deepcopy(folds)
    changed[2]["provenance"]["dataset_sha256"] = "different"
    with pytest.raises(ValueError, match="provenance"):
        aggregate_tuning_reports(changed)
    changed = deepcopy(folds)
    changed[2]["splits"]["test_gene_ids"] = changed[1]["splits"]["test_gene_ids"]
    with pytest.raises(ValueError, match="exactly once"):
        aggregate_tuning_reports(changed)
