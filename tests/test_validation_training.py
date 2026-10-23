"""Variant B: select on validation targets, restore state, then test once."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch import nn

from src.dataset import DriverMONIDataset, GAT_FEATURES
from src.evaluate import EvaluationResult, METRIC_NAMES
from src.graph_utils import PPIGraph
from src.train import run_cross_validation
from src.utils import load_config, validate_config


def make_dataset(size: int = 30, positives: int = 9) -> DriverMONIDataset:
    """Synthetic directed-ring graphs with imbalanced target labels."""
    genes = [f"validation_gene_{index}" for index in range(size)]
    rows = []
    for index, gene in enumerate(genes):
        row = {feature: float(np.sin(index + offset))
               for offset, feature in enumerate(GAT_FEATURES)}
        row.update(gene_ens=gene, gene_symbol=gene, label=int(index < positives))
        rows.append(row)
    edges = pd.DataFrame({"entrez_id_1": genes, "entrez_id_2": genes[1:] + genes[:1]})
    ppi = PPIGraph(edges, genes)
    return DriverMONIDataset(pd.DataFrame(rows), ppi, ppi.discover_neighbors(genes))


@pytest.fixture
def validation_dataset():
    return make_dataset()


@pytest.fixture
def validation_config():
    config = load_config(Path(__file__).parents[1] / "configs/brca.yaml")
    config["device"] = "cpu"
    config["training"]["epochs"] = 3
    config["training"]["batch_size"] = 4
    config["cross_validation"].update(
        epoch_selection="best_validation_roc_auc", validation_fraction=0.25,
    )
    return config


def dataset_labels(dataset):
    return np.array([int(dataset.context.loc[gene, "label"])
                     for gene in dataset.target_gene_ids])


def expected_splits(dataset, seed=42):
    labels = dataset_labels(dataset)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for outer_train, test in splitter.split(np.zeros(len(labels)), labels):
        fit, validation = train_test_split(
            outer_train, test_size=0.25, random_state=seed, stratify=labels[outer_train],
        )
        yield outer_train, fit, validation, test


def gene_ids(dataset, indices):
    return [dataset.target_gene_ids[index] for index in indices]


def test_validation_cv_runs_real_model_and_replays(validation_dataset, validation_config, tmp_path):
    output = tmp_path / "validation.json"
    report = run_cross_validation(validation_dataset, validation_config, output, log=lambda _: None)
    assert report == json.loads(output.read_text())
    assert report["status"] == "complete"
    protocol = report["protocol"]
    assert protocol["test_fold_used_for_epoch_selection"] is False
    assert protocol["test_evaluations_per_fold"] == 1
    assert protocol["validation_fraction"] == 0.25
    assert protocol["validation_split_seed"] == validation_config["seed"]
    assert protocol["source_preserve_eval_mode"] is True
    held_out = []
    for fold, (outer, fit, validation, test) in zip(
        report["folds"], expected_splits(validation_dataset, validation_config["seed"]), strict=True,
    ):
        # Outer folds remain exactly the original StratifiedKFold assignment.
        assert fold["outer_train_gene_ids"] == gene_ids(validation_dataset, outer)
        assert fold["train_gene_ids"] == gene_ids(validation_dataset, fit)
        assert fold["validation_gene_ids"] == gene_ids(validation_dataset, validation)
        assert fold["test_gene_ids"] == gene_ids(validation_dataset, test)
        fit_ids, validation_ids, test_ids = map(set, (
            fold["train_gene_ids"], fold["validation_gene_ids"], fold["test_gene_ids"],
        ))
        assert not (fit_ids & validation_ids or fit_ids & test_ids or validation_ids & test_ids)
        assert fit_ids | validation_ids == set(fold["outer_train_gene_ids"])
        assert fold["outer_train_counts"] == {"driver": 6, "non_driver": 14}
        assert fold["train_counts"] == {"driver": 5, "non_driver": 10}
        assert fold["validation_counts"] == {"driver": 1, "non_driver": 4}
        assert fold["test_counts"] == {"driver": 3, "non_driver": 7}
        assert fold["class_weights"] == {"driver": (1 / 5) * (15 / 2),
                                          "non_driver": (1 / 10) * (15 / 2)}
        assert {prediction["gene_id"] for prediction in fold["predictions"]} == test_ids
        assert {prediction["gene_id"] for prediction in fold["validation_predictions"]} == validation_ids
        history = fold["epoch_history"]
        assert [epoch["training_mode_during_updates"] for epoch in history] == [True, False, False]
        assert [epoch["learning_rate"] for epoch in history] == [0.001] * 3
        assert all(epoch["evaluation_split"] == "validation" for epoch in history)
        assert all("metrics" not in epoch and "test_metrics" not in epoch for epoch in history)
        best_index = int(np.argmax([epoch["validation_metrics"]["auc"] for epoch in history]))
        assert fold["selected_epoch"] == best_index + 1
        assert fold["selected_validation_metrics"] == history[best_index]["validation_metrics"]
        held_out.extend(fold["test_gene_ids"])
    assert sorted(held_out) == sorted(validation_dataset.target_gene_ids)
    for metric, summary in report["summary"].items():
        values = [fold["metrics"][metric] for fold in report["folds"]]
        assert summary == {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0))}

    replay = run_cross_validation(validation_dataset, validation_config, tmp_path / "replay.json", log=lambda _: None)
    assert replay["summary"] == report["summary"]
    for first, second in zip(report["folds"], replay["folds"], strict=True):
        for key in ("outer_train_gene_ids", "train_gene_ids", "validation_gene_ids", "test_gene_ids",
                    "class_weights", "selected_epoch", "selected_validation_metrics",
                    "validation_predictions", "metrics", "predictions"):
            assert first[key] == second[key]
        assert [epoch["mean_batch_loss"] for epoch in first["epoch_history"]] == [
            epoch["mean_batch_loss"] for epoch in second["epoch_history"]
        ]


@pytest.mark.parametrize("validation_aucs, selected_epoch", [([0.9, 0.8, 0.7], 1), ([0.8, 0.9, 0.9], 2)])
def test_only_validation_selects_epoch_and_full_state_is_restored(
    validation_dataset, validation_config, tmp_path, monkeypatch, validation_aucs, selected_epoch,
):
    splits = list(expected_splits(validation_dataset, validation_config["seed"]))
    models, events = [], []

    class StatefulModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))
            # BatchNorm has both floating-point and integer persistent buffers.
            self.register_buffer("running_mean", torch.zeros(1))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
            self.fold = len(models)
            self.epochs_run = 0
            models.append(self)

    def fake_train(model, loader, optimizer, weights, device):
        _, fit, _, _ = splits[model.fold]
        assert loader.dataset.indices == fit.tolist()
        assert model.training is (model.epochs_run == 0)
        torch.testing.assert_close(weights, torch.tensor([(1 / 10) * (15 / 2), (1 / 5) * (15 / 2)], dtype=torch.float64))
        model.epochs_run += 1
        with torch.no_grad():
            model.weight.fill_(model.epochs_run)
            model.running_mean.fill_(10 * model.epochs_run)
            model.num_batches_tracked.fill_(100 + model.epochs_run)
        events.append((model.fold, "train", model.epochs_run))
        return float(model.epochs_run)

    def fake_evaluate(model, loader, device, threshold, zero_division):
        model.eval()  # Preserve evaluate()'s source-compatible mode side effect.
        _, _, validation, test = splits[model.fold]
        indices = loader.dataset.indices
        if indices == validation.tolist():
            split = "validation"
            assert model.weight.item() == model.epochs_run
            auc = validation_aucs[model.epochs_run - 1]
        else:
            assert indices == test.tolist()
            split = "test"
            # No test scoring during training; both parameters and buffers restored.
            assert model.epochs_run == validation_config["training"]["epochs"]
            assert model.weight.item() == selected_epoch
            assert model.running_mean.item() == 10 * selected_epoch
            assert model.num_batches_tracked.item() == 100 + selected_epoch
            auc = 0.51
        events.append((model.fold, split, int(model.weight.item())))
        predictions = [{"gene_id": validation_dataset.target_gene_ids[index],
                        "label": int(dataset_labels(validation_dataset)[index]),
                        "driver_probability": model.weight.item() / 10}
                       for index in indices]
        return EvaluationResult({metric: auc for metric in METRIC_NAMES}, predictions)

    monkeypatch.setattr("src.train.DriverMONI", StatefulModel)
    monkeypatch.setattr("src.train.train_one_epoch", fake_train)
    monkeypatch.setattr("src.train.evaluate", fake_evaluate)
    report = run_cross_validation(validation_dataset, validation_config, tmp_path / "controlled.json", log=lambda _: None)
    assert len(models) == 3
    for number, fold in enumerate(report["folds"]):
        assert [event[1] for event in events if event[0] == number] == [
            "train", "validation", "train", "validation", "train", "validation", "test",
        ]
        assert fold["selected_epoch"] == selected_epoch
        assert fold["selected_validation_metrics"]["auc"] == max(validation_aucs)
        assert fold["metrics"]["auc"] == 0.51
        assert all(prediction["driver_probability"] == selected_epoch / 10
                   for prediction in fold["validation_predictions"] + fold["predictions"])


def test_validation_failure_is_reported_without_completed_fold(
    validation_dataset, validation_config, tmp_path, monkeypatch,
):
    def fail(*args, **kwargs):
        raise RuntimeError("controlled validation failure")
    monkeypatch.setattr("src.train.evaluate", fail)
    output = tmp_path / "failure.json"
    with pytest.raises(RuntimeError, match="controlled validation failure"):
        run_cross_validation(validation_dataset, validation_config, output, log=lambda _: None)
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert report["folds"] == []
    assert "controlled validation failure" in report["error"]


@pytest.mark.parametrize("fraction", [None, True, False, "0.25", 0, 1, -0.1, 1.1, np.nan, np.inf])
def test_validation_fraction_requires_finite_number_between_zero_and_one(validation_config, fraction):
    validation_config["cross_validation"]["validation_fraction"] = fraction
    with pytest.raises(ValueError, match="validation_fraction"):
        validate_config(validation_config)


def test_validation_fraction_is_required_only_for_validation_selection(validation_config):
    del validation_config["cross_validation"]["validation_fraction"]
    with pytest.raises(ValueError, match="validation_fraction"):
        validate_config(validation_config)
    validation_config["cross_validation"]["epoch_selection"] = "best_test_roc_auc"
    validate_config(validation_config)
    validation_config["cross_validation"]["validation_fraction"] = 0.25
    with pytest.raises(ValueError, match="validation_fraction"):
        validate_config(validation_config)


@pytest.mark.parametrize("size, positives, fraction, batch_size, error", [
    (30, 9, 0.01, 4, "validation|both.*class|two.*class"),
    (60, 3, 0.25, 4, "validation|both.*class"),
    (30, 9, 0.25, 7, "singleton"),
])
def test_unusable_inner_splits_fail_clearly(validation_config, tmp_path, size, positives, fraction, batch_size, error):
    validation_config["cross_validation"]["validation_fraction"] = fraction
    validation_config["training"]["batch_size"] = batch_size
    with pytest.raises(ValueError, match=error):
        run_cross_validation(make_dataset(size, positives), validation_config, tmp_path / "invalid.json", log=lambda _: None)


def test_published_validation_config_only_changes_protocol_and_output():
    root = Path(__file__).parents[1]
    original = load_config(root / "configs/brca_updated_data.yaml")
    variant = load_config(root / "configs/brca_validation.yaml")
    assert variant["cross_validation"]["epoch_selection"] == "best_validation_roc_auc"
    assert variant["cross_validation"]["validation_fraction"] == 0.25
    assert variant["training"]["preserve_eval_mode_after_first_evaluation"] is True
    assert variant["output_path"] != original["output_path"]
    normalized = deepcopy(variant)
    normalized["cross_validation"] = original["cross_validation"]
    normalized["output_path"] = original["output_path"]
    assert normalized == original
