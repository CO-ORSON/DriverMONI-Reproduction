"""Independent controlled probes of D's selection, RNG and resume boundaries."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.evaluate import EvaluationResult, METRIC_NAMES
from src import tuning
from src.utils import load_config
from tests.test_validation_training import make_dataset, expected_splits, dataset_labels


@pytest.fixture
def tuning_inputs():
    dataset = make_dataset()
    config = load_config(Path(__file__).parents[1] / "configs/brca_validation_train_mode.yaml")
    config["device"] = "cpu"
    config["training"].update(epochs=3, batch_size=4)
    search = {"learning_rates": [0.001, 0.0003, 0.0001],
              "dropout_pairs": [[0.7, 0.6], [0.5, 0.3]], "selection_metric": "auc"}
    return dataset, config, search


class ProtocolProbe:
    """Real loaders/Adam/checkpoints, controlled training and validation scores."""

    def __init__(self, monkeypatch, dataset, config, search, fold_number=1, *, control_wins=False):
        self.dataset, self.config = dataset, config
        self.candidates = tuning.tuning_candidates(search)
        self.by_id = {candidate["candidate_id"]: candidate for candidate in self.candidates}
        self.events, self.models, self.train_orders, self.initial_states = [], [], [], []
        self.completed, self.fail_once = set(), None
        _, self.fit, self.validation, self.test = list(expected_splits(dataset, config["seed"]))[fold_number - 1]
        # Candidate 2 wins the cross-candidate tie even though candidate 3 reaches
        # its equal maximum at an earlier epoch. Within candidate 2, epoch 2 wins.
        self.aucs = [[0.6, 0.7, 0.8], [0.7, 0.9, 0.9], [0.9, 0.8, 0.7],
                     [0.6, 0.6, 0.6], [0.5, 0.6, 0.7], [0.6, 0.7, 0.65]]
        if control_wins:
            self.aucs[0] = [0.95, 0.95, 0.9]
        probe = self

        class Model(nn.Module):
            def __init__(self, candidate):
                super().__init__()
                self.weight = nn.Parameter(torch.rand(4))
                self.bn = nn.BatchNorm1d(2)
                self.register_buffer("saved_epoch", torch.tensor(0, dtype=torch.long))
                self.candidate = candidate["candidate_id"]
                self.epoch = 0

        def new_model(candidate, device):
            model = Model(candidate).to(device)
            probe.models.append(model)
            return model

        def train(model, loader, optimizer, weights, device):
            assert model.training and model.bn.training
            assert loader.dataset.indices == probe.fit.tolist()
            assert not set(loader.dataset.indices) & set(probe.test.tolist())
            candidate = probe.by_id[model.candidate]
            assert optimizer.param_groups[0]["lr"] == candidate["learning_rate"]
            torch.testing.assert_close(weights.cpu(), torch.tensor([0.75, 1.5], dtype=torch.float64))
            if model.epoch == 0:
                probe.initial_states.append((model.candidate, deepcopy(model.state_dict())))
            model.epoch += 1
            if probe.fail_once == (model.candidate, model.epoch):
                probe.fail_once = None
                raise RuntimeError("controlled interrupted candidate")
            order = [gene for batch in loader for gene in batch.gene_ids]
            probe.train_orders.append((model.candidate, model.epoch, order))
            number = int(model.candidate.rsplit("_", 1)[1])
            # Different stochastic workload must not change the explicit loader
            # generator's permutations, nor a later candidate's initialization.
            draw = torch.rand(number * 13 + model.epoch).sum().item()
            with torch.no_grad():
                model.weight.fill_(number * 10 + model.epoch)
                model.bn.running_mean.fill_(100 + model.epoch)
                model.bn.running_var.fill_(200 + model.epoch)
                model.bn.num_batches_tracked.fill_(300 + model.epoch)
                model.saved_epoch.fill_(model.epoch)
            probe.events.append((model.candidate, "train", model.epoch))
            return draw

        def evaluate(model, loader, device, threshold, zero_division):
            assert threshold == 0.5
            assert zero_division == config["evaluation"]["zero_division"]
            model.eval()
            indices = loader.dataset.indices
            number = int(model.candidate.rsplit("_", 1)[1])
            saved_epoch = int(model.saved_epoch.item())
            if indices == probe.validation.tolist():
                split = "validation"
                auc = probe.aucs[number - 1][model.epoch - 1]
                assert saved_epoch == model.epoch
                if model.epoch == config["training"]["epochs"]:
                    probe.completed.add(model.candidate)
            else:
                assert indices == probe.test.tolist()
                split = "test"
                # No target test evaluation is allowed before every trial has
                # completed its prespecified training/validation budget.
                assert probe.completed == set(probe.by_id)
                expected_epoch = int(np.argmax(probe.aucs[number - 1])) + 1
                assert saved_epoch == expected_epoch
                assert torch.all(model.weight == number * 10 + expected_epoch)
                assert torch.all(model.bn.running_mean == 100 + expected_epoch)
                assert torch.all(model.bn.running_var == 200 + expected_epoch)
                assert model.bn.num_batches_tracked.item() == 300 + expected_epoch
                # Deliberately reverse test performance: test AUC must not alter
                # the validation-selected candidate or epoch.
                auc = 0.99 if number == 1 else 0.11
            # Exercise evaluation loader construction too, so iterator base-seed
            # generation cannot accidentally be omitted from the RNG probe.
            observed_genes = [gene for batch in loader for gene in batch.gene_ids]
            assert set(observed_genes) == {dataset.target_gene_ids[index] for index in indices}
            probe.events.append((model.candidate, split, saved_epoch))
            labels = dataset_labels(dataset)
            predictions = [{"gene_id": dataset.target_gene_ids[index], "label": int(labels[index]),
                            "driver_probability": saved_epoch / 10} for index in indices]
            return EvaluationResult({metric: auc for metric in METRIC_NAMES}, predictions)

        monkeypatch.setattr(tuning, "_new_model", new_model)
        monkeypatch.setattr(tuning, "train_one_epoch", train)
        monkeypatch.setattr(tuning, "evaluate", evaluate)


def test_validation_only_selection_ties_and_full_state_restore(tuning_inputs, tmp_path, monkeypatch):
    dataset, config, search = tuning_inputs
    probe = ProtocolProbe(monkeypatch, dataset, config, search)
    report = tuning.run_tuning_fold(dataset, config, search, 1, tmp_path / "fold", log=lambda _: None)
    assert report["status"] == "complete"
    assert report["selected_candidate_id"] == "candidate_02"
    assert report["tuned_test"]["selected_epoch"] == 2
    assert report["fixed_c_test"]["selected_epoch"] == 3
    assert report["tuned_test"]["metrics"]["auc"] == 0.11
    assert report["fixed_c_test"]["metrics"]["auc"] == 0.99
    assert report["validation_ranking"][:2] == ["candidate_02", "candidate_03"]
    assert report["test_evaluation_count"] == 2
    expected = [(f"candidate_{candidate:02d}", phase, epoch)
                for candidate in range(1, 7) for epoch in range(1, 4)
                for phase in ("train", "validation")]
    assert probe.events == expected + [("candidate_02", "test", 2), ("candidate_01", "test", 3)]
    assert all("test_metrics" not in epoch for result in report["candidate_results"]
               for epoch in result["epoch_history"])
    # Inspect every persisted best state, including unselected candidates; this
    # catches aliasing of the selected checkpoint with later parameter updates.
    for number, result in enumerate(report["candidate_results"], 1):
        state = torch.load(tmp_path / "fold" / f"candidate_{number:02d}.pt", weights_only=True)
        epoch = int(np.argmax(probe.aucs[number - 1])) + 1
        assert result["selected_epoch"] == epoch
        assert torch.all(state["weight"] == number * 10 + epoch)
        assert state["bn.num_batches_tracked"].item() == 300 + epoch


def test_fixed_control_winner_reuses_one_test_pass(tuning_inputs, tmp_path, monkeypatch):
    dataset, config, search = tuning_inputs
    probe = ProtocolProbe(monkeypatch, dataset, config, search, control_wins=True)
    report = tuning.run_tuning_fold(dataset, config, search, 1, tmp_path / "fold", log=lambda _: None)
    assert report["selected_candidate_id"] == "candidate_01"
    assert report["tuned_test"]["selected_epoch"] == 1
    assert report["test_evaluation_count"] == 1
    assert report["tuned_test"] == report["fixed_c_test"]
    assert [event for event in probe.events if event[1] == "test"] == [("candidate_01", "test", 1)]


@pytest.mark.parametrize("fold_number", [1, 3])
def test_initialization_and_epoch_order_match_despite_different_rng_workloads(
    tuning_inputs, tmp_path, monkeypatch, fold_number,
):
    dataset, config, search = tuning_inputs
    probe = ProtocolProbe(monkeypatch, dataset, config, search, fold_number)
    report = tuning.run_tuning_fold(dataset, config, search, fold_number, tmp_path / "fold", log=lambda _: None)
    expected_seed = config["seed"] + fold_number - 1
    assert report["seeds"] == {"model": expected_seed, "train_loader": expected_seed,
                               "validation_loader": expected_seed + 10000, "test_loader": expected_seed + 20000}
    initial = probe.initial_states[0][1]
    assert len(probe.initial_states) == 6
    for _, state in probe.initial_states[1:]:
        for key, value in initial.items():
            torch.testing.assert_close(state[key], value, rtol=0, atol=0)
    for epoch in range(1, 4):
        orders = [order for _, observed_epoch, order in probe.train_orders if observed_epoch == epoch]
        assert len(orders) == 6 and all(order == orders[0] for order in orders)
    assert probe.train_orders[0][2] != probe.train_orders[1][2]


def test_resume_restarts_failed_trial_without_rng_drift_or_early_test(
    tuning_inputs, tmp_path, monkeypatch,
):
    dataset, config, search = tuning_inputs
    probe = ProtocolProbe(monkeypatch, dataset, config, search)
    probe.fail_once = ("candidate_03", 2)
    directory = tmp_path / "resumed"
    with pytest.raises(RuntimeError, match="controlled interrupted candidate"):
        tuning.run_tuning_fold(dataset, config, search, 1, directory, log=lambda _: None)
    interrupted = json.loads((directory / "report.json").read_text())
    assert interrupted["status"] == "failed"
    assert len(interrupted["candidate_results"]) == 2
    assert not any(event[1] == "test" for event in probe.events)
    resumed = tuning.run_tuning_fold(dataset, config, search, 1, directory, resume=True, log=lambda _: None)
    assert [candidate for candidate, _ in probe.initial_states].count("candidate_01") == 1
    assert [candidate for candidate, _ in probe.initial_states].count("candidate_02") == 1
    assert [candidate for candidate, _ in probe.initial_states].count("candidate_03") == 2
    candidate_3_first_epoch = [order for candidate, epoch, order in probe.train_orders
                             if candidate == "candidate_03" and epoch == 1]
    assert len(candidate_3_first_epoch) == 2 and candidate_3_first_epoch[0] == candidate_3_first_epoch[1]
    fresh_probe = ProtocolProbe(monkeypatch, dataset, config, search)
    fresh = tuning.run_tuning_fold(dataset, config, search, 1, tmp_path / "fresh", log=lambda _: None)
    assert resumed["tuned_test"] == fresh["tuned_test"]
    assert resumed["fixed_c_test"] == fresh["fixed_c_test"]
    assert resumed["selected_candidate_id"] == fresh["selected_candidate_id"]
    for first, second in zip(resumed["candidate_results"], fresh["candidate_results"], strict=True):
        assert first["selected_validation_metrics"] == second["selected_validation_metrics"]
        assert first["validation_predictions"] == second["validation_predictions"]
        assert [epoch["mean_batch_loss"] for epoch in first["epoch_history"]] == [
            epoch["mean_batch_loss"] for epoch in second["epoch_history"]]
    before = len(fresh_probe.events)
    completed = tuning.run_tuning_fold(dataset, config, search, 1, tmp_path / "fresh", resume=True, log=lambda _: None)
    assert completed == fresh and len(fresh_probe.events) == before


def test_resume_after_selected_test_preserves_that_result_without_rescoring(
    tuning_inputs, tmp_path, monkeypatch,
):
    dataset, config, search = tuning_inputs
    probe = ProtocolProbe(monkeypatch, dataset, config, search)
    evaluate = tuning.evaluate
    stop_before_control = True

    def interrupt_before_control(model, loader, *args):
        nonlocal stop_before_control
        if (stop_before_control and model.candidate == "candidate_01"
                and loader.dataset.indices == probe.test.tolist()):
            stop_before_control = False
            raise KeyboardInterrupt("controlled interruption before control test")
        return evaluate(model, loader, *args)

    monkeypatch.setattr(tuning, "evaluate", interrupt_before_control)
    directory = tmp_path / "fold"
    with pytest.raises(KeyboardInterrupt):
        tuning.run_tuning_fold(dataset, config, search, 1, directory, log=lambda _: None)
    partial = json.loads((directory / "report.json").read_text())
    assert partial["status"] == "interrupted"
    assert "tuned_test" in partial and "fixed_c_test" not in partial
    assert len(partial["candidate_results"]) == 6
    train_calls = [event for event in probe.events if event[1] == "train"]
    resumed = tuning.run_tuning_fold(dataset, config, search, 1, directory, resume=True, log=lambda _: None)
    assert resumed["status"] == "complete"
    assert resumed["tuned_test"] == partial["tuned_test"]
    assert [event for event in probe.events if event[1] == "train"] == train_calls
    assert [event for event in probe.events if event[1] == "test"] == [
        ("candidate_02", "test", 2), ("candidate_01", "test", 3)]


@pytest.mark.parametrize("changed", ["features", "input_manifest", "threshold"])
def test_resume_rejects_changed_data_provenance_and_threshold(
    tuning_inputs, tmp_path, monkeypatch, changed,
):
    dataset, config, search = tuning_inputs
    probe = ProtocolProbe(monkeypatch, dataset, config, search)
    directory = tmp_path / "fold"
    manifest = [{"role": "features", "sha256": "original"}]
    tuning.run_tuning_fold(dataset, config, search, 1, directory, input_files=manifest, log=lambda _: None)
    calls = len(probe.events)
    if changed == "features":
        dataset.node_features[0, 0] += 1.0
    elif changed == "input_manifest":
        manifest[0]["sha256"] = "changed"
    else:
        config["evaluation"]["probability_threshold"] = 0.4
    with pytest.raises(ValueError, match="changed|threshold"):
        tuning.run_tuning_fold(dataset, config, search, 1, directory, input_files=manifest,
                               resume=True, log=lambda _: None)
    assert len(probe.events) == calls
