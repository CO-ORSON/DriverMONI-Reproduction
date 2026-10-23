"""Experiment C: validation selection with real training mode every epoch."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from src.evaluate import EvaluationResult, evaluate
from src.models import DriverMONI
from src.train import run_cross_validation, train_one_epoch
from src.utils import load_config
from tests.test_validation_training import (
    expected_splits, validation_config, validation_dataset,
)


@pytest.mark.parametrize("preserve_source_mode", [False, True], ids=["C", "B"])
def test_real_module_modes_and_selected_state(
    validation_dataset, validation_config, tmp_path, monkeypatch, preserve_source_mode,
):
    """Exercise actual Dropout, BatchNorm, optimizer and evaluation in every fold.

    Only the validation AUC is controlled, selecting epoch one to make restoration
    observable. Predictions, forward passes, gradients and buffers remain real.
    """
    validation_config["training"]["preserve_eval_mode_after_first_evaluation"] = preserve_source_mode
    splits = list(expected_splits(validation_dataset, validation_config["seed"]))
    records = []

    def create_observed_model():
        model = DriverMONI()
        record = {"fold": len(records), "epoch": 0, "phase": None, "events": [],
                  "states": [], "dropout_calls": [], "batch_norm_calls": []}
        records.append(record)
        model._experiment_test_record = record

        def check_dropout(module, inputs, output):
            expected_training = record["phase"] == "train" and (
                record["epoch"] == 1 or not preserve_source_mode
            )
            assert module.training is expected_training
            if expected_training:
                values = inputs[0]
                # Active Dropout both removes nonzero activations and scales the
                # survivors. The synthetic model batches provide many activations.
                assert torch.any((values != 0) & (output == 0))
                survivors = output != 0
                assert torch.any(survivors)
                torch.testing.assert_close(output[survivors], values[survivors] / (1 - module.p))
            else:
                torch.testing.assert_close(output, inputs[0], rtol=0, atol=0)
            record["dropout_calls"].append((record["epoch"], record["phase"], module.training))

        def check_batch_norm(module, inputs, output):
            expected_training = record["phase"] == "train" and (
                record["epoch"] == 1 or not preserve_source_mode
            )
            assert module.training is expected_training
            record["batch_norm_calls"].append((record["epoch"], record["phase"], module.training))

        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.register_forward_hook(check_dropout)
            elif isinstance(module, nn.BatchNorm1d):
                module.register_forward_hook(check_batch_norm)
        return model

    def observed_train(model, loader, optimizer, weights, device):
        record = model._experiment_test_record
        record["epoch"] += 1
        record["phase"] = "train"
        record["events"].append("train")
        expected_training = record["epoch"] == 1 or not preserve_source_mode
        assert model.training is expected_training
        batch_norms = [module for module in model.modules() if isinstance(module, nn.BatchNorm1d)]
        before = [(module.num_batches_tracked.item(), module.running_mean.clone(),
                   module.running_var.clone()) for module in batch_norms]
        loss = train_one_epoch(model, loader, optimizer, weights, device)
        for module, (count, mean, variance) in zip(batch_norms, before, strict=True):
            assert module.num_batches_tracked.item() == count + (len(loader) if expected_training else 0)
            if expected_training:
                assert not torch.equal(module.running_mean, mean)
                assert not torch.equal(module.running_var, variance)
            else:
                torch.testing.assert_close(module.running_mean, mean, rtol=0, atol=0)
                torch.testing.assert_close(module.running_var, variance, rtol=0, atol=0)
        return loss

    def observed_evaluate(model, loader, device, threshold, zero_division):
        record = model._experiment_test_record
        _, _, validation, test = splits[record["fold"]]
        indices = loader.dataset.indices
        if indices == validation.tolist():
            record["phase"] = "validation"
        else:
            assert indices == test.tolist()
            record["phase"] = "test"
            # Test evaluation happens after all three training/validation passes,
            # with epoch-one parameters AND every persistent buffer restored.
            assert record["epoch"] == 3
            selected_state = record["states"][0]
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, selected_state[name], rtol=0, atol=0)
            assert any(not torch.equal(selected_state[name], record["states"][-1][name])
                       for name, _ in model.named_parameters())
            if not preserve_source_mode:
                assert any(not torch.equal(selected_state[name], record["states"][-1][name])
                           for name, _ in model.named_buffers())

        record["events"].append(record["phase"])
        before = deepcopy(model.state_dict())
        result = evaluate(model, loader, device, threshold, zero_division)
        assert model.training is False
        # Evaluation must never update running statistics or model parameters.
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        if record["phase"] == "validation":
            record["states"].append(deepcopy(model.state_dict()))
            return EvaluationResult(dict(result.metrics, auc=[0.9, 0.8, 0.7][record["epoch"] - 1]),
                                    result.predictions)
        return result

    monkeypatch.setattr("src.train.DriverMONI", create_observed_model)
    monkeypatch.setattr("src.train.train_one_epoch", observed_train)
    monkeypatch.setattr("src.train.evaluate", observed_evaluate)
    report = run_cross_validation(validation_dataset, validation_config, tmp_path / "observed.json",
                                  log=lambda _: None)
    assert len(records) == 3
    assert report["protocol"]["source_preserve_eval_mode"] is preserve_source_mode
    assert report["protocol"]["test_fold_used_for_epoch_selection"] is False
    for record, fold in zip(records, report["folds"], strict=True):
        assert record["events"] == ["train", "validation"] * 3 + ["test"]
        assert fold["selected_epoch"] == 1
        assert [epoch["training_mode_during_updates"] for epoch in fold["epoch_history"]] == (
            [True, False, False] if preserve_source_mode else [True, True, True]
        )
        for calls in (record["dropout_calls"], record["batch_norm_calls"]):
            assert {epoch for epoch, phase, _ in calls if phase == "train"} == {1, 2, 3}
            assert {epoch for epoch, phase, _ in calls if phase == "validation"} == {1, 2, 3}
            assert any(phase == "test" for _, phase, _ in calls)
            assert all(not mode for _, phase, mode in calls if phase != "train")


def test_c_replays_deterministically_and_keeps_b_splits(
    validation_dataset, validation_config, tmp_path,
):
    """All C folds replay exactly apart from wall-clock fields; B splits match."""
    b_config = deepcopy(validation_config)
    validation_config["training"]["preserve_eval_mode_after_first_evaluation"] = False
    first = run_cross_validation(validation_dataset, validation_config, tmp_path / "first.json",
                                 log=lambda _: None)
    replay = run_cross_validation(validation_dataset, validation_config, tmp_path / "replay.json",
                                  log=lambda _: None)

    def without_timings(value):
        if isinstance(value, dict):
            return {key: without_timings(item) for key, item in value.items()
                    if key not in {"seconds", "elapsed_seconds", "started_at_utc", "finished_at_utc"}}
        if isinstance(value, list):
            return [without_timings(item) for item in value]
        return value

    assert without_timings(first) == without_timings(replay)
    assert first["status"] == "complete"
    assert first["protocol"]["source_preserve_eval_mode"] is False
    baseline = run_cross_validation(validation_dataset, b_config, tmp_path / "b.json", log=lambda _: None)
    for c_fold, b_fold in zip(first["folds"], baseline["folds"], strict=True):
        for key in ("outer_train_gene_ids", "train_gene_ids", "validation_gene_ids", "test_gene_ids",
                    "outer_train_counts", "train_counts", "validation_counts", "test_counts", "class_weights"):
            assert c_fold[key] == b_fold[key]
        assert all(epoch["training_mode_during_updates"] for epoch in c_fold["epoch_history"])
        assert [epoch["training_mode_during_updates"] for epoch in b_fold["epoch_history"]] == [True, False, False]


def test_c_config_changes_only_training_mode_and_output():
    """The published comparison must not silently change data or hyperparameters."""
    root = Path(__file__).parents[1]
    b_config = load_config(root / "configs/brca_validation.yaml")
    c_config = load_config(root / "configs/brca_validation_train_mode.yaml")
    assert b_config["training"]["preserve_eval_mode_after_first_evaluation"] is True
    assert c_config["training"]["preserve_eval_mode_after_first_evaluation"] is False
    assert c_config["output_path"] == "results/brca_validation_train_mode_metrics.json"
    normalized = deepcopy(c_config)
    normalized["training"]["preserve_eval_mode_after_first_evaluation"] = True
    normalized["output_path"] = b_config["output_path"]
    assert normalized == b_config
