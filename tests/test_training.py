"""Three-fold protocol, weighted loss, probabilistic AUC, and seeded replay."""

import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.dataset import DriverMONIDataset, GAT_FEATURES, collate_gene_samples
from src.evaluate import classification_metrics, evaluate
from src.graph_utils import PPIGraph
from src.models import DriverMONI
from src.train import class_weights, run_cross_validation, train_one_epoch, weighted_nll
from src.utils import load_config, seed_everything, validate_config


@pytest.fixture
def small_dataset():
    genes = [f"gene_{index}" for index in range(12)]
    rows = []
    for index, gene in enumerate(genes):
        row = {feature: float(np.sin(index + offset)) for offset, feature in enumerate(GAT_FEATURES)}
        row.update(gene_ens=gene, gene_symbol=gene, label=index % 2)
        rows.append(row)
    table = pd.DataFrame(rows)
    edges = pd.DataFrame({"entrez_id_1": genes, "entrez_id_2": genes[1:] + genes[:1]})
    ppi = PPIGraph(edges, genes)
    return DriverMONIDataset(table, ppi, ppi.discover_neighbors(genes))


@pytest.fixture
def small_config():
    config = load_config(Path(__file__).parents[1] / "configs/brca.yaml")
    config["device"] = "cpu"
    config["training"]["epochs"] = 3
    config["training"]["batch_size"] = 4
    return config


def test_auc_uses_scores_and_threshold_is_inclusive():
    metrics = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4])
    assert metrics["auc"] == 1.0  # All hard predictions are 0, but ranking is perfect.
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0
    assert classification_metrics([0, 1], [0.4, 0.5])["accuracy"] == 1.0


def test_class_weights_and_weighted_nll():
    weights = class_weights(np.array([0, 0, 0, 1]))
    torch.testing.assert_close(weights, torch.tensor([2 / 3, 2], dtype=torch.float64))
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0]], requires_grad=True)
    labels = torch.tensor([0, 1])
    expected = -(weights[0] * torch.log_softmax(logits.double(), 1)[0, 0]
                 + weights[1] * torch.log_softmax(logits.double(), 1)[1, 1]) / weights.sum()
    torch.testing.assert_close(weighted_nll(logits, labels, weights), expected)
    assert weighted_nll(logits, labels, weights).dtype == torch.float64


def test_eval_mode_updates_parameters_without_updating_batchnorm(small_dataset):
    loader = DataLoader(small_dataset, batch_size=4, collate_fn=collate_gene_samples)
    model = DriverMONI()
    result = evaluate(model, loader, "cpu")
    assert len(result.predictions) == 12
    assert not model.training
    weights_before = model.classifier.weight.detach().clone()
    mean_before = model.dnn.batch_norm1.running_mean.clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_one_epoch(model, loader, optimizer, torch.ones(2, dtype=torch.float64), "cpu")
    assert not torch.equal(model.classifier.weight, weights_before)
    torch.testing.assert_close(model.dnn.batch_norm1.running_mean, mean_before)
    assert model.dnn.batch_norm1.num_batches_tracked == 0


def test_complete_cv_counts_selection_seeds_and_json(small_dataset, small_config, tmp_path, monkeypatch):
    models = []
    def make_model():
        model = DriverMONI()
        models.append(model)
        return model
    monkeypatch.setattr("src.train.DriverMONI", make_model)
    output = tmp_path / "metrics.json"
    report = run_cross_validation(small_dataset, small_config, output, log=lambda _: None)
    assert json.loads(output.read_text()) == report
    assert report["status"] == "complete"
    assert len(report["folds"]) == len(models) == 3
    held_out = []
    for model, fold in zip(models, report["folds"]):
        assert fold["train_counts"] == {"driver": 4, "non_driver": 4}
        assert fold["test_counts"] == {"driver": 2, "non_driver": 2}
        assert not set(fold["train_gene_ids"]) & set(fold["test_gene_ids"])
        assert len(fold["predictions"]) == 4
        assert {p["gene_id"] for p in fold["predictions"]} == set(fold["test_gene_ids"])
        history = fold["epoch_history"]
        assert [epoch["training_mode_during_updates"] for epoch in history] == [True, False, False]
        assert [epoch["learning_rate"] for epoch in history] == [0.001] * 3
        assert int(model.dnn.batch_norm1.num_batches_tracked) == 2  # First epoch only.
        best_index = int(np.argmax([epoch["metrics"]["auc"] for epoch in history]))
        assert fold["selected_epoch"] == best_index + 1
        assert fold["metrics"] == history[best_index]["metrics"]
        held_out.extend(fold["test_gene_ids"])
    assert sorted(held_out) == sorted(small_dataset.target_gene_ids)
    for metric, summary in report["summary"].items():
        values = [fold["metrics"][metric] for fold in report["folds"]]
        assert summary == {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0))}
    replay = run_cross_validation(small_dataset, small_config, tmp_path / "replay.json", log=lambda _: None)
    assert replay["summary"] == report["summary"]
    for first, second in zip(report["folds"], replay["folds"]):
        for key in ["train_gene_ids", "test_gene_ids", "predictions", "selected_epoch", "metrics"]:
            assert first[key] == second[key]
        assert [e["mean_batch_loss"] for e in first["epoch_history"]] == [e["mean_batch_loss"] for e in second["epoch_history"]]


def test_explicit_mode_correction_is_supported_and_recorded(small_dataset, small_config, tmp_path):
    small_config["training"]["preserve_eval_mode_after_first_evaluation"] = False
    result = run_cross_validation(small_dataset, small_config, tmp_path / "corrected.json", log=lambda _: None)
    assert not result["protocol"]["source_preserve_eval_mode"]
    assert all(e["training_mode_during_updates"] for fold in result["folds"] for e in fold["epoch_history"])


def test_seed_all_cpu_rngs():
    seed_everything(43, True, 1)
    expected = (random.random(), np.random.rand(), torch.rand(2))
    seed_everything(43, True, 1)
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(2), expected[2], rtol=0, atol=0)


def test_reject_unsupported_scheduler(small_config):
    small_config["training"]["scheduler"] = "cosine"
    with pytest.raises(ValueError, match="scheduler"):
        validate_config(small_config)


def test_failure_does_not_claim_complete(small_dataset, small_config, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("controlled failure")
    monkeypatch.setattr("src.train.train_one_epoch", fail)
    output = tmp_path / "failure.json"
    with pytest.raises(RuntimeError, match="controlled failure"):
        run_cross_validation(small_dataset, small_config, output, log=lambda _: None)
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and report["folds"] == []
