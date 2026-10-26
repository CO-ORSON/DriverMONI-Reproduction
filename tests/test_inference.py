"""Export parity, safe model loading, data binding and real inference tests."""

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest
import torch

from src.dataset import collate_gene_samples
from src.inference import (
    _json_digest, _state_digest, load_inference_bundle, load_inference_dataset,
    predict_gene, prepare_demo_checkpoint,
)
from src.models import DriverMONI
from src.train import run_cross_validation
from src.utils import input_manifest
from tests.test_validation_training import validation_config, validation_dataset


@pytest.fixture
def exported_demo(validation_config, validation_dataset, tmp_path):
    """Build real input files and a small real trained checkpoint, without mocks."""
    data = tmp_path / "inputs"
    data.mkdir()
    validation_dataset.gene_table.drop(columns="label").to_csv(data / "features.csv", index=False)
    positives = validation_dataset.gene_table.query("label == 1")["gene_ens"].tolist()
    pd.DataFrame({"gene_label": positives}).to_csv(data / "labels.csv", index=False)
    validation_dataset.ppi.edge_rows.to_csv(data / "ppi.csv", index=False)
    validation_config["data"].update(upstream_root=str(data), features="features.csv", labels="labels.csv",
                                     ppi_edges="ppi.csv", use_comparison_filter=False, use_supplied_neighbors=False)
    dataset = load_inference_dataset(validation_config)
    checkpoint = tmp_path / "demo.pt"
    metadata = prepare_demo_checkpoint(dataset, validation_config, checkpoint,
                                       input_files=input_manifest(validation_config["data"]), log=lambda _: None)
    return checkpoint, metadata, dataset, validation_config


@pytest.mark.parametrize("preserve_source_mode", [True, False], ids=["B", "C"])
def test_export_matches_first_fold_cv_exactly(
    exported_demo, tmp_path, monkeypatch, preserve_source_mode,
):
    """Same seed/inputs must match weights, BN buffers, losses and all predictions."""
    checkpoint, _, dataset, config = exported_demo
    config["training"]["preserve_eval_mode_after_first_evaluation"] = preserve_source_mode
    models = []

    def observed_model():
        model = DriverMONI()
        models.append(model)
        return model

    monkeypatch.setattr("src.train.DriverMONI", observed_model)
    report = run_cross_validation(dataset, config, tmp_path / "cv.json", log=lambda _: None)
    metadata = prepare_demo_checkpoint(dataset, config, checkpoint,
                                       input_files=input_manifest(config["data"]), log=lambda _: None)
    bundle = load_inference_bundle(checkpoint)
    reference = report["folds"][0]
    assert metadata["experiment"] == ("B" if preserve_source_mode else "C")
    for key in ("train_gene_ids", "validation_gene_ids", "test_gene_ids", "outer_train_gene_ids",
                "class_weights", "selected_epoch", "selected_validation_metrics", "validation_predictions"):
        assert metadata[key] == reference[key]
    assert metadata["test_metrics"] == reference["metrics"]
    assert metadata["test_predictions"] == reference["predictions"]
    for exported_epoch, cv_epoch in zip(metadata["epoch_history"], reference["epoch_history"], strict=True):
        assert exported_epoch == {key: value for key, value in cv_epoch.items() if key != "seconds"}
    for name, value in bundle.model.state_dict().items():
        torch.testing.assert_close(value, models[0].state_dict()[name], rtol=0, atol=0)


def test_round_trip_predicts_real_logits_and_preserves_eval_buffers(exported_demo, monkeypatch):
    checkpoint, metadata, _, _ = exported_demo
    original_load = torch.load
    load_arguments = []

    def observed_load(*args, **kwargs):
        load_arguments.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr("src.inference.torch.load", observed_load)
    bundle = load_inference_bundle(checkpoint)
    assert load_arguments == [{"map_location": "cpu", "weights_only": True}]
    assert bundle.device == torch.device("cpu")
    assert torch.get_num_threads() == metadata["config"]["reproducibility"]["cpu_threads"]
    assert not bundle.model.training
    assert json.loads(checkpoint.with_suffix(".json").read_text())["selected_epoch"] == metadata["selected_epoch"]
    before = deepcopy(bundle.model.state_dict())
    for split, key in (("fit", "train_gene_ids"), ("validation", "validation_gene_ids"),
                       ("held_out_test", "test_gene_ids")):
        gene = metadata[key][0]
        result = predict_gene(bundle, gene)
        sample = bundle.dataset[bundle.dataset.target_gene_ids.index(gene)]
        batch = collate_gene_samples([sample])
        with torch.inference_mode():
            expected_logits = bundle.model(batch.gene_features, batch.graph)[0]
        assert result["logits"] == expected_logits.tolist()
        assert result["driver_score"] == torch.softmax(expected_logits, dim=0)[1].item()
        assert result["split"] == split
        assert result["num_nodes"] == sample.graph.num_nodes()
        assert result["num_edges"] == sample.graph.num_edges()
        assert len(result["dnn_features"]) == 12
        assert len(result["target_node_features"]) == 13
        assert result["predicted_class"] == int(result["driver_score"] >= 0.5)
        assert result == predict_gene(bundle, gene)
    for name, value in bundle.model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="not a target"):
        predict_gene(bundle, "unknown_gene")


@pytest.mark.parametrize("filename", ["features.csv", "labels.csv", "ppi.csv"])
def test_changed_input_file_is_rejected(exported_demo, filename):
    checkpoint, _, _, config = exported_demo
    source = Path(config["data"]["upstream_root"]) / filename
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="Input files do not match checkpoint hashes"):
        load_inference_bundle(checkpoint)


def test_identical_data_can_be_relocated_and_missing_input_is_clear(exported_demo, tmp_path):
    checkpoint, _, _, config = exported_demo
    relocated = tmp_path / "relocated"
    shutil.copytree(config["data"]["upstream_root"], relocated)
    bundle = load_inference_bundle(checkpoint, data_root=relocated)
    assert bundle.dataset.target_gene_ids
    (relocated / "labels.csv").unlink()
    with pytest.raises(FileNotFoundError, match="labels.csv"):
        load_inference_bundle(checkpoint, data_root=relocated)


@pytest.mark.parametrize("corruption", ["weights", "metadata", "schema", "shape", "dtype", "split", "dataset", "config"])
def test_bad_checkpoint_is_rejected(exported_demo, corruption):
    checkpoint, _, _, _ = exported_demo
    payload = torch.load(checkpoint, weights_only=True)
    expected = "integrity"
    if corruption == "weights":
        payload["state_dict"]["classifier.bias"][0] += 1
    elif corruption == "metadata":
        payload["metadata"]["selected_epoch"] += 1
    elif corruption == "schema":
        payload["metadata"]["feature_schema"]["dnn"].reverse()
        payload["metadata_sha256"] = _json_digest(payload["metadata"])
        expected = "schema/order"
    elif corruption == "shape":
        payload["state_dict"]["classifier.weight"] = torch.zeros(3, 96)
        payload["state_sha256"] = _state_digest(payload["state_dict"])
        expected = "tensor shapes"
    elif corruption == "dtype":
        payload["state_dict"]["classifier.weight"] = payload["state_dict"]["classifier.weight"].double()
        payload["state_sha256"] = _state_digest(payload["state_dict"])
        expected = "tensor dtype"
    elif corruption == "split":
        payload["metadata"]["test_gene_ids"].reverse()
        payload["metadata_sha256"] = _json_digest(payload["metadata"])
        expected = "split"
    elif corruption == "dataset":
        payload["metadata"]["dataset_sha256"] = "0" * 64
        payload["metadata_sha256"] = _json_digest(payload["metadata"])
        expected = "Dataset features"
    else:
        del payload["metadata"]["config"]["reproducibility"]
        payload["metadata_sha256"] = _json_digest(payload["metadata"])
        expected = "Missing configuration sections"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match=expected):
        load_inference_bundle(checkpoint)


def test_export_rejects_test_selected_config(exported_demo, tmp_path):
    _, _, dataset, config = exported_demo
    config["cross_validation"]["epoch_selection"] = "best_test_roc_auc"
    del config["cross_validation"]["validation_fraction"]
    with pytest.raises(ValueError, match="requires validation-based"):
        prepare_demo_checkpoint(dataset, config, tmp_path / "no.pt", input_files=[], log=lambda _: None)


@pytest.mark.parametrize("contents", [b"", b"not a checkpoint"])
def test_unreadable_checkpoint_has_actionable_error(tmp_path, contents):
    checkpoint = tmp_path / "invalid.pt"
    checkpoint.write_bytes(contents)
    with pytest.raises(ValueError, match="could not be read safely"):
        load_inference_bundle(checkpoint)
