"""Validation-selected D import: full state, provenance, scores and RNG isolation."""

from copy import deepcopy
import hashlib
import json

import pytest
import torch

from src.inference import (
    _json_digest, load_inference_bundle, predict_gene, prepare_tuned_demo_checkpoint,
)
from src.models import DriverMONI
from src.tuning import run_tuning_fold
from src.utils import input_manifest
from tests.test_inference import exported_demo
from tests.test_validation_training import validation_config, validation_dataset


@pytest.fixture
def tuned_inputs(exported_demo, tmp_path):
    b_checkpoint, _, dataset, config = exported_demo
    config = deepcopy(config)
    config["training"].update(epochs=2, preserve_eval_mode_after_first_evaluation=False)
    search = {"learning_rates": [0.001], "dropout_pairs": [[0.7, 0.6], [0.5, 0.3]], "selection_metric": "auc"}
    directory = tmp_path / "fold_1"
    report = run_tuning_fold(dataset, config, search, 1, directory,
                             input_files=input_manifest(config["data"]), log=lambda _: None)
    return directory / "report.json", report, tmp_path / "demo_d.pt", b_checkpoint


def test_d_import_replays_saved_scores_without_training_or_rng_changes(tuned_inputs, monkeypatch):
    report_path, report, output, b_checkpoint = tuned_inputs
    b_hash = hashlib.sha256(b_checkpoint.read_bytes()).hexdigest()

    def never_train(*args, **kwargs):
        pytest.fail("D demo import must never train")

    monkeypatch.setattr("src.inference.train_one_epoch", never_train)
    monkeypatch.setattr("src.tuning.train_one_epoch", never_train)
    rng = torch.random.get_rng_state().clone()
    metadata = prepare_tuned_demo_checkpoint(report_path, output, log=lambda _: None)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    bundle = load_inference_bundle(output)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    winner = max(report["candidate_results"], key=lambda row: row["selected_validation_metrics"]["auc"])
    assert metadata["experiment"] == "D"
    assert metadata["selected_hyperparameters"]["candidate_id"] == winner["candidate_id"]
    assert metadata["selected_epoch"] == winner["selected_epoch"]
    assert [bundle.model.dnn.dropout1.p, bundle.model.dnn.dropout2.p] == winner["dropout_pair"]
    assert (DriverMONI().dnn.dropout1.p, DriverMONI().dnn.dropout2.p) == (0.7, 0.6)
    original_state = torch.load(report_path.parent / f"{winner['candidate_id']}.pt", weights_only=True)
    for name, value in bundle.model.state_dict().items():
        torch.testing.assert_close(value, original_state[name], rtol=0, atol=0)
    for name in ("validation", "test"):
        assert metadata["verification"][name]["max_score_difference"] == 0
    for record in report["tuned_test"]["predictions"]:
        result = predict_gene(bundle, record["gene_id"])
        assert result["split"] == "held_out_test"
        assert result["driver_score"] == pytest.approx(record["driver_probability"], abs=1e-7)
    assert hashlib.sha256(b_checkpoint.read_bytes()).hexdigest() == b_hash
    assert load_inference_bundle(b_checkpoint).metadata["experiment"] == "B"


@pytest.mark.parametrize("change", ["winner", "epoch", "source", "seed", "checkpoint", "status"])
def test_d_export_rejects_inconsistent_original_evidence(tuned_inputs, change):
    report_path, report, output, _ = tuned_inputs
    if change == "winner":
        report["selected_candidate_id"] = "candidate_99"
    elif change == "epoch":
        report["candidate_results"][0]["selected_epoch"] = 99
    elif change == "source":
        report["provenance"]["source_sha256"]["src/models/dnn.py"] = "0" * 64
    elif change == "seed":
        report["seeds"]["test_loader"] += 1
    elif change == "status":
        report["status"] = "running"
    else:
        checkpoint = report_path.parent / f"{report['selected_candidate_id']}.pt"
        checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        prepare_tuned_demo_checkpoint(report_path, output, log=lambda _: None)
    assert not output.exists()


@pytest.mark.parametrize("change", ["dropout", "winner", "label", "source", "state"])
def test_d_loader_rejects_rehashed_but_inconsistent_metadata(tuned_inputs, change):
    report_path, _, output, _ = tuned_inputs
    prepare_tuned_demo_checkpoint(report_path, output, log=lambda _: None)
    payload = torch.load(output, weights_only=True)
    metadata = payload["metadata"]
    if change == "dropout":
        metadata["selected_hyperparameters"]["dropout_pair"] = [0.1, 0.1]
    elif change == "winner":
        metadata["tuning_report"]["selected_candidate_id"] = "candidate_99"
    elif change == "label":
        metadata["experiment"] = "C"
    elif change == "source":
        metadata["tuning_report"]["provenance"]["source_sha256"]["src/train.py"] = "0" * 64
    else:
        metadata["selected_state_sha256"] = "0" * 64
    metadata["tuning_report_content_sha256"] = _json_digest(metadata["tuning_report"])
    payload["metadata_sha256"] = _json_digest(metadata)
    torch.save(payload, output)
    with pytest.raises(ValueError):
        load_inference_bundle(output)


def test_b_cannot_be_mislabeled_as_d(exported_demo):
    checkpoint, _, _, _ = exported_demo
    payload = torch.load(checkpoint, weights_only=True)
    payload["metadata"]["experiment"] = "D"
    payload["metadata_sha256"] = _json_digest(payload["metadata"])
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="malformed"):
        load_inference_bundle(checkpoint)


def test_d_export_never_overwrites_existing_checkpoint(tuned_inputs):
    report_path, _, output, b_checkpoint = tuned_inputs
    before = b_checkpoint.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_tuned_demo_checkpoint(report_path, b_checkpoint, log=lambda _: None)
    assert b_checkpoint.read_bytes() == before
    with pytest.raises(FileExistsError):
        prepare_tuned_demo_checkpoint(report_path, output.parent / "brca_demo.pt", log=lambda _: None)
