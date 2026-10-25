"""Publication snapshots validate report consistency without scientific dependencies."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.export_experiment_results import (
    METRICS, PROJECT_ROOT, REPORTS, assert_public_snapshot, build_snapshot, export_all,
)


PUBLISHED = PROJECT_ROOT / "results/published"
PERFECT = {metric: 1.0 for metric in METRICS}
SUMMARY = {metric: {"mean": 1.0, "std": 0.0} for metric in METRICS}
ZERO = {metric: 0.0 for metric in METRICS}


def synthetic_full_report(experiment="b"):
    """Full-sized synthetic IDs exercise coverage checks without original data."""
    snapshot = json.loads((PUBLISHED / f"{experiment}.json").read_text())
    provenance = deepcopy(snapshot["provenance"])
    config = provenance.pop("effective_config")
    provenance.pop("seed")
    tuning = experiment == "d"
    if tuning:
        provenance.update(format_version=1, base_config=config)
    total = snapshot["num_targets"]
    ids = [f"synthetic_{index}" for index in range(total)]
    test_ranges, offset = [], 0
    for fold in snapshot["folds"]:
        size = sum(fold["test_counts"].values())
        test_ranges.append(ids[offset:offset + size])
        offset += size
    positive = {gene for block in test_ranges for gene in block[:14]}

    def records(genes):
        return [{"gene_id": gene, "label": int(gene in positive),
                 "driver_probability": 0.9 if gene in positive else 0.1} for gene in genes]

    def history(validation, always_train):
        return [{"epoch": epoch, "training_mode_during_updates": always_train or epoch == 1,
                 "validation_metrics" if validation else "metrics": deepcopy(PERFECT),
                 "mean_batch_loss": 0.2, "learning_rate": 0.001} for epoch in range(1, 11)]

    folds = []
    for template, test_ids in zip(snapshot["folds"], test_ranges):
        split = {key: deepcopy(value) for key, value in template.items() if key.endswith("_counts")}
        excluded_test = set(test_ids)
        outer = [gene for gene in ids if gene not in excluded_test]
        split["test_gene_ids"] = test_ids
        use_validation = experiment in ("b", "c", "d")
        if use_validation:
            val_positive = [gene for gene in outer if gene in positive][:7]
            val_negative = [gene for gene in outer if gene not in positive][:split["validation_counts"]["non_driver"]]
            split["validation_gene_ids"] = val_positive + val_negative
            split["outer_train_gene_ids"] = outer
            excluded = set(split["validation_gene_ids"])
            split["train_gene_ids"] = [gene for gene in outer if gene not in excluded]
        else:
            split["train_gene_ids"] = outer
        fold = {"fold": template["fold"], "class_weights": deepcopy(template["class_weights"])}
        if not tuning:
            fold.update(split)
            fold.update(selected_epoch=1, metrics=deepcopy(PERFECT), predictions=records(test_ids),
                        epoch_history=history(use_validation, experiment == "c"))
            if use_validation:
                fold.update(selected_validation_metrics=deepcopy(PERFECT),
                            validation_predictions=records(split["validation_gene_ids"]))
        else:
            trials = []
            for trial in template["validation_candidates"]:
                trials.append(dict(deepcopy(trial), selected_epoch=1, selected_validation_metrics=deepcopy(PERFECT),
                                   validation_predictions=records(split["validation_gene_ids"]),
                                   epoch_history=history(True, True)))
            final = {key: trials[0][key] for key in ("candidate_id", "learning_rate", "dropout_pair", "selected_epoch")}
            final.update(metrics=deepcopy(PERFECT), predictions=records(test_ids))
            fold.update(status="complete", splits=split, candidate_results=trials, selected_candidate_id="candidate_01",
                        provenance=dict(deepcopy(provenance), fold=fold["fold"]), seeds=deepcopy(template["seeds"]),
                        tuned_test=deepcopy(final), fixed_c_test=deepcopy(final), test_evaluation_count=1,
                        protocol=deepcopy(snapshot["protocol"]), started_at_utc="2026-10-25T00:00:00+00:00",
                        finished_at_utc="2026-10-25T00:01:00+00:00", elapsed_seconds=60)
        folds.append(fold)
    if not tuning:
        return dict(provenance, status="complete", cancer="BRCA", config=config,
                    num_targets=total, dataset_counts=snapshot["dataset_counts"], folds=folds,
                    protocol=snapshot["protocol"], summary=deepcopy(SUMMARY),
                    started_at_utc="2026-10-25T00:00:00+00:00", finished_at_utc="2026-10-25T00:01:00+00:00",
                    elapsed_seconds=60)
    return {"status": "complete", "experiment": "D", "provenance": provenance, "std_ddof": 0,
            "folds": folds, "tuned_summary": deepcopy(SUMMARY), "fixed_c_summary": deepcopy(SUMMARY),
            "paired_differences": [{"fold": index, "metrics": deepcopy(ZERO)} for index in (1, 2, 3)],
            "paired_difference_summary": {metric: {"mean": 0.0, "std": 0.0} for metric in METRICS},
            "interpretation": "Synthetic consistency-test fixture, not experimental evidence."}


@pytest.fixture(scope="module")
def synthetic_b():
    return synthetic_full_report("b")


def test_snapshot_retains_metrics_and_sanitizes_machine_paths(synthetic_b):
    report = deepcopy(synthetic_b)
    report["config"]["data"]["upstream_root"] = "/private/research/input"
    report["config"]["data"]["features"] = "C:\\Users\\researcher\\BRCA.csv"
    report["config"]["output_path"] = "/private/output/full.json"
    for item in report["input_files"]:
        item["path"] = "/Users/example/secret/location.csv"
    raw = json.dumps(report).encode()
    snapshot = build_snapshot("b", report, hashlib.sha256(raw).hexdigest())
    assert snapshot["summary"] == report["summary"]
    assert snapshot["folds"][0]["test_metrics"] == report["folds"][0]["metrics"]
    assert snapshot["provenance"]["effective_config"]["data"]["upstream_root"] == "input:upstream_root"
    assert snapshot["provenance"]["effective_config"]["data"]["features"] == "input:features"
    assert snapshot["provenance"]["effective_config"]["output_path"] == "output:raw_report"
    assert snapshot["export"]["raw_report_sha256"] == hashlib.sha256(raw).hexdigest()
    assert "no independent training replay" in snapshot["export"]["validation_scope"]
    assert_public_snapshot(snapshot)


@pytest.mark.parametrize("mutation", ["incomplete", "two_folds", "smoke", "nonfinite", "mean", "std", "ids", "epoch"])
def test_rejects_incomplete_or_inconsistent_reports(synthetic_b, mutation):
    report = deepcopy(synthetic_b)
    if mutation == "incomplete":
        report["status"] = "running"
    elif mutation == "two_folds":
        report["folds"].pop()
    elif mutation == "smoke":
        report["config"]["training"]["epochs"] = 1
    elif mutation == "nonfinite":
        report["folds"][0]["metrics"]["auc"] = float("nan")
    elif mutation in ("mean", "std"):
        report["summary"]["auc"][mutation] = 0.5
    elif mutation == "ids":
        report["folds"][1]["test_gene_ids"] = report["folds"][0]["test_gene_ids"]
    elif mutation == "epoch":
        report["folds"][0]["selected_epoch"] = 2
    with pytest.raises(ValueError):
        build_snapshot("b", report, "a" * 64)


def test_d_requires_full_search_and_correct_paired_aggregates():
    report = synthetic_full_report("d")
    snapshot = build_snapshot("d", report, "d" * 64)
    assert len(snapshot["folds"][0]["validation_candidates"]) == 6
    assert snapshot["folds"][0]["tuned_test"]["dropout_pair"] == [0.7, 0.6]
    assert snapshot["paired_difference_summary"]["auc"] == {"mean": 0.0, "std": 0.0}
    for mutation in ("incomplete_fold", "missing_trial", "paired_difference", "control_summary"):
        changed = deepcopy(report)
        if mutation == "incomplete_fold":
            changed["folds"][2]["status"] = "running"
        elif mutation == "missing_trial":
            changed["folds"][0]["candidate_results"].pop()
        elif mutation == "paired_difference":
            changed["paired_differences"][0]["metrics"]["auc"] = 0.1
        else:
            changed["fixed_c_summary"]["auc"]["mean"] = 0.9
        with pytest.raises(ValueError):
            build_snapshot("d", changed, "d" * 64)


def test_export_validates_every_report_before_writing(tmp_path):
    raw_dir, output = tmp_path / "raw", tmp_path / "published"
    for experiment, filename in REPORTS.items():
        report = synthetic_full_report(experiment)
        if experiment == "d":
            report["status"] = "failed"
        path = raw_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Only complete"):
        export_all(raw_dir, output)
    assert not output.exists()
    report = synthetic_full_report("d")
    (raw_dir / REPORTS["d"]).write_text(json.dumps(report))
    before = {name: hashlib.sha256((raw_dir / name).read_bytes()).hexdigest() for name in REPORTS.values()}
    files = export_all(raw_dir, output)
    assert len(files) == 5
    for name, digest in before.items():
        assert hashlib.sha256((raw_dir / name).read_bytes()).hexdigest() == digest
    for path in files:
        assert_public_snapshot(json.loads(path.read_text()))


def test_committed_public_snapshots_are_compact_safe_and_complete():
    assert {path.name for path in PUBLISHED.glob("*.json")} == {f"{name}.json" for name in REPORTS}
    for experiment in REPORTS:
        path = PUBLISHED / f"{experiment}.json"
        snapshot = json.loads(path.read_text())
        assert_public_snapshot(snapshot)
        assert snapshot["schema_version"] == 1
        assert snapshot["status"] == "complete"
        assert len(snapshot["folds"]) == 3
        assert path.stat().st_size < 50_000
        assert len(snapshot["export"]["raw_report_sha256"]) == 64
        assert snapshot["num_targets"] == (15028 if experiment == "original" else 16786)
        for item in snapshot["provenance"]["input_files"]:
            assert set(item) == {"role", "bytes", "sha256"}


@pytest.mark.parametrize("payload", [
    {"predictions": []}, {"train_gene_ids": []}, {"anything": "/Users/name/private.csv"},
    {"anything": "C:\\Users\\name\\private.csv"}, {"anything": "~/private.csv"},
])
def test_public_guard_blocks_unintended_payloads(payload):
    with pytest.raises(ValueError):
        assert_public_snapshot(payload)
