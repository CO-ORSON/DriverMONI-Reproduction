"""Meaningful validation tests for saved results, independent of model execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from src.app_results import (
    METRIC_NAMES, export_summary, gene_prediction, load_experiment_summary,
    load_report, validate_experiment,
)


def _report(*, experiment: str = "B") -> dict:
    """Three perfectly classified outer splits, each with one driver/non-driver."""
    metrics = dict.fromkeys(METRIC_NAMES, 1.0)
    ids = [f"ENSG{index:011d}" for index in range(6)]
    report = {
        "status": "complete", "cancer": "BRCA", "device": "cpu",
        "started_at_utc": "2026-10-23T00:00:00+00:00",
        "finished_at_utc": "2026-10-23T00:01:00+00:00", "elapsed_seconds": 60,
        "runtime": {"python": "3.12", "packages": {}},
        "num_targets": 6, "dataset_counts": {"driver": 3, "non_driver": 3},
        "config": {
            "seed": 42, "data": {"upstream_root": "/private/local-data", "features": "/private/data.csv", "use_comparison_filter": False, "use_supplied_neighbors": False},
            "training": {"epochs": 1, "preserve_eval_mode_after_first_evaluation": experiment != "C"},
            "cross_validation": {"n_splits": 3, "stratified": True, "shuffle": True, "validation_fraction": .25, "epoch_selection": "best_test_roc_auc" if experiment == "A" else "best_validation_roc_auc"},
            "evaluation": {"probability_threshold": 0.5, "positive_class": 1, "zero_division": 0},
            "output_path": "/private/result.json",
        },
        "input_files": [{"role": "features", "path": "/private/data.csv", "sha256": "a" * 64, "bytes": 100}],
        "protocol": {"std_ddof": 0, "test_fold_used_for_epoch_selection": experiment == "A",
                     "source_preserve_eval_mode": experiment != "C", "validation_fraction": .25,
                     "test_evaluations_per_fold": 1, "refit_on_outer_training": False,
                     "class_weight_population": "inner training targets only"},
        "summary": {name: {"mean": 1.0, "std": 0.0} for name in METRIC_NAMES},
        "folds": [],
    }
    for fold_id in range(3):
        test_ids = ids[fold_id * 2:fold_id * 2 + 2]
        remaining = [gene for gene in ids if gene not in test_ids]
        val_ids = remaining[:2] if experiment != "A" else []
        train_ids = remaining[2:] if val_ids else remaining
        predictions = [{"gene_id": gene, "label": index, "driver_probability": .1 if index == 0 else .9} for index, gene in enumerate(test_ids)]
        fold = {
            "fold": fold_id + 1, "selected_epoch": 1, "metrics": metrics.copy(),
            "test_gene_ids": test_ids, "train_gene_ids": train_ids,
            "test_counts": {"driver": 1, "non_driver": 1},
            "train_counts": {"driver": len(train_ids) // 2, "non_driver": len(train_ids) // 2},
            "class_weights": {"driver": 1.0, "non_driver": 1.0},
            "predictions": predictions,
            "epoch_history": [{"epoch": 1, "training_mode_during_updates": True, "metrics" if experiment == "A" else "validation_metrics": metrics.copy()}],
        }
        if val_ids:
            fold.update({
                "validation_gene_ids": val_ids, "validation_counts": {"driver": 1, "non_driver": 1},
                "selected_validation_metrics": metrics.copy(),
                "validation_predictions": [{"gene_id": gene, "label": index, "driver_probability": .1 if index == 0 else .9} for index, gene in enumerate(val_ids)],
            })
        report["folds"].append(fold)
    return report


def _raw(report: dict) -> bytes:
    return json.dumps(report).encode()


def test_load_full_report_and_gene_lookup() -> None:
    report = load_report(_raw(_report()), expected_feature_sha256="a" * 64)
    prediction = gene_prediction(report, " ENSG00000000001 ")
    assert prediction == {
        "gene_id": "ENSG00000000001", "label": 1, "driver_probability": .9,
        "fold": 1, "selected_epoch": 1, "threshold": .5, "predicted_class": 1,
        "test_fold_used_for_epoch_selection": False, "source": "recorded_outer_test_prediction",
    }
    assert gene_prediction(report, "unknown") is None
    original = load_report(_raw(_report(experiment="A")))
    assert gene_prediction(original, "ENSG00000000001")["test_fold_used_for_epoch_selection"] is True


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d.update(status="running"), "complete BRCA"),
    (lambda d: d.update(smoke_test={"targets": 6}), "Smoke-test"),
    (lambda d: d["folds"].pop(), "three-fold"),
    (lambda d: d["folds"][0]["epoch_history"].clear(), "incomplete epoch"),
    (lambda d: d["folds"][0]["predictions"][0].update(driver_probability=float("nan")), "finite"),
    (lambda d: d["folds"][0]["predictions"][0].update(driver_probability=.95), "does not match"),
    (lambda d: d["summary"]["auc"].update(mean=.9), "does not match"),
    (lambda d: d["folds"][0]["predictions"].append(d["folds"][0]["predictions"][0].copy()), "duplicate"),
    (lambda d: d["folds"][0]["train_gene_ids"].__setitem__(0, d["folds"][0]["test_gene_ids"][0]), "overlap"),
    (lambda d: d["protocol"].update(test_fold_used_for_epoch_selection=True), "disagree"),
])
def test_reject_incomplete_or_tampered_report(mutation, match: str) -> None:
    report = _report()
    mutation(report)
    with pytest.raises(ValueError, match=match):
        load_report(_raw(report))


def test_reject_wrong_feature_dataset() -> None:
    with pytest.raises(ValueError, match="dataset SHA-256"):
        load_report(_raw(_report()), expected_feature_sha256="b" * 64)


def test_export_roundtrip_is_path_free_and_prediction_free(tmp_path: Path) -> None:
    paths = {}
    for name in ("A", "B", "C"):
        path = tmp_path / f"{name}.json"
        path.write_bytes(_raw(_report(experiment=name)))
        paths[name] = path
    output = tmp_path / "snapshot.json"
    exported = export_summary(paths, output)
    assert load_experiment_summary(output) == exported
    serialized = output.read_text()
    assert "/private/" not in serialized
    assert '"predictions":' not in serialized
    assert "ENSG" not in serialized
    assert len(exported["experiments"]) == 3
    for entry in exported["experiments"]:
        assert entry["report_sha256"] == hashlib.sha256(paths[entry["experiment"]].read_bytes()).hexdigest()
    assert exported["experiments"][0]["protocol"]["test_fold_used_for_epoch_selection"] is True


def test_export_rejects_different_datasets_and_mislabeled_experiments(tmp_path: Path) -> None:
    b, c = _report(), _report(experiment="C")
    c["input_files"][0]["sha256"] = "b" * 64
    bpath, cpath = tmp_path / "b.json", tmp_path / "c.json"
    bpath.write_bytes(_raw(b))
    cpath.write_bytes(_raw(c))
    with pytest.raises(ValueError, match="different feature/label/PPI"):
        export_summary({"B": bpath, "C": cpath}, tmp_path / "out.json")
    with pytest.raises(ValueError, match="epoch selection"):
        export_summary({"A": bpath}, tmp_path / "out.json")
    with pytest.raises(ValueError, match="training mode"):
        export_summary({"C": bpath}, tmp_path / "out.json")


def test_committed_snapshot_matches_fold_aggregates_without_model_imports() -> None:
    path = Path(__file__).resolve().parents[1] / "app_assets" / "experiment_summary.json"
    snapshot = load_experiment_summary(path)
    assert [item["experiment"] for item in snapshot["experiments"]] == ["A", "B", "C"]
    assert {item["num_targets"] for item in snapshot["experiments"]} == {16786}
    assert {item["dataset_counts"]["driver"] for item in snapshot["experiments"]} == {42}
    assert len({item["feature_sha256"] for item in snapshot["experiments"]}) == 1


def test_summary_rejects_tampering_and_local_paths(tmp_path: Path) -> None:
    report_path = tmp_path / "b.json"
    report_path.write_bytes(_raw(_report()))
    original = export_summary({"B": report_path}, tmp_path / "summary.json")
    changed = deepcopy(original)
    changed["experiments"][0]["summary"]["auc"]["mean"] = .8
    with pytest.raises(ValueError, match="does not match"):
        load_experiment_summary(_raw(changed))
    changed = deepcopy(original)
    changed["experiments"][0]["private"] = "/Users/private/location"
    with pytest.raises(ValueError, match="absolute paths"):
        load_experiment_summary(_raw(changed))


def test_score_ties_and_inclusive_positive_threshold() -> None:
    report = _report()
    metrics = {"auc": .5, "accuracy": .5, "precision": .5, "recall": 1., "f1": 2 / 3, "balanced_accuracy": .5}
    for fold in report["folds"]:
        for field in ("predictions", "validation_predictions"):
            for record in fold[field]:
                record["driver_probability"] = .5
        fold["metrics"] = metrics.copy()
        fold["selected_validation_metrics"] = metrics.copy()
        fold["epoch_history"][0]["validation_metrics"] = metrics.copy()
    report["summary"] = {key: {"mean": value, "std": 0.} for key, value in metrics.items()}
    loaded = load_report(_raw(report))
    assert gene_prediction(loaded, "ENSG00000000000")["predicted_class"] == 1


@pytest.mark.parametrize("name", ["A", "B", "C"])
def test_experiment_identity_rejects_swapped_reports(name: str) -> None:
    report = load_report(_raw(_report(experiment=name)))
    validate_experiment(report, name)
    for wrong in {"A", "B", "C"} - {name}:
        with pytest.raises(ValueError, match="epoch selection|training mode"):
            validate_experiment(report, wrong)


def test_experiment_identity_rejects_changed_protocol() -> None:
    report = load_report(_raw(_report()))
    report["config"]["cross_validation"]["validation_fraction"] = .1
    with pytest.raises(ValueError, match="25%"):
        validate_experiment(report, "B")
    report = _report()
    report["protocol"]["source_preserve_eval_mode"] = False
    with pytest.raises(ValueError, match="training mode"):
        validate_experiment(report, "B")


def test_experiment_identity_checks_recorded_update_modes() -> None:
    report = _report(experiment="C")
    report["folds"][0]["epoch_history"][0]["training_mode_during_updates"] = False
    with pytest.raises(ValueError, match="Epoch update modes"):
        validate_experiment(report, "C")


def _d_report(base_report: dict | None = None) -> dict:
    """Convert a small self-consistent C fixture into a complete six-choice D run."""
    base = deepcopy(_report(experiment="C") if base_report is None else base_report)
    base["config"]["training"]["preserve_eval_mode_after_first_evaluation"] = False
    base["config"]["evaluation"]["std_ddof"] = 0
    grid = {"learning_rates": [.001, .0003, .0001], "dropout_pairs": [[.7, .6], [.5, .3]], "selection_metric": "auc"}
    provenance = {"format_version": 1, "base_config": base["config"], "search_config": grid,
                  "input_files": base["input_files"], "dataset_sha256": "e" * 64,
                  "source_sha256": {"src/tuning.py": "f" * 64},
                  "runtime": base.get("runtime", {"python": "3.12", "packages": {}}), "device": "cpu"}
    protocol = {"test_fold_used_for_selection": False, "refit_on_outer_training": False,
                "training_mode_every_epoch": True, "class_weight_population": "inner fitting targets only",
                "fixed_c_control": {"learning_rate": .001, "dropout_pair": [.7, .6]}}
    folds = []
    for original in base["folds"]:
        choices = []
        for index, (rate, pair) in enumerate(((r, p) for r in grid["learning_rates"] for p in grid["dropout_pairs"]), 1):
            history = deepcopy(original["epoch_history"])
            for row in history:
                row.update(learning_rate=rate, training_mode_during_updates=True)
            choices.append({"candidate_id": f"candidate_{index:02d}", "learning_rate": rate,
                            "dropout_pair": pair, "selected_epoch": original["selected_epoch"],
                            "selected_validation_metrics": deepcopy(original["selected_validation_metrics"]),
                            "validation_predictions": deepcopy(original["validation_predictions"]),
                            "epoch_history": history, "initial_state_sha256": "b" * 64})
        final = {key: deepcopy(choices[0][key]) for key in ("candidate_id", "learning_rate", "dropout_pair", "selected_epoch")}
        final.update(metrics=deepcopy(original["metrics"]), predictions=deepcopy(original["predictions"]))
        folds.append({"fold": original["fold"], "status": "complete", "provenance": dict(provenance, fold=original["fold"]),
                      "splits": {key: deepcopy(value) for key, value in original.items() if key.endswith(("_counts", "_gene_ids"))},
                      "class_weights": original["class_weights"], "protocol": protocol,
                      "candidate_results": choices, "selected_candidate_id": "candidate_01",
                      "tuned_test": deepcopy(final), "fixed_c_test": deepcopy(final), "test_evaluation_count": 1,
                      "started_at_utc": "2026-10-25T00:00:00+00:00", "finished_at_utc": "2026-10-25T00:01:00+00:00",
                      "elapsed_seconds": 60})
    return {"status": "complete", "experiment": "D", "provenance": provenance, "std_ddof": 0,
            "folds": folds, "tuned_summary": deepcopy(base["summary"]), "fixed_c_summary": deepcopy(base["summary"]),
            "paired_differences": [{"fold": fold["fold"], "metrics": dict.fromkeys(METRIC_NAMES, 0.0)} for fold in folds],
            "paired_difference_summary": {metric: {"mean": 0.0, "std": 0.0} for metric in METRIC_NAMES}}


def test_d_raw_report_validates_trials_control_and_gene_scores():
    source = _d_report()
    loaded = load_report(_raw(source), expected_feature_sha256="a" * 64)
    validate_experiment(loaded, "D")
    score = gene_prediction(loaded, "ENSG00000000001")
    assert score["driver_probability"] == .9
    assert score["candidate_id"] == "candidate_01"
    assert score["learning_rate"] == .001
    assert score["dropout_pair"] == [.7, .6]
    assert score["test_fold_used_for_epoch_selection"] is False
    assert loaded["matched_control"]["summary"] == loaded["summary"]
    with pytest.raises(ValueError, match="tuned experiment D"):
        validate_experiment(loaded, "C")
    with pytest.raises(ValueError, match="requires the recorded"):
        validate_experiment(load_report(_raw(_report(experiment="C"))), "D")


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d["folds"][0]["candidate_results"].pop(), "six completed"),
    (lambda d: d["folds"][0].update(selected_candidate_id="candidate_02"), "winner"),
    (lambda d: d["folds"][0]["candidate_results"][2]["validation_predictions"][0].update(driver_probability=.99), "does not match"),
    (lambda d: d["folds"][0]["fixed_c_test"]["predictions"][0].update(driver_probability=.99), "does not match"),
    (lambda d: d["folds"][0]["candidate_results"][2].update(initial_state_sha256="c" * 64), "paired initial"),
    (lambda d: d["paired_difference_summary"]["auc"].update(mean=.1), "paired mean"),
    (lambda d: d["folds"][0].update(test_evaluation_count=2), "test-evaluation count"),
])
def test_d_raw_rejects_tampered_selection_or_predictions(mutation, match):
    report = _d_report()
    mutation(report)
    with pytest.raises(ValueError, match=match):
        load_report(_raw(report))


def test_original_identity_is_separate_from_updated_data():
    report = _report(experiment="A")
    report["config"]["data"].update(use_comparison_filter=True, use_supplied_neighbors=True)
    report["input_files"][0]["sha256"] = "c" * 64
    loaded = load_report(_raw(report))
    validate_experiment(loaded, "original")
    with pytest.raises(ValueError, match="Original uses"):
        validate_experiment(loaded, "A")
    with pytest.raises(ValueError, match="dataset SHA-256"):
        load_report(_raw(report), expected_feature_sha256="a" * 64)


def test_published_experiments_are_ordered_and_include_original_and_d():
    from src.app_results import EXPERIMENT_ORDER, load_published_experiments
    directory = Path(__file__).resolve().parents[1] / "results/published"
    snapshot = load_published_experiments(directory)
    assert tuple(entry["experiment"] for entry in snapshot["experiments"]) == EXPERIMENT_ORDER
    original, *updated = snapshot["experiments"]
    assert original["num_targets"] == 15028
    assert original["dataset_group"] == "authors_original"
    assert original["feature_sha256"] not in {entry["feature_sha256"] for entry in updated}
    assert len({entry["feature_sha256"] for entry in updated}) == 1
    d = updated[-1]
    assert d["summary"]["auc"]["mean"] == pytest.approx(.92132635870091)
    assert d["matched_control"]["summary"]["auc"]["mean"] == pytest.approx(.8762852945711131)
    assert [(fold["learning_rate"], fold["dropout_pair"]) for fold in d["folds"]] == [
        (.001, [.5, .3]), (.001, [.5, .3]), (.0003, [.7, .6]),
    ]
    assert load_experiment_summary(_raw(snapshot)) == snapshot


@pytest.mark.parametrize("change,match", [
    (lambda d: d["summary"]["auc"].update(mean=.1), "does not match"),
    (lambda d: d["provenance"]["input_files"][0].update(sha256="a" * 64), "different feature/label/PPI"),
    (lambda d: d.update(experiment="original"), "mislabeled"),
    (lambda d: d["folds"][0]["train_counts"].update(driver=20), "counts"),
])
def test_published_summary_rejects_identity_metrics_and_count_tampering(tmp_path, change, match):
    from src.app_results import load_published_experiments
    directory = Path(__file__).resolve().parents[1] / "results/published"
    for path in directory.glob("*.json"):
        data = json.loads(path.read_text())
        if path.name == "b.json":
            change(data)
        (tmp_path / path.name).write_text(json.dumps(data))
    with pytest.raises(ValueError, match=match):
        load_published_experiments(tmp_path)


def test_export_raw_d_and_original_keeps_order_and_distinct_inputs(tmp_path):
    original = _report(experiment="A")
    original["config"]["data"].update(use_comparison_filter=True, use_supplied_neighbors=True)
    original["input_files"][0]["sha256"] = "c" * 64
    first, last = tmp_path / "original.json", tmp_path / "d.json"
    first.write_bytes(_raw(original))
    last.write_bytes(_raw(_d_report()))
    snapshot = export_summary({"D": last, "original": first}, tmp_path / "summary.json")
    assert [entry["experiment"] for entry in snapshot["experiments"]] == ["original", "D"]
    assert load_experiment_summary(_raw(snapshot)) == snapshot
