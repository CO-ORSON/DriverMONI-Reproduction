#!/usr/bin/env python3
"""Publish compact, path-safe summaries of the five completed BRCA experiments.

Uses only the standard library and never changes raw reports. This validates
report consistency and aggregates; it does not independently replay training.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import re
import statistics
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS = ("auc", "accuracy", "precision", "recall", "f1", "balanced_accuracy")
REPORTS = {
    "original": "brca_metrics.json", "a": "brca_updated_data_metrics.json",
    "b": "brca_validation_metrics.json", "c": "brca_validation_train_mode_metrics.json",
    "d": "brca_tuning/summary.json",
}
NAMES = {
    "original": "Original implementation reproduction on authors' processed BRCA data",
    "a": "A: authors' training behavior on updated BRCA data",
    "b": "B: validation epoch selection with authors' train/eval behavior",
    "c": "C: validation epoch selection with training mode every epoch",
    "d": "D: validation-only learning-rate/dropout search with matched fixed-C control",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_tree(value: Any) -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), "Report contains nonfinite numbers")
    elif isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)


def _equal_number(actual: Any, expected: float, name: str) -> None:
    _require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
             and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12),
             f"Inconsistent {name}: recorded {actual!r}, calculated {expected!r}")


def _metrics(metrics: dict[str, float]) -> None:
    _require(set(metrics) == set(METRICS), "Expected all six evaluation metrics")
    _require(all(isinstance(value, (int, float)) and not isinstance(value, bool)
                 and math.isfinite(value) and 0 <= value <= 1 for value in metrics.values()),
             "Metrics must be finite numbers in [0,1]")


def _check_summary(summary: dict, values: list[dict], ddof: int, label: str) -> None:
    _require(set(summary) == set(METRICS), f"Missing metrics in {label}")
    _require(ddof in (0, 1), "Only ddof 0 or 1 is supported")
    for metric in METRICS:
        numbers = [value[metric] for value in values]
        expected_std = statistics.pstdev(numbers) if ddof == 0 else statistics.stdev(numbers)
        _equal_number(summary[metric]["mean"], statistics.fmean(numbers), f"{label}.{metric}.mean")
        _equal_number(summary[metric]["std"], expected_std, f"{label}.{metric}.std")


def _check_predictions(records: list[dict], ids: list[str], counts: dict) -> None:
    _require(len(records) == len(ids) and {row["gene_id"] for row in records} == set(ids),
             "Prediction records must cover the split exactly once")
    _require(all(type(row["label"]) is int and row["label"] in (0, 1) for row in records),
             "Prediction labels must be binary")
    _require(sum(row["label"] for row in records) == counts["driver"], "Prediction label counts disagree")
    _require(all(isinstance(row["driver_probability"], (int, float))
                 and 0 <= row["driver_probability"] <= 1 for row in records), "Invalid driver probability")


def _check_splits(split: dict, use_validation: bool, total: int) -> None:
    names = ("train", "validation", "test", "outer_train") if use_validation else ("train", "test")
    for name in names:
        ids, counts = split[f"{name}_gene_ids"], split[f"{name}_counts"]
        _require(set(counts) == {"driver", "non_driver"}
                 and all(type(value) is int and value > 0 for value in counts.values()),
                 f"Invalid {name} class counts")
        _require(len(ids) == len(set(ids)) == sum(counts.values()), f"Invalid {name} split size")
    train, test = set(split["train_gene_ids"]), set(split["test_gene_ids"])
    _require(not train & test, "Training and test genes overlap")
    if use_validation:
        validation = set(split["validation_gene_ids"])
        _require(not validation & (train | test), "Validation genes overlap training/test")
        _require(train | validation == set(split["outer_train_gene_ids"]), "Invalid outer training split")
        train |= validation
    _require(len(train | test) == total, "Incomplete dataset coverage; smoke runs cannot be published")


def _check_history(item: dict, use_validation: bool, training_mode_every_epoch: bool) -> None:
    history = item["epoch_history"]
    _require(len(history) == 10, "Only full ten-epoch experiments may be published; smoke run rejected")
    metric_key = "validation_metrics" if use_validation else "metrics"
    for epoch, row in enumerate(history, 1):
        _require(row["epoch"] == epoch, "Epoch history is incomplete")
        _require(row["training_mode_during_updates"] is (training_mode_every_epoch or epoch == 1),
                 "Training mode disagrees with the experiment")
        _metrics(row[metric_key])
    best = max(history, key=lambda row: row[metric_key]["auc"])
    _require(item["selected_epoch"] == best["epoch"], "Selected epoch is not the earliest maximum AUC")
    selected = item["selected_validation_metrics"] if use_validation else item["metrics"]
    _require(selected == best[metric_key], "Selected metrics disagree with selected epoch")


def _check_config(config: dict, experiment: str) -> None:
    _require(config["cancer"] == "BRCA", "Only BRCA snapshots are supported")
    _require(config["training"]["epochs"] == 10 and config["training"]["batch_size"] == 32,
             "Only the full ten-epoch/batch-32 experiment may be published")
    _require(config["cross_validation"]["n_splits"] == 3, "Expected three outer folds")
    _require(type(config["seed"]) is int, "A configured seed is required")
    use_validation = experiment in ("b", "c", "d")
    _require(config["cross_validation"]["epoch_selection"] == (
        "best_validation_roc_auc" if use_validation else "best_test_roc_auc"), "Wrong epoch-selection protocol")
    _require(config["training"]["preserve_eval_mode_after_first_evaluation"] is (experiment not in ("c", "d")),
             "Wrong training-mode protocol")
    _require(config["evaluation"]["probability_threshold"] == 0.5, "Expected fixed 0.5 threshold")


def _public_config(config: dict) -> dict:
    result = deepcopy(config)
    for key, value in result["data"].items():
        if isinstance(value, str):
            result["data"][key] = _safe_path(value, f"input:{key}")
    result["output_path"] = _safe_path(result["output_path"], "output:raw_report")
    return result


def _safe_path(value: str, fallback: str) -> str:
    if Path(value).is_absolute():
        try:
            return Path(value).relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return fallback
    if PureWindowsPath(value).is_absolute() or value.startswith("~") or ".." in Path(value).parts:
        return fallback
    return value


def assert_public_snapshot(value: Any) -> None:
    """Reject absolute machine paths and accidentally included per-gene payloads."""
    if isinstance(value, dict):
        for key, item in value.items():
            _require(not key.endswith("gene_ids") and key not in {
                "predictions", "validation_predictions", "epoch_history", "path", "checkpoint_path",
            }, f"Publication contains omitted field: {key}")
            assert_public_snapshot(item)
    elif isinstance(value, list):
        for item in value:
            assert_public_snapshot(item)
    elif isinstance(value, str):
        _require(not value.startswith(("/", "~", "file://")) and not PureWindowsPath(value).is_absolute()
                 and not any(part in value for part in ("/Users/", "/home/", "C:\\Users\\")),
                 "Publication contains a local absolute path")


def _input_hashes(items: list[dict], experiment: str) -> list[dict]:
    required = {"features", "labels", "ppi_edges"}
    if experiment == "original":
        required |= {"comparison_genes", "gene_annotations", "neighbors"}
    _require({item["role"] for item in items} == required and len(items) == len(required),
             "Input provenance is incomplete")
    for item in items:
        _require(type(item["bytes"]) is int and item["bytes"] > 0
                 and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is not None, "Invalid input checksum metadata")
    return [{key: item[key] for key in ("role", "bytes", "sha256")} for item in items]


def build_snapshot(experiment: str, report: dict, raw_sha256: str) -> dict:
    """Validate a full raw report and return a compact version-1 publication."""
    _require(experiment in REPORTS, "Unknown experiment")
    _finite_tree(report)
    _require(report.get("status") == "complete", "Only complete experiments may be published")
    _require(re.fullmatch(r"[0-9a-f]{64}", raw_sha256) is not None, "Invalid raw-report SHA256")
    folds = report["folds"]
    _require(len(folds) == 3 and {fold["fold"] for fold in folds} == {1, 2, 3}, "Expected three complete unique folds")
    folds = sorted(folds, key=lambda fold: fold["fold"])
    tuning = experiment == "d"
    provenance = report["provenance"] if tuning else report
    config = provenance["base_config"] if tuning else report["config"]
    _check_config(config, experiment)
    ddof = config["evaluation"]["std_ddof"]
    total = 15028 if experiment == "original" else 16786
    counts = {"driver": 42, "non_driver": total - 42}
    use_validation = experiment in ("b", "c", "d")
    all_tests, sum_positive = [], 0
    public_folds = []
    for fold in folds:
        split = fold["splits"] if tuning else fold
        _check_splits(split, use_validation, total)
        all_tests.extend(split["test_gene_ids"])
        sum_positive += split["test_counts"]["driver"]
        selected = {"fold": fold["fold"], "class_weights": deepcopy(fold["class_weights"])}
        selected.update({key: deepcopy(value) for key, value in split.items() if key.endswith("_counts")})
        for role in ("driver", "non_driver"):
            fitting_count = split["train_counts"][role]
            _equal_number(fold["class_weights"][role], sum(split["train_counts"].values()) / (2 * fitting_count),
                          "class weight")
        if not tuning:
            _check_history(fold, use_validation, experiment == "c")
            _metrics(fold["metrics"])
            _check_predictions(fold["predictions"], split["test_gene_ids"], split["test_counts"])
            selected.update({"selected_epoch": fold["selected_epoch"], "test_metrics": deepcopy(fold["metrics"])})
            if use_validation:
                _check_predictions(fold["validation_predictions"], split["validation_gene_ids"], split["validation_counts"])
                selected["selected_validation_metrics"] = deepcopy(fold["selected_validation_metrics"])
        else:
            _require(fold["status"] == "complete", "Partial tuning folds cannot be published")
            _require({key: value for key, value in fold["provenance"].items() if key != "fold"} == provenance,
                     "Tuning fold provenance disagrees")
            grid = provenance["search_config"]
            _require(grid == {"learning_rates": [0.001, 0.0003, 0.0001],
                              "dropout_pairs": [[0.7, 0.6], [0.5, 0.3]], "selection_metric": "auc"},
                     "Publication expects the prespecified six-candidate search")
            trials = fold["candidate_results"]
            expected = [(lr, pair) for lr in grid["learning_rates"] for pair in grid["dropout_pairs"]]
            _require(len(trials) == 6, "All six candidates must be complete")
            for index, (trial, (lr, pair)) in enumerate(zip(trials, expected), 1):
                _require(trial["candidate_id"] == f"candidate_{index:02d}" and trial["learning_rate"] == lr
                         and trial["dropout_pair"] == pair, "Invalid candidate grid/order")
                _check_history(trial, True, True)
                _check_predictions(trial["validation_predictions"], split["validation_gene_ids"], split["validation_counts"])
            winner = max(trials, key=lambda trial: trial["selected_validation_metrics"]["auc"])
            _require(fold["selected_candidate_id"] == winner["candidate_id"], "Wrong validation-selected candidate")
            selected["seeds"] = deepcopy(fold["seeds"])
            selected["validation_candidates"] = [{key: deepcopy(trial[key]) for key in (
                "candidate_id", "learning_rate", "dropout_pair", "selected_epoch", "selected_validation_metrics",
                "initial_state_sha256",
            )} for trial in trials]
            for key, trial in (("tuned_test", winner), ("fixed_c_test", trials[0])):
                result = fold[key]
                _require(all(result[name] == trial[name] for name in (
                    "candidate_id", "learning_rate", "dropout_pair", "selected_epoch")), "Final candidate mismatch")
                _metrics(result["metrics"])
                _check_predictions(result["predictions"], split["test_gene_ids"], split["test_counts"])
                selected[key] = {name: deepcopy(value) for name, value in result.items() if name != "predictions"}
                selected[key]["selected_validation_metrics"] = deepcopy(trial["selected_validation_metrics"])
            selected["test_evaluation_count"] = fold["test_evaluation_count"]
        public_folds.append(selected)
    _require(len(all_tests) == len(set(all_tests)) == total and sum_positive == 42,
             "Test folds must cover the full dataset once, including all 42 positive labels")
    if not tuning:
        _require(report["num_targets"] == total and report["dataset_counts"] == counts, "Unexpected dataset counts")
        _check_summary(report["summary"], [fold["metrics"] for fold in folds], ddof, "summary")
    else:
        _require(report["std_ddof"] == ddof, "Inconsistent ddof")
        _check_summary(report["tuned_summary"], [fold["tuned_test"]["metrics"] for fold in folds], ddof, "tuned_summary")
        _check_summary(report["fixed_c_summary"], [fold["fixed_c_test"]["metrics"] for fold in folds], ddof, "fixed_c_summary")
        deltas = [{metric: fold["tuned_test"]["metrics"][metric] - fold["fixed_c_test"]["metrics"][metric]
                   for metric in METRICS} for fold in folds]
        _require(len(report["paired_differences"]) == 3, "Missing paired differences")
        for index, (recorded, expected_delta) in enumerate(zip(report["paired_differences"], deltas), 1):
            _require(recorded["fold"] == index, "Invalid paired-difference fold")
            for metric in METRICS:
                _equal_number(recorded["metrics"][metric], expected_delta[metric], "paired difference")
        _check_summary(report["paired_difference_summary"], deltas, ddof, "paired_difference_summary")
    result = {
        "schema_version": 1, "artifact_type": "compact_experiment_result", "status": "complete",
        "experiment": experiment.upper() if tuning or experiment != "original" else "original",
        "description": NAMES[experiment], "cancer": "BRCA", "dataset_counts": counts, "num_targets": total,
        "std_ddof": ddof,
        "export": {
            "raw_report": f"results/{REPORTS[experiment]}", "raw_report_sha256": raw_sha256,
            "omitted": ["per-gene predictions", "gene split membership", "epoch histories", "checkpoints",
                        "source input files", "logs", "machine-specific absolute paths"],
            "validation_scope": "Report structure, full-run coverage, selection consistency and aggregate arithmetic checked; no independent training replay or prediction-metric recomputation by this exporter.",
            "path_normalization": "Configured paths are project-relative where possible, otherwise input/output role labels; input checksums retain file identity.",
            "replay_limit": "A compact summary is not a complete replay artifact; obtain the documented inputs and rerun the experiment to produce detailed reports.",
        },
        "provenance": {"effective_config": _public_config(config), "seed": config["seed"],
                       "runtime": deepcopy(provenance["runtime"]), "device": provenance["device"],
                       "input_files": _input_hashes(provenance["input_files"], experiment)},
        "protocol": deepcopy(folds[0]["protocol"] if tuning else report["protocol"]), "folds": public_folds,
    }
    if tuning:
        result["provenance"].update({key: deepcopy(provenance[key]) for key in ("search_config", "dataset_sha256", "source_sha256")})
        result.update({key: deepcopy(report[key]) for key in (
            "tuned_summary", "fixed_c_summary", "paired_differences", "paired_difference_summary", "interpretation")})
        result["run_times"] = [{key: fold[key] for key in ("fold", "started_at_utc", "finished_at_utc", "elapsed_seconds")}
                               for fold in folds]
    else:
        result["summary"] = deepcopy(report["summary"])
        result["run_times"] = {key: report[key] for key in ("started_at_utc", "finished_at_utc", "elapsed_seconds")}
    assert_public_snapshot(result)
    return result


def export_all(input_dir: Path, output_dir: Path) -> list[Path]:
    """Validate all five reports before publishing any of their compact snapshots."""
    snapshots = []
    for experiment, relative in REPORTS.items():
        raw = (input_dir / relative).read_bytes()
        try:
            report = json.loads(raw)
            snapshot = build_snapshot(experiment, report, hashlib.sha256(raw).hexdigest())
        except (KeyError, TypeError, IndexError, ValueError) as error:
            raise ValueError(f"Cannot publish {relative}: {error}") from error
        snapshots.append((output_dir / f"{experiment}.json", snapshot))
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, snapshot in snapshots:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    return [path for path, _ in snapshots]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/published")
    args = parser.parse_args()
    for path in export_all(args.input_dir, args.output_dir):
        print(f"Exported {path.name} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
