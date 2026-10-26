"""Read recorded CV results without importing PyTorch, DGL, or running a model.

The compact, committed summary has aggregate results only. Full JSON reports may
be loaded locally to inspect previously recorded held-out gene scores. Neither
format is a model checkpoint, and loading a report does not make new predictions.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import re
import statistics
from typing import Any

METRIC_NAMES = ("auc", "accuracy", "precision", "recall", "f1", "balanced_accuracy")
EXPERIMENT_ORDER = ("original", "A", "B", "C", "D")
DEFAULT_REPORT_FILENAMES = {
    "original": "brca_metrics.json",
    "A": "brca_updated_data_metrics.json",
    "B": "brca_validation_metrics.json",
    "C": "brca_validation_train_mode_metrics.json",
    "D": "brca_tuning/summary.json",
}
MAX_REPORT_BYTES = 32 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: Any, name: str, low: float = 0, high: float = 1) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(value) and low <= value <= high,
             f"{name} must be a finite number in [{low}, {high}]")
    return float(value)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
             f"{name} must be an integer >= {minimum}")
    return value


def _read_json(source: Path | str | bytes) -> tuple[dict[str, Any], bytes]:
    if isinstance(source, bytes):
        raw = source
    else:
        path = Path(source)
        _require(path.stat().st_size <= MAX_REPORT_BYTES, "Report exceeds 32 MiB limit")
        raw = path.read_bytes()
    _require(len(raw) <= MAX_REPORT_BYTES, "Report exceeds 32 MiB limit")
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("Not a valid UTF-8 JSON result file") from error
    _require(isinstance(data, dict), "Result file must contain a JSON object")
    return data, raw


def _counts(value: dict[str, Any], name: str) -> int:
    return sum(_integer(value[key], f"{name}.{key}") for key in ("driver", "non_driver"))


def _metrics(metrics: dict[str, Any], name: str) -> None:
    for metric in METRIC_NAMES:
        _number(metrics[metric], f"{name}.{metric}")


def _close(left: float, right: float, name: str) -> None:
    _require(math.isclose(left, right, rel_tol=1e-8, abs_tol=1e-10),
             f"{name} does not match recorded predictions/folds")


def _prediction_metrics(records: list[dict[str, Any]], threshold: float,
                        zero_division: int) -> dict[str, float]:
    """Compute binary metrics independently; AUC uses average ranks for ties."""
    positives = sum(row["label"] for row in records)
    negatives = len(records) - positives
    _require(positives > 0 and negatives > 0, "Each evaluated split requires both classes")
    ordered = sorted(records, key=lambda row: row["driver_probability"])
    rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end]["driver_probability"] == ordered[start]["driver_probability"]:
            end += 1
        rank_sum += sum(row["label"] for row in ordered[start:end]) * (start + 1 + end) / 2
        start = end
    true_positive = sum(row["label"] == 1 and row["driver_probability"] >= threshold for row in records)
    false_positive = sum(row["label"] == 0 and row["driver_probability"] >= threshold for row in records)
    true_negative = negatives - false_positive
    false_negative = positives - true_positive
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else zero_division
    recall = true_positive / positives
    f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
    return {
        "auc": (rank_sum - positives * (positives + 1) / 2) / (positives * negatives),
        "accuracy": (true_positive + true_negative) / len(records),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": (recall + true_negative / negatives) / 2,
    }


def _ids(value: Any, name: str) -> set[str]:
    _require(isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value),
             f"{name} must be a list of nonempty gene IDs")
    _require(len(value) == len(set(value)), f"{name} contains duplicate gene IDs")
    return set(value)


def _prediction_split(fold: dict[str, Any], name: str, threshold: float,
                      zero_division: int) -> set[str]:
    predictions_key = "predictions" if name == "test" else "validation_predictions"
    metrics_key = "metrics" if name == "test" else "selected_validation_metrics"
    records = fold[predictions_key]
    _require(isinstance(records, list) and bool(records), f"Missing {name} predictions")
    for row in records:
        _require(isinstance(row["gene_id"], str) and bool(row["gene_id"].strip()), "Invalid gene ID")
        _require(type(row["label"]) is int and row["label"] in (0, 1), "Prediction labels must be 0 or 1")
        _number(row["driver_probability"], "driver_probability")
    identifiers = _ids([row["gene_id"] for row in records], f"{name} predictions")
    _require(identifiers == _ids(fold[f"{name}_gene_ids"], f"{name}_gene_ids"),
             f"{name} predictions do not match split gene IDs")
    _require(len(records) == _counts(fold[f"{name}_counts"], f"{name}_counts"), f"{name} count mismatch")
    _require(sum(row["label"] for row in records) == fold[f"{name}_counts"]["driver"],
             f"{name} driver count mismatch")
    _metrics(fold[metrics_key], metrics_key)
    recalculated = _prediction_metrics(records, threshold, zero_division)
    for metric in METRIC_NAMES:
        _close(fold[metrics_key][metric], recalculated[metric], f"{name} {metric}")
    return identifiers


def _feature_hash(report: dict[str, Any]) -> str:
    features = [entry for entry in report["input_files"] if entry["role"] == "features"]
    _require(len(features) == 1, "Report must identify exactly one feature input")
    result = features[0]["sha256"]
    _require(isinstance(result, str) and bool(_SHA256.fullmatch(result)), "Invalid feature SHA-256")
    return result


def _validate_aggregates(report: dict[str, Any]) -> None:
    _require(report["status"] == "complete" and report["cancer"] == "BRCA",
             "Only complete BRCA experiment reports are supported")
    _require(not report.get("smoke_test"), "Smoke-test results are not complete BRCA experiments")
    _require(len(report["folds"]) == 3 and report["config"]["cross_validation"]["n_splits"] == 3,
             "A complete three-fold experiment is required")
    _require(sorted(fold["fold"] for fold in report["folds"]) == [1, 2, 3], "Invalid fold IDs")
    _require(_integer(report["num_targets"], "num_targets", 1) == _counts(report["dataset_counts"], "dataset_counts"),
             "Dataset count mismatch")
    _integer(report["config"]["seed"], "seed")
    _require(report["protocol"]["std_ddof"] == 0, "Only population standard deviations (ddof=0) are supported")
    _feature_hash(report)
    for fold in report["folds"]:
        _metrics(fold["metrics"], "fold metrics")
        _require(1 <= _integer(fold["selected_epoch"], "selected_epoch", 1) <= report["config"]["training"]["epochs"],
                 "Selected epoch exceeds configured epochs")
    for metric in METRIC_NAMES:
        values = [fold["metrics"][metric] for fold in report["folds"]]
        _close(_number(report["summary"][metric]["mean"], f"mean {metric}"), statistics.mean(values), f"mean {metric}")
        _close(_number(report["summary"][metric]["std"], f"std {metric}"), statistics.pstdev(values), f"std {metric}")


def load_report(source: Path | str | bytes, *, expected_feature_sha256: str | None = None) -> dict[str, Any]:
    """Validate a full report, including splits, scores and recomputed metrics.

    ``expected_feature_sha256`` prevents confusing original-data results with
    the teammate-data experiments. JSON only: this never loads pickle/checkpoints.
    Malformed or incomplete reports raise a readable ``ValueError``.
    """
    report, _ = _read_json(source)
    try:
        if report.get("experiment") == "D" and "provenance" in report:
            report = _adapt_tuning_report(report)
        _validate_aggregates(report)
        if expected_feature_sha256 is not None:
            _require(_feature_hash(report) == expected_feature_sha256, "Feature dataset SHA-256 does not match the selected experiment")
        config = report["config"]
        threshold = _number(config["evaluation"]["probability_threshold"], "probability_threshold")
        _require(config["evaluation"]["positive_class"] == 1, "Expected positive driver class 1")
        zero_division = config["evaluation"]["zero_division"]
        _require(zero_division in (0, 1), "Unsupported zero_division setting")
        uses_validation = config["cross_validation"]["epoch_selection"] == "best_validation_roc_auc"
        _require(config["cross_validation"]["epoch_selection"] in ("best_validation_roc_auc", "best_test_roc_auc"),
                 "Unknown epoch selection procedure")
        _require(report["protocol"]["test_fold_used_for_epoch_selection"] is (not uses_validation),
                 "Epoch-selection protocol and configuration disagree")
        seen: set[str] = set()
        universes: list[set[str]] = []
        for fold in report["folds"]:
            test_ids = _prediction_split(fold, "test", threshold, zero_division)
            train_ids = _ids(fold["train_gene_ids"], "train_gene_ids")
            _require(len(train_ids) == _counts(fold["train_counts"], "train_counts"), "Training count mismatch")
            _require(not test_ids & seen, "Gene appears in more than one outer test fold")
            _require(not train_ids & test_ids, "Training/test gene overlap")
            seen.update(test_ids)
            validation_ids: set[str] = set()
            if uses_validation:
                validation_ids = _prediction_split(fold, "validation", threshold, zero_division)
                _require(not validation_ids & (train_ids | test_ids), "Validation split overlaps training/test genes")
            universes.append(train_ids | validation_ids | test_ids)
            history = fold["epoch_history"]
            _require(len(history) == config["training"]["epochs"], "Report has an incomplete epoch history")
            _require([row["epoch"] for row in history] == list(range(1, len(history) + 1)), "Invalid epoch sequence")
            metric_key = "validation_metrics" if uses_validation else "metrics"
            for row in history:
                _metrics(row[metric_key], "epoch metrics")
            selected = max(history, key=lambda row: row[metric_key]["auc"])
            _require(selected["epoch"] == fold["selected_epoch"], "Selected epoch is not the earliest maximum AUC")
            selected_metrics = fold["selected_validation_metrics"] if uses_validation else fold["metrics"]
            for metric in METRIC_NAMES:
                _close(selected_metrics[metric], selected[metric_key][metric], "Selected epoch metrics")
        _require(len(seen) == report["num_targets"] and all(universe == seen for universe in universes),
                 "Cross-validation does not cover the same complete target population")
        _require(sum(fold["test_counts"]["driver"] for fold in report["folds"]) == report["dataset_counts"]["driver"],
                 "Outer test labels disagree with dataset counts")
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError(f"Malformed DriverMONI report: missing or invalid field {error}") from error
    return report


def validate_experiment(report: dict[str, Any], experiment: str) -> None:
    """Reject a report whose dataset/training/selection procedure is mislabeled.

    Run after ``load_report``; dataset identity is checked there using the
    expected feature hash. This checks experiment-defining settings, not whether
    a rerun is byte-for-byte identical to the committed recorded result.
    """
    _require(experiment in DEFAULT_REPORT_FILENAMES, "Unknown experiment identifier")
    try:
        config = report["config"]
        cross_validation = config["cross_validation"]
        protocol = report["protocol"]
        uses_validation = experiment not in ("original", "A")
        expected_selection = "best_validation_roc_auc" if uses_validation else "best_test_roc_auc"
        _require(cross_validation["epoch_selection"] == expected_selection,
                 f"Report does not implement experiment {experiment}'s epoch selection")
        preserve_mode = experiment not in ("C", "D")
        _require(config["training"]["preserve_eval_mode_after_first_evaluation"] is preserve_mode
                 and protocol["source_preserve_eval_mode"] is preserve_mode,
                 f"Report does not implement experiment {experiment}'s training mode")
        _require(protocol["test_fold_used_for_epoch_selection"] is (not uses_validation),
                 f"Report epoch-selection protocol disagrees with experiment {experiment}")
        _require(cross_validation["n_splits"] == 3
                 and cross_validation["stratified"] is True
                 and cross_validation["shuffle"] is True,
                 "Experiments use three shuffled stratified outer folds")
        original_data = experiment == "original"
        _require(config["data"]["use_comparison_filter"] is original_data
                 and config["data"]["use_supplied_neighbors"] is original_data,
                 "Original uses authors' filtered data and supplied neighborhoods; A–D use updated data and rebuilt neighborhoods")
        if experiment == "D":
            _require(report.get("experiment") == "D" and protocol.get("hyperparameter_search") is True,
                     "Experiment D requires the recorded validation hyperparameter search")
            _check_search_grid(report["search_config"])
        else:
            _require(not protocol.get("hyperparameter_search"), "A tuned experiment D report cannot be labeled as another experiment")
        for fold in report["folds"]:
            # Full reports preserve actual update modes; compact snapshots do not
            # retain epoch history and have already been checked during export.
            for row in fold.get("epoch_history", []):
                expected_training = not preserve_mode or row["epoch"] == 1
                _require(row["training_mode_during_updates"] is expected_training,
                         f"Epoch update modes do not implement experiment {experiment}")
        if uses_validation:
            _require(cross_validation["validation_fraction"] == .25
                     and protocol["validation_fraction"] == .25,
                     "B/C/D use a 25% inner-validation fraction")
            _require(protocol["test_evaluations_per_fold"] == 1
                     and protocol["refit_on_outer_training"] is False
                     and protocol["class_weight_population"] == "inner training targets only",
                     "B/C/D require one final test per selected model, no outer-fold refit, and inner-fit-only class weights")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"Missing experiment-defining setting for {experiment}: {error}") from error


def gene_prediction(report: dict[str, Any], gene_id: str) -> dict[str, Any] | None:
    """Return one recorded outer-test score; call with a validated ``load_report`` result.

    Score is a model softmax output, not a calibrated cancer risk. Label 0 means
    absent from the positive reference list, not a proven biological non-driver.
    A's outer test score is test-selected and must not be called independent.
    """
    threshold = report["config"]["evaluation"]["probability_threshold"]
    for fold in report["folds"]:
        for row in fold["predictions"]:
            if row["gene_id"] == gene_id.strip():
                prediction = {
                    **row, "fold": fold["fold"], "selected_epoch": fold["selected_epoch"],
                    "threshold": threshold, "predicted_class": int(row["driver_probability"] >= threshold),
                    "test_fold_used_for_epoch_selection": report["protocol"]["test_fold_used_for_epoch_selection"],
                    "source": "recorded_outer_test_prediction",
                }
                if report.get("experiment") == "D":
                    prediction.update({key: deepcopy(fold[key]) for key in ("candidate_id", "learning_rate", "dropout_pair")})
                return prediction
    return None


def _safe_config(config: dict[str, Any]) -> dict[str, Any]:
    """Keep scientific settings; exclude all filesystem path settings."""
    allowed = ("cancer", "upstream", "seed", "device", "reproducibility", "model", "graph", "training", "cross_validation", "evaluation")
    result = {key: deepcopy(config[key]) for key in allowed if key in config}
    result["data"] = {key: config["data"][key] for key in ("use_comparison_filter", "use_supplied_neighbors")}
    return result


def _no_private_data(value: Any) -> None:
    if isinstance(value, dict):
        _require(not set(value) & {"path", "predictions", "validation_predictions", "train_gene_ids", "test_gene_ids", "validation_gene_ids", "outer_train_gene_ids"},
                 "Compact summary must not include paths or per-gene records")
        for child in value.values():
            _no_private_data(child)
    elif isinstance(value, list):
        for child in value:
            _no_private_data(child)
    elif isinstance(value, str):
        _require(not Path(value).is_absolute() and not PureWindowsPath(value).is_absolute(),
                 "Compact summary must not contain local absolute paths")


def export_summary(report_paths: Mapping[str, Path | str], output_path: Path | str,
                   *, expected_feature_sha256: str | None = None) -> dict[str, Any]:
    """Export path-free, prediction-free summaries of verified experiment reports.

    A–D must share feature, label, and PPI inputs. The original reproduction uses
    a separate dataset. Input and source-report hashes preserve provenance.
    """
    experiments = []
    _require(set(report_paths) <= set(EXPERIMENT_ORDER), "Unknown experiment identifier")
    for name in EXPERIMENT_ORDER:
        if name not in report_paths:
            continue
        source = report_paths[name]
        raw = Path(source).read_bytes()
        report = load_report(raw, expected_feature_sha256=expected_feature_sha256 if name != "original" else None)
        validate_experiment(report, name)
        inputs = [{key: entry[key] for key in ("role", "sha256", "bytes")} for entry in report["input_files"]]
        keep = ("status", "cancer", "device", "runtime", "dataset_counts", "num_targets", "protocol", "summary", "started_at_utc", "finished_at_utc", "elapsed_seconds")
        entry = {key: deepcopy(report[key]) for key in keep}
        entry.update({
            "experiment": name, "source_filename": DEFAULT_REPORT_FILENAMES[name],
            "report_sha256": hashlib.sha256(raw).hexdigest(),
            "seed": report["config"]["seed"], "config": _safe_config(report["config"]),
            "input_files": inputs, "feature_sha256": _feature_hash(report),
            "folds": [{key: deepcopy(fold[key]) for key in ("fold", "selected_epoch", "metrics", "train_counts", "test_counts", "class_weights", "validation_counts", "selected_validation_metrics") if key in fold} for fold in report["folds"]],
        })
        if name == "D":
            keep_fold = ("fold", "selected_epoch", "metrics", "train_counts", "test_counts", "class_weights",
                         "validation_counts", "selected_validation_metrics", "candidate_id", "learning_rate", "dropout_pair")
            entry["folds"] = [{key: deepcopy(fold[key]) for key in keep_fold} for fold in report["folds"]]
            entry["matched_control"] = {
                "summary": deepcopy(report["matched_control"]["summary"]),
                "folds": [{key: deepcopy(fold[key]) for key in keep_fold}
                          for fold in report["matched_control"]["folds"]],
            }
            for key in ("search_config", "paired_differences", "paired_difference_summary"):
                entry[key] = deepcopy(report[key])
        experiments.append(entry)
    _require(bool(experiments), "At least one complete report is required")
    _same_updated_inputs(experiments)
    snapshot = {
        "schema_version": 1, "scope": "recorded_cross_validation_results", "experiments": experiments,
        "notes": [
            "Recorded results, not fresh predictions; no trained model is included in this summary.",
            "Original and A use test-fold AUC for epoch selection; their scores are not independent estimates comparable to B/C/D.",
            "B and C use inner-validation selection. C restores training mode every epoch; B preserves the authors' mode behavior.",
            "D chooses learning rate and dropout on inner validation data; its matched fixed-C control uses the same random initialization and batch order.",
            "One seed and three folds do not establish a consistent performance difference. Shared precomputed graph/features remain.",
            "Class 0 means absent from the positive reference list. Scores are not calibrated cancer risks.",
        ],
    }
    _no_private_data(snapshot)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return snapshot


def load_experiment_summary(source: Path | str | bytes) -> dict[str, Any]:
    """Load the small aggregate artifact and verify its folds/summary consistency."""
    snapshot, _ = _read_json(source)
    try:
        _require(snapshot["schema_version"] == 1 and snapshot["scope"] == "recorded_cross_validation_results", "Unsupported experiment summary schema")
        _require(bool(snapshot["experiments"]), "Empty experiment summary")
        identifiers = [entry["experiment"] for entry in snapshot["experiments"]]
        _require(len(set(identifiers)) == len(identifiers) and set(identifiers) <= DEFAULT_REPORT_FILENAMES.keys(), "Invalid or duplicate experiment IDs")
        for entry in snapshot["experiments"]:
            _validate_aggregates(entry)
            validate_experiment(entry, entry["experiment"])
            _require(bool(_SHA256.fullmatch(entry["report_sha256"])), "Invalid source report SHA-256")
            _require(entry["feature_sha256"] == _feature_hash(entry), "Feature hash inconsistency")
            if entry["experiment"] == "D":
                control = entry["matched_control"]
                _validate_aggregates(dict(entry, summary=control["summary"], folds=control["folds"]))
                _verify_paired_differences(entry, entry["folds"], control["folds"])
        _same_updated_inputs(snapshot["experiments"])
        snapshot["experiments"].sort(key=lambda entry: EXPERIMENT_ORDER.index(entry["experiment"]))
        _no_private_data(snapshot)
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError(f"Malformed experiment summary: {error}") from error
    return snapshot


def _check_search_grid(grid: dict[str, Any]) -> list[tuple[float, list[float]]]:
    expected = {"learning_rates": [.001, .0003, .0001],
                "dropout_pairs": [[.7, .6], [.5, .3]], "selection_metric": "auc"}
    _require(grid == expected, "D must use the recorded six-candidate learning-rate/dropout grid")
    return [(rate, pair) for rate in grid["learning_rates"] for pair in grid["dropout_pairs"]]


def _tuning_protocol(protocol: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    _require(protocol["test_fold_used_for_selection"] is False
             and protocol["refit_on_outer_training"] is False
             and protocol["training_mode_every_epoch"] is True
             and protocol["class_weight_population"] == "inner fitting targets only"
             and protocol["fixed_c_control"] == {"learning_rate": .001, "dropout_pair": [.7, .6]},
             "D's recorded search/control protocol is inconsistent")
    result = deepcopy(protocol)
    result.update({"std_ddof": config["evaluation"]["std_ddof"], "hyperparameter_search": True,
                   "test_fold_used_for_epoch_selection": False, "source_preserve_eval_mode": False,
                   "validation_fraction": config["cross_validation"]["validation_fraction"],
                   "test_evaluations_per_fold": 1, "class_weight_population": "inner training targets only"})
    return result


def _tuning_choices(trials: list[dict[str, Any]], grid: dict[str, Any], *, full: bool,
                    split: dict[str, Any], config: dict[str, Any]) -> tuple[dict, dict]:
    expected = _check_search_grid(grid)
    _require(len(trials) == len(expected), "D requires all six completed candidates")
    fingerprints = []
    for index, (trial, (rate, dropout)) in enumerate(zip(trials, expected), 1):
        _require(trial["candidate_id"] == f"candidate_{index:02d}"
                 and trial["learning_rate"] == rate and trial["dropout_pair"] == dropout,
                 "D candidate order/settings disagree with the search grid")
        _metrics(trial["selected_validation_metrics"], "candidate validation metrics")
        _require(1 <= _integer(trial["selected_epoch"], "selected epoch", 1) <= config["training"]["epochs"],
                 "Invalid D candidate epoch")
        fingerprint = trial["initial_state_sha256"]
        _require(isinstance(fingerprint, str) and bool(_SHA256.fullmatch(fingerprint)), "Invalid initial-state checksum")
        fingerprints.append(fingerprint)
        if full:
            _prediction_split(dict(split, selected_validation_metrics=trial["selected_validation_metrics"],
                                   validation_predictions=trial["validation_predictions"]),
                              "validation", config["evaluation"]["probability_threshold"],
                              config["evaluation"]["zero_division"])
            history = trial["epoch_history"]
            _require(len(history) == config["training"]["epochs"]
                     and [row["epoch"] for row in history] == list(range(1, len(history) + 1)),
                     "D candidate has incomplete epoch history")
            for row in history:
                _metrics(row["validation_metrics"], "candidate epoch metrics")
                _require(row["training_mode_during_updates"] is True and row["learning_rate"] == rate,
                         "D candidate history has inconsistent learning rate or training mode")
            best = max(history, key=lambda row: row["validation_metrics"]["auc"])
            _require(best["epoch"] == trial["selected_epoch"], "D candidate did not select earliest maximum validation AUC")
            for metric in METRIC_NAMES:
                _close(best["validation_metrics"][metric], trial["selected_validation_metrics"][metric],
                       "D selected validation metrics")
    _require(len(set(fingerprints)) == 1, "D candidates do not share paired initial weights")
    return max(trials, key=lambda trial: trial["selected_validation_metrics"]["auc"]), trials[0]


def _selected_tuning_fold(fold: dict[str, Any], split: dict[str, Any], trial: dict[str, Any],
                         key: str, *, full: bool) -> dict[str, Any]:
    final = fold[key]
    for field in ("candidate_id", "learning_rate", "dropout_pair", "selected_epoch"):
        _require(final[field] == trial[field], "D final model disagrees with validation-selected candidate/control")
    if "selected_validation_metrics" in final:
        _metrics(final["selected_validation_metrics"], "final selected validation metrics")
        for metric in METRIC_NAMES:
            _close(final["selected_validation_metrics"][metric], trial["selected_validation_metrics"][metric],
                   "D selected validation metrics")
    counts = {name: deepcopy(value) for name, value in split.items() if name.endswith("_counts")}
    result = dict(counts, fold=fold["fold"], class_weights=deepcopy(fold["class_weights"]),
                  selected_validation_metrics=deepcopy(trial["selected_validation_metrics"]))
    result.update({field: deepcopy(final[field]) for field in (
        "candidate_id", "learning_rate", "dropout_pair", "selected_epoch", "metrics")})
    if full:
        result.update({name: deepcopy(value) for name, value in split.items() if name.endswith("_gene_ids")})
        result.update(predictions=deepcopy(final["predictions"]),
                      validation_predictions=deepcopy(trial["validation_predictions"]),
                      epoch_history=deepcopy(trial["epoch_history"]))
    return result


def _verify_paired_differences(report: dict[str, Any], tuned: list[dict], control: list[dict]) -> None:
    _require(len(report["paired_differences"]) == 3, "Missing D paired differences")
    values = []
    for left, right, recorded in zip(tuned, control, report["paired_differences"]):
        _require(left["fold"] == right["fold"] == recorded["fold"], "Invalid paired fold order")
        delta = {metric: left["metrics"][metric] - right["metrics"][metric] for metric in METRIC_NAMES}
        for metric, value in delta.items():
            _close(_number(recorded["metrics"][metric], "paired difference", -1, 1), value, "paired difference")
        values.append(delta)
    for metric in METRIC_NAMES:
        numbers = [value[metric] for value in values]
        _close(_number(report["paired_difference_summary"][metric]["mean"], "paired mean", -1, 1),
               statistics.mean(numbers), "paired mean")
        _close(_number(report["paired_difference_summary"][metric]["std"], "paired std", 0, 2),
               statistics.pstdev(numbers), "paired std")


def _adapt_tuning_report(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate all D trials and expose selected held-out scores in legacy shape."""
    _require(raw["status"] == "complete" and len(raw["folds"]) == 3, "Only complete three-fold D reports are supported")
    provenance = raw["provenance"]
    config = provenance["base_config"]
    grid = provenance["search_config"]
    _check_search_grid(grid)
    _require(raw["std_ddof"] == config["evaluation"]["std_ddof"] == 0, "D requires population standard deviation")
    _require(bool(_SHA256.fullmatch(provenance["dataset_sha256"])), "Invalid D dataset checksum")
    _require(bool(provenance["source_sha256"]) and all(bool(_SHA256.fullmatch(value))
             for value in provenance["source_sha256"].values()), "Invalid D source checksums")
    folds, control = [], []
    ordered = sorted(raw["folds"], key=lambda fold: fold["fold"])
    for fold in ordered:
        _require(fold["status"] == "complete", "Incomplete D fold")
        _require({key: value for key, value in fold["provenance"].items() if key != "fold"} == provenance,
                 "D fold provenance disagrees with aggregate report")
        _require(fold["provenance"]["fold"] == fold["fold"], "D fold provenance has wrong fold number")
        _tuning_protocol(fold["protocol"], config)
        winner, fixed = _tuning_choices(fold["candidate_results"], grid, full=True, split=fold["splits"], config=config)
        _require(fold["selected_candidate_id"] == winner["candidate_id"], "D winner is not selected by validation AUC")
        _require(fold["test_evaluation_count"] == (1 if winner is fixed else 2), "D test-evaluation count is inconsistent")
        folds.append(_selected_tuning_fold(fold, fold["splits"], winner, "tuned_test", full=True))
        control.append(_selected_tuning_fold(fold, fold["splits"], fixed, "fixed_c_test", full=True))
    counts = {role: sum(fold["test_counts"][role] for fold in folds) for role in ("driver", "non_driver")}
    report = {
        "status": "complete", "cancer": "BRCA", "experiment": "D", "config": deepcopy(config),
        "device": provenance["device"], "runtime": deepcopy(provenance["runtime"]),
        "input_files": deepcopy(provenance["input_files"]), "dataset_counts": counts,
        "num_targets": sum(counts.values()), "protocol": _tuning_protocol(ordered[0]["protocol"], config),
        "folds": folds, "summary": deepcopy(raw["tuned_summary"]), "search_config": deepcopy(grid),
        "matched_control": {"summary": deepcopy(raw["fixed_c_summary"]), "folds": control},
        "paired_difference_summary": deepcopy(raw["paired_difference_summary"]),
        "paired_differences": deepcopy(raw["paired_differences"]),
        "started_at_utc": min(fold["started_at_utc"] for fold in ordered),
        "finished_at_utc": max(fold["finished_at_utc"] for fold in ordered),
        "elapsed_seconds": sum(fold["elapsed_seconds"] for fold in ordered),
        "elapsed_seconds_scope": "sum of per-fold elapsed time", "source_sha256": deepcopy(provenance["source_sha256"]),
    }
    # The selected D model is checked by load_report's normal path. Reuse that
    # full validation for the prespecified control, including its predictions.
    control_report = dict(report, folds=control, summary=raw["fixed_c_summary"])
    load_report(json.dumps(control_report, allow_nan=False).encode())
    _verify_paired_differences(raw, folds, control)
    validate_experiment(report, "D")
    return report


def _same_updated_inputs(entries: list[dict]) -> None:
    signatures = [sorted((item["role"], item["sha256"]) for item in entry["input_files"])
                  for entry in entries if entry["experiment"] != "original"]
    _require(not signatures or all(signature == signatures[0] for signature in signatures),
             "Updated-data experiments use different feature/label/PPI inputs; do not combine them")


def load_published_experiments(directory: Path | str) -> dict[str, Any]:
    """Load the five committed compact results in Original/A/B/C/D order.

    Original uses different processed inputs and is explicitly kept separate.
    A–D must share input identities. Compact files verify aggregates/protocols;
    they omit per-gene scores, which require an optional full local report.
    """
    entries = []
    try:
        for name in EXPERIMENT_ORDER:
            published, _ = _read_json(Path(directory) / f"{name.lower()}.json")
            _require(published["schema_version"] == 1
                     and published["artifact_type"] == "compact_experiment_result"
                     and published["experiment"] == name and published["status"] == "complete",
                     f"Unsupported or mislabeled published experiment {name}")
            provenance = published["provenance"]
            config = provenance["effective_config"]
            _require(published["std_ddof"] == config["evaluation"]["std_ddof"] == 0, "Expected population standard deviation")
            expected_total = 15028 if name == "original" else 16786
            _require(published["num_targets"] == expected_total
                     and published["dataset_counts"] == {"driver": 42, "non_driver": expected_total - 42},
                     "Published report does not describe the full expected BRCA dataset")
            inputs = deepcopy(provenance["input_files"])
            required_roles = {"features", "labels", "ppi_edges"}
            if name == "original":
                required_roles |= {"comparison_genes", "gene_annotations", "neighbors"}
            _require(len(inputs) == len(required_roles) and {item["role"] for item in inputs} == required_roles,
                     "Incomplete published input provenance")
            for item in inputs:
                _require(bool(_SHA256.fullmatch(item["sha256"])), "Invalid published input checksum")
                _integer(item["bytes"], "input size", 1)
            protocol = deepcopy(published["protocol"])
            fold_entries = []
            control_entries = []
            for fold in sorted(published["folds"], key=lambda fold: fold["fold"]):
                if name == "D":
                    winner, fixed = _tuning_choices(fold["validation_candidates"], provenance["search_config"],
                                                    full=False, split=fold, config=config)
                    _require(fold["test_evaluation_count"] == (1 if winner is fixed else 2), "Invalid D test count")
                    fold_entries.append(_selected_tuning_fold(fold, fold, winner, "tuned_test", full=False))
                    control_entries.append(_selected_tuning_fold(fold, fold, fixed, "fixed_c_test", full=False))
                    fold_entries[-1]["seeds"] = deepcopy(fold["seeds"])
                else:
                    normalized = {key: deepcopy(value) for key, value in fold.items() if key != "test_metrics"}
                    normalized["metrics"] = deepcopy(fold["test_metrics"])
                    fold_entries.append(normalized)
            if name == "D":
                protocol = _tuning_protocol(protocol, config)
            entry = {
                "experiment": name, "status": "complete", "cancer": "BRCA", "device": provenance["device"],
                "runtime": deepcopy(provenance["runtime"]), "seed": provenance["seed"], "config": _safe_config(config),
                "protocol": protocol, "dataset_counts": deepcopy(published["dataset_counts"]),
                "num_targets": published["num_targets"], "input_files": inputs,
                "source_filename": DEFAULT_REPORT_FILENAMES[name], "report_sha256": published["export"]["raw_report_sha256"],
                "summary": deepcopy(published["tuned_summary"] if name == "D" else published["summary"]),
                "folds": fold_entries, "dataset_group": "authors_original" if name == "original" else "updated",
                "source": "published_compact_results",
            }
            if name == "D":
                entry.update(search_config=deepcopy(provenance["search_config"]),
                             matched_control={"summary": deepcopy(published["fixed_c_summary"]), "folds": control_entries},
                             paired_difference_summary=deepcopy(published["paired_difference_summary"]),
                             paired_differences=deepcopy(published["paired_differences"]))
                _require(bool(_SHA256.fullmatch(provenance["dataset_sha256"]))
                         and bool(provenance["source_sha256"])
                         and all(bool(_SHA256.fullmatch(value)) for value in provenance["source_sha256"].values()),
                         "Invalid D dataset/source checksums")
                entry["dataset_sha256"] = provenance["dataset_sha256"]
                entry["source_sha256"] = deepcopy(provenance["source_sha256"])
                _validate_aggregates(dict(entry, folds=control_entries, summary=published["fixed_c_summary"]))
                _verify_paired_differences(published, fold_entries, control_entries)
            entry["feature_sha256"] = _feature_hash(entry)
            _require(entry["seed"] == config["seed"], "Published seed disagrees with configuration")
            _require(bool(_SHA256.fullmatch(entry["report_sha256"])), "Invalid raw-report checksum")
            _validate_aggregates(entry)
            validate_experiment(entry, name)
            _validate_compact_counts(entry)
            entries.append(entry)
        _same_updated_inputs(entries)
        snapshot = {"schema_version": 1, "scope": "recorded_cross_validation_results", "experiments": entries,
                    "notes": ["Original uses the authors' processed dataset; A–D use the updated dataset.",
                              "Original and A use test scores to choose an epoch, which can make their results optimistic.",
                              "D's matched fixed-C control shares its random initialization and batch order; historical C does not.",
                              "These are recorded experimental results, not new predictions or proven biological diagnoses."]}
        _no_private_data(snapshot)
        return snapshot
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError(f"Malformed published result: {error}") from error


def _validate_compact_counts(entry: dict[str, Any]) -> None:
    total = entry["num_targets"]
    for fold in entry["folds"]:
        train = _counts(fold["train_counts"], "train counts")
        test = _counts(fold["test_counts"], "test counts")
        validation = _counts(fold["validation_counts"], "validation counts") if "validation_counts" in fold else 0
        _require(train + test + validation == total, "Published fold counts do not cover the dataset")
        for role in ("driver", "non_driver"):
            _require(sum(fold[key][role] for key in ("train_counts", "test_counts", "validation_counts") if key in fold)
                     == entry["dataset_counts"][role], "Published fold class counts disagree")
            _close(fold["class_weights"][role], train / (2 * fold["train_counts"][role]), "Published class weights")
        if "selected_validation_metrics" in fold:
            _metrics(fold["selected_validation_metrics"], "selected validation metrics")
    for role in ("driver", "non_driver"):
        _require(sum(fold["test_counts"][role] for fold in entry["folds"]) == entry["dataset_counts"][role],
                 "Published test folds do not cover the class population")
