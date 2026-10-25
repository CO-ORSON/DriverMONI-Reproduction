"""Experiment D: validation-only learning-rate/dropout search, separate from CV.

Each outer fold searches one fixed inner holdout, then evaluates only its winner
and the prespecified C control. The original model/trainer defaults are untouched.
"""

from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from numbers import Real
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .dataset import DriverMONIDataset, collate_gene_samples
from .evaluate import evaluate
from .models import DriverMONI
from .train import _make_splits, class_counts, class_weights, summarize_folds, train_one_epoch
from .utils import runtime_metadata, save_json, seed_everything, select_device, validate_config

_FORMAT_VERSION = 1
_MODULUS = 2**31


def tuning_candidates(search_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the grid; return learning-rate-major, dropout-minor candidates."""
    if not isinstance(search_config, dict) or set(search_config) != {
        "learning_rates", "dropout_pairs", "selection_metric",
    }:
        raise ValueError("Search requires learning_rates, dropout_pairs and selection_metric")
    if search_config["selection_metric"] != "auc":
        raise ValueError("Experiment D selects by validation AUC only")
    rates, pairs = search_config["learning_rates"], search_config["dropout_pairs"]
    if not isinstance(rates, list) or not rates or not isinstance(pairs, list) or not pairs:
        raise ValueError("Learning rates and dropout pairs must be nonempty lists")

    def finite(value: Any) -> bool:
        return isinstance(value, Real) and not isinstance(value, bool) and bool(np.isfinite(value))

    if any(not finite(rate) or rate <= 0 for rate in rates):
        raise ValueError("Learning rates must be finite and positive")
    if any(not isinstance(pair, list) or len(pair) != 2 or
           any(not finite(p) or not 0 <= p < 1 for p in pair) for pair in pairs):
        raise ValueError("Each dropout pair must contain two finite values in [0, 1)")
    if len(set(rates)) != len(rates) or len({tuple(pair) for pair in pairs}) != len(pairs):
        raise ValueError("Duplicate grid candidates are not allowed")
    candidates = [
        {"candidate_id": f"candidate_{index:02d}", "learning_rate": float(rate),
         "dropout_pair": [float(p) for p in pair]}
        for index, (rate, pair) in enumerate(((r, p) for r in rates for p in pairs), 1)
    ]
    if not any(candidate["learning_rate"] == 0.001 and candidate["dropout_pair"] == [0.7, 0.6]
               for candidate in candidates):
        raise ValueError("The grid must include the prespecified C control: 0.001, [0.7, 0.6]")
    return candidates


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    paths = [root / name for name in (
        "tuning.py", "train.py", "evaluate.py", "dataset.py", "graph_utils.py", "utils.py",
        "feature_schema.py", "models/__init__.py", "models/dnn.py", "models/gat.py", "models/drivermoni.py",
    )]
    return {str(path.relative_to(root.parent)): _sha256(path) for path in paths}


def _dataset_fingerprint(dataset: DriverMONIDataset) -> str:
    """Hash consumed features/labels, target order, global graph and neighborhoods."""
    digest = hashlib.sha256()

    def add(value: Any) -> None:
        encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)

    add(list(dataset.target_gene_ids))
    add(list(dataset.context.index))
    add(dataset.context["label"].astype(int).tolist())
    values = dataset.node_features.detach().cpu().contiguous().numpy()
    add({"shape": list(values.shape), "dtype": str(values.dtype)})
    digest.update(values.tobytes())
    add(list(dataset.ppi.gene_ids))
    for edges in dataset.ppi.graph.edges(order="eid"):
        values = edges.detach().cpu().contiguous().numpy()
        add({"shape": list(values.shape), "dtype": str(values.dtype)})
        digest.update(values.tobytes())
    # Only target neighborhoods are consumed by dataset.__getitem__.
    for gene in dataset.target_gene_ids:
        add([gene, list(dataset.neighbors[gene])])
    return digest.hexdigest()


def _new_model(candidate: dict[str, Any], device: torch.device) -> DriverMONI:
    """Create a fresh unchanged-width model; override only its two dropout rates."""
    model = DriverMONI()
    model.dnn.dropout1.p, model.dnn.dropout2.p = candidate["dropout_pair"]
    return model.to(device)


def _save_state(path: Path, state: dict[str, torch.Tensor]) -> str:
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    return _sha256(path)


def _state_fingerprint(model: DriverMONI) -> str:
    """Hash initial parameters and buffers independently of checkpoint containers."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        values = tensor.detach().cpu().contiguous().numpy()
        digest.update(json.dumps([name, str(values.dtype), list(values.shape)]).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _verify_saved_candidates(report: dict[str, Any], candidates: list[dict[str, Any]], directory: Path) -> None:
    results = report["candidate_results"]
    if len(results) > len(candidates):
        raise ValueError("Resume report has too many candidates")
    if report.get("status") not in {"running", "failed", "interrupted", "complete"}:
        raise ValueError("Resume report has an invalid status")
    epochs = report["provenance"]["base_config"]["training"]["epochs"]
    for expected, result in zip(candidates, results, strict=False):
        if any(result.get(key) != value for key, value in expected.items()):
            raise ValueError("Resume candidates do not match the configured ordered grid")
        path = directory / f"{expected['candidate_id']}.pt"
        if not path.is_file() or _sha256(path) != result["checkpoint_sha256"]:
            raise ValueError(f"Resume checkpoint is missing or changed: {path.name}")
        history = result.get("epoch_history", [])
        if len(history) != epochs or any(
            row.get("epoch") != index or row.get("training_mode_during_updates") is not True
            or row.get("learning_rate") != expected["learning_rate"]
            or row.get("evaluation_split") != "validation"
            for index, row in enumerate(history, 1)
        ):
            raise ValueError("Resume candidate history is incomplete or inconsistent")
        best = max(history, key=lambda row: row["validation_metrics"]["auc"])
        if (result.get("selected_epoch") != best["epoch"]
                or result.get("selected_validation_metrics") != best["validation_metrics"]):
            raise ValueError("Resume candidate does not use the earliest maximum validation epoch")
        predicted_ids = [row["gene_id"] for row in result.get("validation_predictions", [])]
        if (len(predicted_ids) != len(set(predicted_ids))
                or set(predicted_ids) != set(report["splits"]["validation_gene_ids"])):
            raise ValueError("Resume candidate validation predictions do not match the split")
        if not isinstance(result.get("initial_state_sha256"), str):
            raise ValueError("Resume candidate lacks its initialization fingerprint")
    if len({result["initial_state_sha256"] for result in results}) > 1:
        raise ValueError("Resume candidates do not share paired initial parameters and buffers")
    if report["status"] == "complete" or "tuned_test" in report or "fixed_c_test" in report:
        if len(results) != len(candidates):
            raise ValueError("Resume final evaluation requires every completed candidate")
        winner = max(results, key=lambda result: result["selected_validation_metrics"]["auc"])
        control = next(result for result in results
                       if result["learning_rate"] == 0.001 and result["dropout_pair"] == [0.7, 0.6])
        if report.get("selected_candidate_id") != winner["candidate_id"]:
            raise ValueError("Resume winner is inconsistent with validation-only selection")
        for name, selected in (("tuned_test", winner), ("fixed_c_test", control)):
            if name not in report and report["status"] != "complete":
                continue
            final = report.get(name, {})
            if any(final.get(key) != selected[key] for key in
                   ("candidate_id", "learning_rate", "dropout_pair", "selected_epoch")):
                raise ValueError("Resume final result disagrees with its selected candidate")
            predicted_ids = [row["gene_id"] for row in final.get("predictions", [])]
            if (len(predicted_ids) != len(set(predicted_ids))
                    or set(predicted_ids) != set(report["splits"]["test_gene_ids"])):
                raise ValueError("Resume final predictions do not match the test split")


def run_tuning_fold(
    dataset: DriverMONIDataset, base_config: dict[str, Any], search_config: dict[str, Any],
    fold_number: int, output_dir: str | Path, *, input_files: list[dict[str, Any]] | None = None,
    resume: bool = False, log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Search one outer fold and save candidate-boundary resumable provenance.

    ``output_dir`` is unique to this fold. A failed candidate restarts from its
    deterministic initial state; completed candidates require matching hashes.
    Predictions are probabilities [N], evaluated at the unchanged 0.5 threshold.
    """
    validate_config(base_config)
    if base_config["cross_validation"]["epoch_selection"] != "best_validation_roc_auc":
        raise ValueError("Tuning requires validation epoch selection")
    if base_config["training"]["preserve_eval_mode_after_first_evaluation"]:
        raise ValueError("Tuning requires C's training mode before every epoch")
    if base_config["evaluation"]["probability_threshold"] != 0.5:
        raise ValueError("Experiment D keeps the fixed 0.5 decision threshold")
    if type(fold_number) is not int or fold_number not in (1, 2, 3):
        raise ValueError("fold_number must be 1, 2 or 3")
    candidates = tuning_candidates(search_config)
    settings = base_config["training"]
    reproducibility = base_config["reproducibility"]
    fold_seed = (base_config["seed"] + fold_number - 1) % _MODULUS
    seed_everything(fold_seed, reproducibility["deterministic_algorithms"], reproducibility["cpu_threads"])
    device = select_device(base_config["device"])
    labels = np.array([int(dataset.context.loc[gene, "label"]) for gene in dataset.target_gene_ids])
    if min(class_counts(labels).values()) < 3:
        raise ValueError("Three-fold evaluation requires at least three genes of each class")
    train, validation, test, outer_train = _make_splits(labels, base_config)[fold_number - 1]
    assert validation is not None
    weights = class_weights(labels[train], device)
    splits = {
        f"{name}_gene_ids": [dataset.target_gene_ids[index] for index in indices]
        for name, indices in (("train", train), ("validation", validation), ("test", test), ("outer_train", outer_train))
    }
    splits.update({f"{name}_counts": class_counts(labels[indices])
                   for name, indices in (("train", train), ("validation", validation), ("test", test), ("outer_train", outer_train))})
    seeds = {"model": fold_seed, "train_loader": fold_seed,
             "validation_loader": (fold_seed + 10000) % _MODULUS,
             "test_loader": (fold_seed + 20000) % _MODULUS}
    provenance = {
        "format_version": _FORMAT_VERSION, "base_config": deepcopy(base_config),
        "search_config": deepcopy(search_config), "input_files": deepcopy(input_files or []),
        "dataset_sha256": _dataset_fingerprint(dataset), "source_sha256": _source_manifest(),
        "runtime": runtime_metadata(), "device": str(device), "fold": fold_number,
    }
    directory = Path(output_dir)
    path = directory / "report.json"
    if directory.exists() and any(directory.iterdir()) and not resume:
        raise FileExistsError(f"Output directory is not empty; use --resume explicitly: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    if resume and path.exists():
        report = json.loads(path.read_text())
        if report.get("provenance") != provenance or report.get("splits") != splits:
            raise ValueError("Cannot resume: configuration, input/data, source, runtime or splits changed")
        _verify_saved_candidates(report, candidates, directory)
        if report["status"] == "complete":
            log(f"Fold {fold_number}: verified completed report; nothing to resume")
            return report
        report.pop("error", None)
    elif resume and any(directory.iterdir()):
        raise ValueError("Cannot resume a nonempty directory without report.json")
    else:
        report = {
            "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "fold": fold_number, "provenance": provenance, "splits": splits,
            "class_weights": {"non_driver": weights[0].item(), "driver": weights[1].item()},
            "seeds": seeds, "candidate_results": [], "elapsed_seconds": 0.0,
            "protocol": {
                "selection": "maximum inner-validation AUC; earliest epoch within candidate; first grid candidate wins ties",
                "validation_split": "one stratified holdout within each outer training fold",
                "test_fold_used_for_selection": False, "refit_on_outer_training": False,
                "training_mode_every_epoch": True, "shared_graph_feature_context": True,
                "class_weight_population": "inner fitting targets only", "threshold": 0.5,
                "fixed_c_control": {"learning_rate": 0.001, "dropout_pair": [0.7, 0.6]},
                "seed_strategy": "(base seed + fold - 1) mod 2**31 reset per candidate; independent seeded loaders",
                "test_evaluations": "one for D winner and one for prespecified fixed C; reuse if winner is fixed C",
                "historical_c_rng_differs": True,
            },
        }
    report["status"] = "running"
    save_json(path, report)
    run_start = time.perf_counter()
    previous_seconds = report["elapsed_seconds"]

    def persist() -> None:
        report["elapsed_seconds"] = previous_seconds + time.perf_counter() - run_start
        save_json(path, report)

    def loader(indices: np.ndarray, shuffle: bool, seed: int) -> DataLoader:
        return DataLoader(Subset(dataset, indices.tolist()), batch_size=settings["batch_size"],
                          shuffle=shuffle, generator=torch.Generator().manual_seed(seed), num_workers=0,
                          drop_last=False, collate_fn=collate_gene_samples)

    def evaluate_state(result: dict[str, Any]) -> dict[str, Any]:
        model = _new_model(result, device)
        state = torch.load(directory / f"{result['candidate_id']}.pt", map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        evaluation = evaluate(model, loader(test, settings["shuffle_test"], seeds["test_loader"]), device,
                              0.5, base_config["evaluation"]["zero_division"])
        return {"candidate_id": result["candidate_id"], "learning_rate": result["learning_rate"],
                "dropout_pair": result["dropout_pair"], "selected_epoch": result["selected_epoch"],
                "metrics": evaluation.metrics, "predictions": evaluation.predictions}

    try:
        for candidate in candidates[len(report["candidate_results"]):]:
            log(f"Fold {fold_number} {candidate['candidate_id']}: lr={candidate['learning_rate']}, dropout={candidate['dropout_pair']}")
            seed_everything(fold_seed, reproducibility["deterministic_algorithms"], reproducibility["cpu_threads"])
            model = _new_model(candidate, device)
            initial_state_sha256 = _state_fingerprint(model)
            train_loader = loader(train, settings["shuffle_train"], seeds["train_loader"])
            validation_loader = loader(validation, False, seeds["validation_loader"])
            optimizer = torch.optim.Adam(model.parameters(), lr=candidate["learning_rate"],
                                         betas=tuple(settings["betas"]), eps=settings["eps"],
                                         weight_decay=settings["weight_decay"], amsgrad=settings["amsgrad"])
            best_result, best_state, selected_epoch = None, None, None
            history = []
            for epoch in range(1, settings["epochs"] + 1):
                epoch_start = time.perf_counter()
                model.train()
                loss = train_one_epoch(model, train_loader, optimizer, weights, device)
                evaluation = evaluate(model, validation_loader, device, 0.5, base_config["evaluation"]["zero_division"])
                history.append({"epoch": epoch, "training_mode_during_updates": True, "mean_batch_loss": loss,
                                "learning_rate": optimizer.param_groups[0]["lr"], "evaluation_split": "validation",
                                "validation_metrics": evaluation.metrics, "seconds": time.perf_counter() - epoch_start})
                if best_result is None or evaluation.metrics["auc"] > best_result.metrics["auc"]:
                    best_result, selected_epoch = evaluation, epoch
                    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                log(f"  epoch {epoch:02d}/{settings['epochs']} loss={loss:.6f} validation AUC={evaluation.metrics['auc']:.6f} seconds={history[-1]['seconds']:.1f}")
            checkpoint_hash = _save_state(directory / f"{candidate['candidate_id']}.pt", best_state)
            report["candidate_results"].append({
                **candidate, "selected_epoch": selected_epoch, "selected_validation_metrics": best_result.metrics,
                "validation_predictions": best_result.predictions, "epoch_history": history,
                "checkpoint_sha256": checkpoint_hash, "initial_state_sha256": initial_state_sha256,
            })
            persist()
        # Python's stable max preserves the first grid candidate on equal AUC.
        winner = max(report["candidate_results"], key=lambda result: result["selected_validation_metrics"]["auc"])
        control = next(result for result in report["candidate_results"]
                       if result["learning_rate"] == 0.001 and result["dropout_pair"] == [0.7, 0.6])
        report["selected_candidate_id"] = winner["candidate_id"]
        report["validation_ranking"] = [result["candidate_id"] for result in sorted(
            report["candidate_results"], key=lambda result: -result["selected_validation_metrics"]["auc"])]
        # Store each final evaluation atomically so resume does not repeat a saved test pass.
        if "tuned_test" not in report:
            report["tuned_test"] = evaluate_state(winner)
            persist()
        if "fixed_c_test" not in report:
            report["fixed_c_test"] = (deepcopy(report["tuned_test"]) if winner is control else evaluate_state(control))
            persist()
        report["test_evaluation_count"] = 1 if winner is control else 2
        report["status"] = "complete"
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        persist()
        log(f"Fold {fold_number} complete: {winner['candidate_id']} selected; D test AUC={report['tuned_test']['metrics']['auc']:.6f}; fixed C={report['fixed_c_test']['metrics']['auc']:.6f}")
        return report
    except BaseException as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        persist()
        raise


def aggregate_tuning_reports(folds: list[dict[str, Any]], ddof: int = 0) -> dict[str, Any]:
    """Aggregate exactly three complete compatible folds; reject overlapping tests."""
    if len(folds) != 3 or {fold.get("fold") for fold in folds} != {1, 2, 3}:
        raise ValueError("Aggregation requires folds 1, 2 and 3 exactly once")
    if type(ddof) is not int or ddof not in (0, 1):
        raise ValueError("ddof must be 0 or 1")
    folds = sorted(folds, key=lambda fold: fold["fold"])
    if any(fold.get("status") != "complete" for fold in folds):
        raise ValueError("Only complete folds may be aggregated")
    reference = {key: value for key, value in folds[0]["provenance"].items() if key != "fold"}
    if any({key: value for key, value in fold["provenance"].items() if key != "fold"} != reference for fold in folds):
        raise ValueError("Cannot aggregate folds with different provenance")
    test_ids = [gene for fold in folds for gene in fold["splits"]["test_gene_ids"]]
    expected = set(folds[0]["splits"]["outer_train_gene_ids"]) | set(folds[0]["splits"]["test_gene_ids"])
    if len(test_ids) != len(set(test_ids)) or set(test_ids) != expected:
        raise ValueError("Outer test folds must cover every target exactly once")
    paired = [{"fold": fold["fold"], "metrics": {
        metric: fold["tuned_test"]["metrics"][metric] - fold["fixed_c_test"]["metrics"][metric]
        for metric in fold["tuned_test"]["metrics"]
    }} for fold in folds]
    return {
        "status": "complete", "experiment": "D", "provenance": reference, "std_ddof": ddof,
        "folds": folds,
        "tuned_summary": summarize_folds([fold["tuned_test"] for fold in folds], ddof),
        "fixed_c_summary": summarize_folds([fold["fixed_c_test"] for fold in folds], ddof),
        "paired_differences": paired,
        "paired_difference_summary": summarize_folds(paired, ddof),
        "interpretation": "Validation-tuned D compared with prespecified fixed-C settings under matched per-fold seeds/loaders; historical C uses different RNG sequencing.",
    }
