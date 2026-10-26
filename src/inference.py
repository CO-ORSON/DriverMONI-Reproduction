"""Local, provenance-checked inference from a validation-selected fold-one model.

The checkpoint contains only tensors and JSON-compatible metadata. It is a demo
model trained on the first fold's fitting targets, not an all-data refit or a new
three-fold experiment. Predictions are uncalibrated positive-class softmax scores.
"""

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import pickle
import tempfile
import threading
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .dataset import DriverMONIDataset, collate_gene_samples, load_brca_dataset
from .evaluate import METRIC_NAMES, classification_metrics, evaluate
from .feature_schema import DNN_FEATURES, GAT_FEATURES
from .models import DriverMONI
from .train import _make_splits, class_counts, class_weights, train_one_epoch
from .utils import (
    input_manifest, project_path, runtime_metadata, save_json, seed_everything,
    select_device, validate_config,
)

CHECKPOINT_FORMAT = "drivermoni-demo-v1"
INPUT_KEYS = ("features", "labels", "comparison_genes", "gene_annotations", "ppi_edges", "neighbors")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _json_digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("Checkpoint state must map names to tensors")
        tensor = tensor.detach().cpu().contiguous()
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"Checkpoint tensor is not finite: {name}")
        digest.update(_json_bytes([name, str(tensor.dtype), list(tensor.shape)]))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def dataset_fingerprint(dataset: DriverMONIDataset) -> str:
    """Hash ordered targets, feature context, labels, graph and neighborhoods.

    This additionally checks the loaded dataset's semantics, independent of file
    location. File-byte hashes separately detect changes to the consumed inputs.
    """
    digest = hashlib.sha256()
    digest.update(_json_bytes(list(dataset.target_gene_ids)))
    digest.update(_json_bytes(list(dataset.context.index)))
    if "gene_symbol" in dataset.context:
        digest.update(_json_bytes(dataset.context["gene_symbol"].tolist()))
    digest.update(dataset.node_features.contiguous().numpy().tobytes())
    digest.update(dataset.context["label"].to_numpy(dtype=np.int64).tobytes())
    digest.update(_json_bytes(list(dataset.ppi.gene_ids)))
    for endpoints in dataset.ppi.graph.edges(order="eid"):
        digest.update(endpoints.numpy().tobytes())
    for gene in dataset.target_gene_ids:
        digest.update(_json_bytes([gene, sorted(set(dataset.neighbors[gene]))]))
    return digest.hexdigest()


def load_inference_dataset(config: dict[str, Any]) -> DriverMONIDataset:
    """Use exactly the existing runner's processed-data and graph loader."""
    data = config["data"]
    return load_brca_dataset(
        project_path(data["upstream_root"]),
        use_comparison_filter=data["use_comparison_filter"],
        use_supplied_neighbors=data["use_supplied_neighbors"],
        paths={key: data[key] for key in INPUT_KEYS},
    )


def _manifest_identity(files: list[dict[str, Any]]) -> dict[str, tuple[str, int]]:
    """Compare input content by role; relocation of identical inputs is allowed."""
    identity = {}
    for record in files:
        role = record["role"]
        if role in identity:
            raise ValueError(f"Duplicate input role in checkpoint: {role}")
        identity[role] = (record["sha256"], record["bytes"])
    return identity


def _save_demo_payload(output_path: Path, metadata: dict[str, Any], state: Mapping[str, torch.Tensor]) -> None:
    """Atomically store tensor weights and bound metadata, plus a readable copy."""
    state = {name: tensor.detach().cpu().clone() for name, tensor in state.items()}
    payload = {"format": CHECKPOINT_FORMAT, "metadata": metadata, "state_dict": state,
               "metadata_sha256": _json_digest(metadata), "state_sha256": _state_digest(state)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    save_json(output_path.with_suffix(".json"), dict(metadata, state_sha256=payload["state_sha256"],
                                                     metadata_sha256=payload["metadata_sha256"]))


def prepare_demo_checkpoint(
    dataset: DriverMONIDataset,
    config: dict[str, Any],
    output_path: str | Path,
    *,
    input_files: list[dict[str, Any]],
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Replay fold one of validation CV and export its complete selected state.

    Loader creation, initialization, random seeding, training/evaluation order,
    class weighting and earliest-max-AUC selection match run_cross_validation.
    All epochs run, then the outer test split is evaluated once. The original
    model, trainer and existing CV reports are never changed.
    """
    validate_config(config)
    if config["cross_validation"]["epoch_selection"] != "best_validation_roc_auc":
        raise ValueError("Demo export requires validation-based epoch selection (B or C config)")
    output_path = Path(output_path).expanduser().resolve()
    if output_path.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("Demo checkpoint output must end with .pt or .pth; metadata uses the companion .json")
    reproducibility = config["reproducibility"]
    seed_everything(config["seed"], reproducibility["deterministic_algorithms"], reproducibility["cpu_threads"])
    device = select_device(config["device"])
    labels = np.array([int(dataset.context.loc[gene, "label"]) for gene in dataset.target_gene_ids])
    if min(class_counts(labels).values()) < 3:
        raise ValueError("Three-fold splitting requires at least three genes in each class")
    train_indices, validation_indices, test_indices, outer_train_indices = _make_splits(labels, config)[0]
    settings = config["training"]
    params = {"batch_size": settings["batch_size"], "num_workers": 0,
              "drop_last": False, "collate_fn": collate_gene_samples}
    train_loader = DataLoader(Subset(dataset, train_indices.tolist()), shuffle=settings["shuffle_train"], **params)
    test_loader = DataLoader(Subset(dataset, test_indices.tolist()), shuffle=settings["shuffle_test"], **params)
    validation_loader = DataLoader(Subset(dataset, validation_indices.tolist()), shuffle=False, **params)
    weights = class_weights(labels[train_indices], device)
    model = DriverMONI().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings["learning_rate"], betas=tuple(settings["betas"]),
        eps=settings["eps"], weight_decay=settings["weight_decay"], amsgrad=settings["amsgrad"],
    )
    best_result, best_state, selected_epoch = None, None, None
    history = []
    for epoch in range(1, settings["epochs"] + 1):
        if epoch == 1 or not settings["preserve_eval_mode_after_first_evaluation"]:
            model.train()
        training_mode = model.training
        loss = train_one_epoch(model, train_loader, optimizer, weights, device)
        result = evaluate(model, validation_loader, device, config["evaluation"]["probability_threshold"],
                          config["evaluation"]["zero_division"])
        history.append({"epoch": epoch, "training_mode_during_updates": training_mode,
                        "mean_batch_loss": loss, "learning_rate": optimizer.param_groups[0]["lr"],
                        "validation_metrics": result.metrics, "evaluation_split": "validation"})
        if best_result is None or result.metrics["auc"] > best_result.metrics["auc"]:
            best_result, best_state, selected_epoch = result, deepcopy(model.state_dict()), epoch
        log(f"Demo fold 1 epoch {epoch}/{settings['epochs']}: loss={loss:.6f}, validation AUC={result.metrics['auc']:.6f}")
    model.load_state_dict(best_state)
    test_result = evaluate(model, test_loader, device, config["evaluation"]["probability_threshold"],
                           config["evaluation"]["zero_division"])
    gene_ids = lambda indices: [dataset.target_gene_ids[index] for index in indices]
    metadata = {
        "format": CHECKPOINT_FORMAT, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "single validation-selected fold-one demonstration; not all-data refit or full CV results",
        "experiment": "B" if settings["preserve_eval_mode_after_first_evaluation"] else "C",
        "fold": 1, "config": deepcopy(config), "runtime": runtime_metadata(), "device": str(device),
        "feature_schema": {"dnn": list(DNN_FEATURES), "gat": list(GAT_FEATURES)},
        "input_files": deepcopy(input_files), "dataset_sha256": dataset_fingerprint(dataset),
        "num_targets": len(dataset), "dataset_counts": class_counts(labels),
        "train_gene_ids": gene_ids(train_indices), "validation_gene_ids": gene_ids(validation_indices),
        "test_gene_ids": gene_ids(test_indices), "outer_train_gene_ids": gene_ids(outer_train_indices),
        "train_counts": class_counts(labels[train_indices]),
        "validation_counts": class_counts(labels[validation_indices]),
        "test_counts": class_counts(labels[test_indices]),
        "class_weights": {"non_driver": weights[0].item(), "driver": weights[1].item()},
        "selected_epoch": selected_epoch, "epoch_history": history,
        "selected_validation_metrics": best_result.metrics, "test_metrics": test_result.metrics,
        "validation_predictions": best_result.predictions, "test_predictions": test_result.predictions,
        "score": "uncalibrated positive-class softmax; not a clinical probability",
        "test_evaluations": 1, "shared_graph_feature_context": True,
    }
    _save_demo_payload(output_path, metadata, model.state_dict())
    log(f"Saved demo checkpoint: {output_path} (selected epoch {selected_epoch})")
    return metadata


def _verify_recorded_metrics(records: list[dict[str, Any]], metrics: dict[str, float],
                             ids: list[str], zero_division: int) -> None:
    observed = [record["gene_id"] for record in records]
    if len(observed) != len(set(observed)) or set(observed) != set(ids):
        raise ValueError("D predictions do not match their recorded split")
    if any(type(record["label"]) is not int or record["label"] not in (0, 1) for record in records):
        raise ValueError("D prediction labels must be binary integers")
    calculated = classification_metrics([record["label"] for record in records],
                                         [record["driver_probability"] for record in records], 0.5, zero_division)
    if set(metrics) != set(METRIC_NAMES) or any(
        not np.isclose(metrics[name], calculated[name], rtol=1e-10, atol=1e-12) for name in METRIC_NAMES
    ):
        raise ValueError("D recorded metrics disagree with saved probabilities")


def _verify_tuned_report(report: dict[str, Any], directory: Path | None = None) -> dict[str, Any]:
    """Validate complete fold-one D selection and return its validation winner."""
    from .tuning import _source_manifest, _verify_saved_candidates, tuning_candidates

    if report["status"] != "complete" or report["fold"] != 1 or report["provenance"]["fold"] != 1:
        raise ValueError("D demo requires the completed prespecified fold-one report")
    provenance = report["provenance"]
    config = provenance["base_config"]
    validate_config(config)
    if (provenance["format_version"] != 1
            or config["training"]["preserve_eval_mode_after_first_evaluation"]
            or config["cross_validation"]["epoch_selection"] != "best_validation_roc_auc"
            or config["evaluation"]["probability_threshold"] != 0.5):
        raise ValueError("D requires validation selection, training mode every epoch and threshold 0.5")
    if provenance["source_sha256"] != _source_manifest():
        raise ValueError("D source fingerprints differ from the code that produced the report")
    seed = config["seed"]
    expected_seeds = {"model": seed, "train_loader": seed,
                      "validation_loader": (seed + 10000) % 2**31,
                      "test_loader": (seed + 20000) % 2**31}
    if report["seeds"] != expected_seeds:
        raise ValueError("D model/loader RNG seeds disagree with its fold-one protocol")
    protocol = report["protocol"]
    if (protocol["test_fold_used_for_selection"] is not False
            or protocol["refit_on_outer_training"] is not False
            or protocol["training_mode_every_epoch"] is not True
            or protocol["shared_graph_feature_context"] is not True or protocol["threshold"] != 0.5):
        raise ValueError("D selection protocol is inconsistent")
    candidates = tuning_candidates(provenance["search_config"])
    results = report["candidate_results"]
    if len(results) != len(candidates):
        raise ValueError("D report must contain the entire completed candidate search")
    for expected, result in zip(candidates, results, strict=True):
        if any(result[key] != value for key, value in expected.items()):
            raise ValueError("D candidate settings/order disagree with the recorded search")
        history = result["epoch_history"]
        if len(history) != config["training"]["epochs"]:
            raise ValueError("D candidate epoch history is incomplete")
        for epoch, row in enumerate(history, 1):
            if (row["epoch"] != epoch or row["training_mode_during_updates"] is not True
                    or row["evaluation_split"] != "validation" or row["learning_rate"] != expected["learning_rate"]
                    or not np.isfinite(row["mean_batch_loss"])):
                raise ValueError("D candidate epoch history is inconsistent")
            if set(row["validation_metrics"]) != set(METRIC_NAMES) or any(
                not isinstance(value, (float, int)) or not np.isfinite(value) or not 0 <= value <= 1
                for value in row["validation_metrics"].values()
            ):
                raise ValueError("D candidate validation metrics are invalid")
        best = max(history, key=lambda row: row["validation_metrics"]["auc"])
        if result["selected_epoch"] != best["epoch"] or result["selected_validation_metrics"] != best["validation_metrics"]:
            raise ValueError("D epoch must be the earliest maximum validation AUC")
        _verify_recorded_metrics(result["validation_predictions"], result["selected_validation_metrics"],
                                 report["splits"]["validation_gene_ids"], config["evaluation"]["zero_division"])
    if len({result["initial_state_sha256"] for result in results}) != 1:
        raise ValueError("D candidates must share the same paired initialization")
    winner = max(results, key=lambda result: result["selected_validation_metrics"]["auc"])
    ranking = [result["candidate_id"] for result in sorted(results, key=lambda item: -item["selected_validation_metrics"]["auc"])]
    if report["selected_candidate_id"] != winner["candidate_id"] or report["validation_ranking"] != ranking:
        raise ValueError("D selected candidate/ranking must follow validation AUC with first-candidate ties")
    control = next(result for result in results if result["learning_rate"] == 0.001 and result["dropout_pair"] == [0.7, 0.6])
    if report["test_evaluation_count"] != (1 if winner is control else 2):
        raise ValueError("D final evaluation count disagrees with winner/control selection")
    for name, selected in (("tuned_test", winner), ("fixed_c_test", control)):
        final = report[name]
        if any(final[key] != selected[key] for key in ("candidate_id", "learning_rate", "dropout_pair", "selected_epoch")):
            raise ValueError("D final result does not match its validation-selected checkpoint")
        _verify_recorded_metrics(final["predictions"], final["metrics"], report["splits"]["test_gene_ids"],
                                 config["evaluation"]["zero_division"])
    if directory is not None:
        _verify_saved_candidates(report, candidates, directory)
    return winner


def _verify_tuned_dataset(report: dict[str, Any], dataset: DriverMONIDataset) -> None:
    from .tuning import _dataset_fingerprint, _state_fingerprint

    config = report["provenance"]["base_config"]
    if _dataset_fingerprint(dataset) != report["provenance"]["dataset_sha256"]:
        raise ValueError("D dataset fingerprint differs from the original tuning inputs")
    labels = np.array([int(dataset.context.loc[gene, "label"]) for gene in dataset.target_gene_ids])
    train, validation, test, outer = _make_splits(labels, config)[0]
    for name, indices in (("train", train), ("validation", validation), ("test", test), ("outer_train", outer)):
        if (report["splits"][f"{name}_gene_ids"] != [dataset.target_gene_ids[index] for index in indices]
                or report["splits"][f"{name}_counts"] != class_counts(labels[indices])):
            raise ValueError("D saved split does not match its configured fold-one assignment")
    weights = class_weights(labels[train])
    if report["class_weights"] != {"non_driver": weights[0].item(), "driver": weights[1].item()}:
        raise ValueError("D class weights do not match the fitting targets")
    records = [candidate["validation_predictions"] for candidate in report["candidate_results"]]
    records += [report[name]["predictions"] for name in ("tuned_test", "fixed_c_test")]
    if any(record["label"] != int(dataset.context.loc[record["gene_id"], "label"])
           for group in records for record in group):
        raise ValueError("D saved labels disagree with the input labels")
    # Recreate the expected initialization without disturbing application RNG.
    with torch.random.fork_rng(devices=[]):
        # The model initializes on CPU; avoid torch.manual_seed's CUDA side effect.
        torch.default_generator.manual_seed(report["seeds"]["model"])
        initial = _state_fingerprint(DriverMONI())
    if any(candidate["initial_state_sha256"] != initial for candidate in report["candidate_results"]):
        raise ValueError("D initialization fingerprint disagrees with the recorded model seed")


def prepare_tuned_demo_checkpoint(
    report_path: str | Path, output_path: str | Path, *, data_root: str | Path | None = None,
    device: str = "cpu", log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Import D's completed fold-one winner and verify its scores; never train.

    Selection is read from the validation report, never inferred from test AUC.
    All source/input/candidate hashes are checked before the selected full state
    is copied. Verification reuses D's explicit evaluation-loader seed/order.
    """
    report_path = Path(report_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if output_path.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("Demo checkpoint output must end with .pt or .pth")
    if output_path.name == "brca_demo.pt" or output_path.exists():
        raise FileExistsError("D export needs a new output path; it cannot overwrite B or an existing checkpoint")
    raw = report_path.read_bytes()
    report = json.loads(raw)
    try:
        winner = _verify_tuned_report(report, report_path.parent)
        config = deepcopy(report["provenance"]["base_config"])
        if data_root is not None:
            config["data"]["upstream_root"] = str(Path(data_root).expanduser().resolve())
        files = input_manifest(config["data"])
        if _manifest_identity(files) != _manifest_identity(report["provenance"]["input_files"]):
            raise ValueError("D input files do not match the original tuning hashes")
        dataset = load_inference_dataset(config)
        _verify_tuned_dataset(report, dataset)
        if files != input_manifest(config["data"]):
            raise ValueError("D input files changed while loading")
        checkpoint = report_path.parent / f"{winner['candidate_id']}.pt"
        checkpoint_bytes = checkpoint.read_bytes()
        if hashlib.sha256(checkpoint_bytes).hexdigest() != winner["checkpoint_sha256"]:
            raise ValueError("D selected checkpoint bytes no longer match the recorded hash")
        state = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
        state_sha = _state_digest(state)
        target_device = select_device(device)
        threads = config["reproducibility"]["cpu_threads"]
        if target_device.type == "cpu" and threads is not None:
            torch.set_num_threads(threads)
        with torch.random.fork_rng(devices=[]):
            model = DriverMONI()
        model.dnn.dropout1.p, model.dnn.dropout2.p = winner["dropout_pair"]
        _load_model_state(model, state)
        model.to(target_device).eval()
        verified = {}
        for name, recorded in (("validation", {"metrics": winner["selected_validation_metrics"],
                                               "predictions": winner["validation_predictions"]}),
                               ("test", report["tuned_test"])):
            indices = [dataset.target_gene_ids.index(gene) for gene in report["splits"][f"{name}_gene_ids"]]
            loader = DataLoader(Subset(dataset, indices), batch_size=config["training"]["batch_size"],
                                shuffle=config["training"]["shuffle_test"] if name == "test" else False,
                                generator=torch.Generator().manual_seed(report["seeds"][f"{name}_loader"]),
                                num_workers=0, drop_last=False, collate_fn=collate_gene_samples)
            result = evaluate(model, loader, target_device, 0.5, config["evaluation"]["zero_division"])
            if ([row["gene_id"] for row in result.predictions] != [row["gene_id"] for row in recorded["predictions"]]
                    or any(not np.isclose(result.metrics[key], recorded["metrics"][key], rtol=1e-8, atol=1e-10) for key in METRIC_NAMES)
                    or any(not np.isclose(current["driver_probability"], previous["driver_probability"], rtol=1e-6, atol=1e-7)
                           for current, previous in zip(result.predictions, recorded["predictions"], strict=True))):
                raise ValueError(f"D checkpoint does not reproduce saved {name} scores/metrics")
            verified[name] = {"num_predictions": len(result.predictions), "metrics": result.metrics,
                              "max_score_difference": max(abs(a["driver_probability"] - b["driver_probability"])
                                                          for a, b in zip(result.predictions, recorded["predictions"], strict=True))}
            log(f"Verified D fold-one {name}: {len(result.predictions)} predictions; AUC={result.metrics['auc']:.6f}")
    except (KeyError, TypeError, AttributeError, IndexError, pickle.UnpicklingError) as error:
        raise ValueError(f"Incomplete or malformed D report/checkpoint: {error}") from error
    metadata = {
        "format": CHECKPOINT_FORMAT, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "prespecified fold-one D demonstration; not a global-best fold or all-data refit",
        "experiment": "D", "fold": 1, "config": config,
        "selected_hyperparameters": {key: winner[key] for key in ("candidate_id", "learning_rate", "dropout_pair")},
        "runtime": deepcopy(report["provenance"]["runtime"]), "device": str(target_device),
        "feature_schema": {"dnn": list(DNN_FEATURES), "gat": list(GAT_FEATURES)},
        "input_files": files, "dataset_sha256": dataset_fingerprint(dataset),
        "num_targets": len(dataset), "dataset_counts": class_counts(np.array(dataset.context.loc[list(dataset.target_gene_ids), "label"])),
        **deepcopy(report["splits"]), "class_weights": deepcopy(report["class_weights"]),
        "selected_epoch": winner["selected_epoch"], "epoch_history": deepcopy(winner["epoch_history"]),
        "selected_validation_metrics": deepcopy(winner["selected_validation_metrics"]),
        "validation_predictions": deepcopy(winner["validation_predictions"]),
        "test_metrics": deepcopy(report["tuned_test"]["metrics"]), "test_predictions": deepcopy(report["tuned_test"]["predictions"]),
        "score": "uncalibrated positive-class softmax; not a clinical probability",
        "test_evaluations": 1, "shared_graph_feature_context": True,
        "tuning_report": report, "tuning_report_sha256": hashlib.sha256(raw).hexdigest(),
        "tuning_report_content_sha256": _json_digest(report),
        "selected_checkpoint_sha256": winner["checkpoint_sha256"], "selected_state_sha256": state_sha,
        "verification": verified,
    }
    _save_demo_payload(output_path, metadata, state)
    log(f"Saved D demo checkpoint: {output_path}; {winner['candidate_id']}, epoch {winner['selected_epoch']}")
    return metadata


def _load_model_state(model: DriverMONI, state: Mapping[str, torch.Tensor]) -> None:
    for name, expected in model.state_dict().items():
        if name in state and state[name].dtype != expected.dtype:
            raise ValueError(f"Checkpoint tensor dtype does not match DriverMONI: {name}")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"Checkpoint tensor shapes or keys do not match DriverMONI: {error}") from error


def _verify_tuned_metadata(metadata: dict[str, Any], state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    report = metadata["tuning_report"]
    if _json_digest(report) != metadata["tuning_report_content_sha256"]:
        raise ValueError("D embedded selection report integrity check failed")
    winner = _verify_tuned_report(report)
    expected_config = deepcopy(report["provenance"]["base_config"])
    # Only the data directory can be relocated; scientific settings remain bound.
    expected_config["data"]["upstream_root"] = metadata["config"]["data"]["upstream_root"]
    if expected_config != metadata["config"]:
        raise ValueError("D base configuration differs from its selection report")
    expected = {
        "selected_hyperparameters": {key: winner[key] for key in ("candidate_id", "learning_rate", "dropout_pair")},
        "selected_epoch": winner["selected_epoch"], "epoch_history": winner["epoch_history"],
        "selected_validation_metrics": winner["selected_validation_metrics"],
        "validation_predictions": winner["validation_predictions"],
        "test_metrics": report["tuned_test"]["metrics"], "test_predictions": report["tuned_test"]["predictions"],
        "class_weights": report["class_weights"], "selected_checkpoint_sha256": winner["checkpoint_sha256"],
        **report["splits"],
    }
    if any(metadata[key] != value for key, value in expected.items()):
        raise ValueError("D demo metadata disagrees with its validation-selected candidate/dropout/epoch")
    if _state_digest(state) != metadata["selected_state_sha256"]:
        raise ValueError("D weights differ from the imported validation-selected state")
    if _manifest_identity(metadata["input_files"]) != _manifest_identity(report["provenance"]["input_files"]):
        raise ValueError("D demo inputs differ from its tuning report")
    return winner


@dataclass(frozen=True)
class InferenceBundle:
    """Validated model plus the exact full-context dataset used to prepare it."""

    model: DriverMONI
    dataset: DriverMONIDataset
    metadata: dict[str, Any]
    device: torch.device
    checkpoint_path: Path
    lock: Any = field(default_factory=threading.RLock, repr=False, compare=False)


def load_inference_bundle(
    checkpoint_path: str | Path, *, data_root: str | Path | None = None, device: str = "cpu",
) -> InferenceBundle:
    """Load local weights safely and reject changed input files/features/graphs.

    The checkpoint must be produced by one of this module's exporters. ``data_root``
    can relocate identical input files; their configured relative paths remain.
    Only tensors and primitive metadata are deserialized, with CPU as default.
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Demo checkpoint not found: {checkpoint_path}. Run scripts/prepare_demo_model.py first.")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError, RuntimeError, OSError) as error:
        raise ValueError("Checkpoint could not be read safely; prepare it again with scripts/prepare_demo_model.py") from error
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported demo checkpoint format")
    try:
        metadata, state = payload["metadata"], payload["state_dict"]
        if _json_digest(metadata) != payload["metadata_sha256"] or _state_digest(state) != payload["state_sha256"]:
            raise ValueError("Checkpoint integrity check failed: weights or metadata changed")
        if metadata["feature_schema"] != {"dnn": list(DNN_FEATURES), "gat": list(GAT_FEATURES)}:
            raise ValueError("Checkpoint feature schema/order does not match this model")
        config = deepcopy(metadata["config"])
        validate_config(config)
        if config["cross_validation"]["epoch_selection"] != "best_validation_roc_auc" or metadata["fold"] != 1:
            raise ValueError("Checkpoint must be a validation-selected fold-one demo model")
        experiment = metadata["experiment"]
        if experiment == "D":
            _verify_tuned_metadata(metadata, state)
        elif experiment in {"B", "C"}:
            expected_experiment = "B" if config["training"]["preserve_eval_mode_after_first_evaluation"] else "C"
            if experiment != expected_experiment or "tuning_report" in metadata or "selected_hyperparameters" in metadata:
                raise ValueError("Checkpoint experiment label disagrees with its training provenance")
        else:
            raise ValueError("Demo checkpoint experiment must be B, C or D")
        if data_root is not None:
            config["data"]["upstream_root"] = str(Path(data_root).expanduser().resolve())
        current_files = input_manifest(config["data"])
        if _manifest_identity(current_files) != _manifest_identity(metadata["input_files"]):
            raise ValueError("Input files do not match checkpoint hashes; restore the training inputs or prepare a new checkpoint")
        dataset = load_inference_dataset(config)
        if dataset_fingerprint(dataset) != metadata["dataset_sha256"]:
            raise ValueError("Dataset features, labels, graph or target ordering differ from this checkpoint")
        if experiment == "D":
            _verify_tuned_dataset(metadata["tuning_report"], dataset)
        labels = np.array([int(dataset.context.loc[gene, "label"]) for gene in dataset.target_gene_ids])
        train, validation, test, outer = _make_splits(labels, config)[0]
        for key, indices in (("train_gene_ids", train), ("validation_gene_ids", validation),
                             ("test_gene_ids", test), ("outer_train_gene_ids", outer)):
            if metadata[key] != [dataset.target_gene_ids[index] for index in indices]:
                raise ValueError(f"Checkpoint split does not match its configuration: {key}")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"Incomplete or malformed demo checkpoint: {error}") from error
    target_device = select_device(device)
    # Tiny per-gene workloads are slower with many BLAS threads. Use the
    # checkpoint's explicit CPU setting, also matching the recorded experiment.
    cpu_threads = config["reproducibility"]["cpu_threads"]
    if target_device.type == "cpu" and cpu_threads is not None:
        torch.set_num_threads(cpu_threads)
    with torch.random.fork_rng(devices=[]):
        model = DriverMONI()
    if experiment == "D":
        model.dnn.dropout1.p, model.dnn.dropout2.p = metadata["selected_hyperparameters"]["dropout_pair"]
    _load_model_state(model, state)
    model.to(target_device).eval()
    return InferenceBundle(model, dataset, metadata, target_device, checkpoint_path)


@torch.inference_mode()
def predict_gene(bundle: InferenceBundle, gene_id: str) -> dict[str, Any]:
    """Run one target's DNN [1,12] and ego graph [N,13] to logits [1,2]."""
    if gene_id not in bundle.dataset.target_gene_ids:
        raise ValueError(f"Gene is not a target in this checkpoint's dataset: {gene_id}")
    sample = bundle.dataset[bundle.dataset.target_gene_ids.index(gene_id)]
    batch = collate_gene_samples([sample]).to(bundle.device)
    with bundle.lock:
        bundle.model.eval()
        logits = bundle.model(batch.gene_features, batch.graph)
    if logits.shape != (1, 2) or not torch.isfinite(logits).all():
        raise ValueError("Model must return finite logits [1,2]")
    score = float(torch.softmax(logits, dim=1)[0, 1].cpu())
    threshold = bundle.metadata["config"]["evaluation"]["probability_threshold"]
    split = next(name for name, key in (("fit", "train_gene_ids"), ("validation", "validation_gene_ids"),
                                       ("held_out_test", "test_gene_ids")) if gene_id in bundle.metadata[key])
    target = sample.graph.ndata["feat"][sample.graph.ndata["is_target"].bool()][0]
    return {
        "gene_id": gene_id, "gene_symbol": str(bundle.dataset.context.loc[gene_id].get("gene_symbol", gene_id)),
        "label": sample.label, "driver_score": score, "predicted_class": int(score >= threshold),
        "threshold": threshold, "split": split,
        "num_nodes": sample.graph.num_nodes(), "num_edges": sample.graph.num_edges(),
        "dnn_features": dict(zip(DNN_FEATURES, sample.gene_features.tolist(), strict=True)),
        "target_node_features": dict(zip(GAT_FEATURES, target.tolist(), strict=True)),
        "logits": logits[0].cpu().tolist(),
    }
