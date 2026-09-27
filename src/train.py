"""Source-compatible weighted training and three-fold BRCA cross-validation."""

from collections.abc import Callable, Iterable
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from .dataset import DriverMONIDataset, GeneBatch, collate_gene_samples
from .evaluate import METRIC_NAMES, evaluate
from .models import DriverMONI
from .utils import runtime_metadata, save_json, seed_everything, select_device, validate_config


def class_counts(labels: np.ndarray) -> dict[str, int]:
    return {"driver": int(np.sum(labels == 1)), "non_driver": int(np.sum(labels == 0))}


def class_weights(labels: np.ndarray, device: str | torch.device = "cpu") -> Tensor:
    """Training-fold inverse frequencies [w_non_driver, w_driver], float64."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2)
    if len(counts) != 2 or np.any(counts == 0):
        raise ValueError("Training requires both binary classes for class weighting")
    # Match the authors' operation order and float64 torch.tensor(dtype=float).
    return torch.tensor([(1 / int(count)) * (len(labels) / 2.0) for count in counts],
                        dtype=torch.float64, device=device)


def weighted_nll(logits: Tensor, labels: Tensor, weights: Tensor) -> Tensor:
    """Original loss: float64 log_softmax([B,2]) then weighted mean NLL."""
    return F.nll_loss(F.log_softmax(logits, dim=1, dtype=torch.float64), labels, weight=weights)


def train_one_epoch(
    model: nn.Module, loader: Iterable[GeneBatch], optimizer: torch.optim.Optimizer,
    weights: Tensor, device: str | torch.device,
) -> float:
    """Optimize without changing model mode; return mean batch loss as upstream."""
    total_loss, batches = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.gene_features, batch.graph)
        optimizer.zero_grad()
        loss = weighted_nll(logits, batch.labels, weights)
        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss is not finite")
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batches += 1
    if not batches:
        raise ValueError("Training loader is empty")
    return total_loss / batches


def summarize_folds(folds: list[dict[str, Any]], ddof: int) -> dict[str, dict[str, float]]:
    """Unweighted mean and explicitly configured standard deviation across folds."""
    return {
        metric: {"mean": float(np.mean([fold["metrics"][metric] for fold in folds])),
                 "std": float(np.std([fold["metrics"][metric] for fold in folds], ddof=ddof))}
        for metric in METRIC_NAMES
    }


def run_cross_validation(
    dataset: DriverMONIDataset,
    config: dict[str, Any],
    output_path: str | Path,
    *,
    input_files: list[dict[str, Any]] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run fresh models over three target-gene folds, retaining shared graph context.

    Scores come from the epoch with maximum held-out AUC, matching upstream.
    This is not an untouched-test-set protocol; the report records that fact.
    """
    validate_config(config)
    reproducibility = config["reproducibility"]
    seed_everything(config["seed"], reproducibility["deterministic_algorithms"], reproducibility["cpu_threads"])
    device = select_device(config["device"])
    labels = np.array([int(dataset.context.loc[gene, "label"]) for gene in dataset.target_gene_ids])
    counts = class_counts(labels)
    if min(counts.values()) < 3:
        raise ValueError("Three-fold ROC-AUC requires at least three genes in each class")
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=config["seed"])
    splits = list(splitter.split(np.zeros(len(labels)), labels))
    settings = config["training"]
    if any(len(train_indices) % settings["batch_size"] == 1 for train_indices, _ in splits):
        raise ValueError("A training fold ends in a singleton batch; source BatchNorm cannot train on it. Change batch_size explicitly.")
    report: dict[str, Any] = {
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "cancer": "BRCA", "config": deepcopy(config), "device": str(device),
        "runtime": runtime_metadata(), "input_files": input_files or [],
        "dataset_counts": counts, "num_targets": len(dataset),
        "protocol": {
            "selection": "maximum test-fold AUC across epochs; earliest epoch wins ties",
            "test_fold_used_for_epoch_selection": True,
            "seed_strategy": "seed Python/NumPy/PyTorch/CUDA/DGL once before CV; explicit splitter random_state",
            "shared_graph_feature_context": True, "std_ddof": config["evaluation"]["std_ddof"],
            "source_preserve_eval_mode": settings["preserve_eval_mode_after_first_evaluation"],
        },
        "folds": [],
    }
    save_json(output_path, report)
    start_time = time.perf_counter()
    try:
        for fold_number, (train_indices, test_indices) in enumerate(splits, 1):
            train_counts, test_counts = class_counts(labels[train_indices]), class_counts(labels[test_indices])
            log(f"Fold {fold_number}\n  train driver/non-driver: {train_counts['driver']}/{train_counts['non_driver']}"
                f"\n  test driver/non-driver: {test_counts['driver']}/{test_counts['non_driver']}")
            params = {"batch_size": settings["batch_size"], "num_workers": 0,
                      "drop_last": False, "collate_fn": collate_gene_samples}
            # Subset changes targets only; each item's neighbors retain the full context.
            train_loader = DataLoader(Subset(dataset, train_indices.tolist()), shuffle=settings["shuffle_train"], **params)
            test_loader = DataLoader(Subset(dataset, test_indices.tolist()), shuffle=settings["shuffle_test"], **params)
            weights = class_weights(labels[train_indices], device)
            model = DriverMONI().to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=settings["learning_rate"], betas=tuple(settings["betas"]),
                eps=settings["eps"], weight_decay=settings["weight_decay"], amsgrad=settings["amsgrad"],
            )
            best_result = None
            selected_epoch = None
            history = []
            for epoch in range(1, settings["epochs"] + 1):
                if epoch == 1 or not settings["preserve_eval_mode_after_first_evaluation"]:
                    model.train()
                training_mode = model.training
                epoch_start = time.perf_counter()
                loss = train_one_epoch(model, train_loader, optimizer, weights, device)
                evaluation = evaluate(model, test_loader, device, config["evaluation"]["probability_threshold"],
                                      config["evaluation"]["zero_division"])
                history.append({"epoch": epoch, "training_mode_during_updates": training_mode,
                                "mean_batch_loss": loss, "learning_rate": optimizer.param_groups[0]["lr"],
                                "metrics": evaluation.metrics, "seconds": time.perf_counter() - epoch_start})
                # First-epoch fallback also handles the upstream uninitialized-best edge case at AUC=0.
                if best_result is None or evaluation.metrics["auc"] > best_result.metrics["auc"]:
                    best_result, selected_epoch = evaluation, epoch
                log(f"  epoch {epoch:02d}/{settings['epochs']} loss={loss:.6f} "
                    f"AUC={evaluation.metrics['auc']:.6f} mode={'train' if training_mode else 'eval'} "
                    f"seconds={history[-1]['seconds']:.1f}")
            fold = {
                "fold": fold_number, "train_counts": train_counts, "test_counts": test_counts,
                "class_weights": {"non_driver": weights[0].item(), "driver": weights[1].item()},
                "selected_epoch": selected_epoch, "metrics": best_result.metrics,
                "epoch_history": history,
                "train_gene_ids": [dataset.target_gene_ids[index] for index in train_indices],
                "test_gene_ids": [dataset.target_gene_ids[index] for index in test_indices],
                "predictions": best_result.predictions,
            }
            report["folds"].append(fold)
            save_json(output_path, report)
            log(f"  selected epoch: {selected_epoch}")
            log("  " + " | ".join(f"{metric}={value:.6f}" for metric, value in fold["metrics"].items()))
        report["summary"] = summarize_folds(report["folds"], config["evaluation"]["std_ddof"])
        report["status"] = "complete"
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["elapsed_seconds"] = time.perf_counter() - start_time
        save_json(output_path, report)
        log(f"Mean ± std (ddof={config['evaluation']['std_ddof']}):")
        for metric, values in report["summary"].items():
            log(f"  {metric}: {values['mean']:.6f} ± {values['std']:.6f}")
        log(f"Saved: {output_path}")
        return report
    except BaseException as error:
        report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        save_json(output_path, report)
        raise
