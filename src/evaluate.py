"""Positive-class score evaluation, retaining the upstream eval-mode side effect."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, auc, balanced_accuracy_score, f1_score, precision_score,
    recall_score, roc_curve,
)
from torch import nn

from .dataset import GeneBatch

METRIC_NAMES = ("auc", "accuracy", "precision", "recall", "f1", "balanced_accuracy")


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics and per-gene scores from one complete evaluation pass."""

    metrics: dict[str, float]
    predictions: list[dict[str, str | int | float]]


def classification_metrics(
    labels: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
    threshold: float = 0.5,
    zero_division: int = 0,
) -> dict[str, float]:
    """ROC-AUC uses driver probabilities [N]; other metrics use scores >= threshold."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels):
        raise ValueError("Labels and scores must be nonempty matching vectors [N]")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("ROC-AUC evaluation requires both binary classes")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Driver probabilities must be finite and between zero and one")
    predictions = (scores >= threshold).astype(np.int64)
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    return {
        "auc": float(auc(false_positive_rate, true_positive_rate)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=zero_division)),
        "recall": float(recall_score(labels, predictions, zero_division=zero_division)),
        "f1": float(f1_score(labels, predictions, zero_division=zero_division)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable[GeneBatch],
    device: torch.device | str,
    threshold: float = 0.5,
    zero_division: int = 0,
) -> EvaluationResult:
    """Evaluate logits [B,2], returning metrics and driver scores with gene IDs.

    Like the authors' test(), this leaves the model in eval mode. The training
    protocol decides explicitly whether to restore train mode next epoch.
    """
    model.eval()
    labels: list[int] = []
    scores: list[float] = []
    records: list[dict[str, str | int | float]] = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.gene_features, batch.graph)
        if logits.shape != (len(batch.gene_ids), 2):
            raise ValueError("Model logits must have shape [B, 2]")
        probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
        batch_labels = batch.labels.cpu().tolist()
        labels.extend(batch_labels)
        scores.extend(probabilities)
        records.extend(
            {"gene_id": gene, "label": int(label), "driver_probability": float(score)}
            for gene, label, score in zip(batch.gene_ids, batch_labels, probabilities)
        )
    return EvaluationResult(classification_metrics(labels, scores, threshold, zero_division), records)
