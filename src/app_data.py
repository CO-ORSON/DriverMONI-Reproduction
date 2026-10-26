"""Lightweight, ordered BRCA feature inspection without importing the model."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .feature_schema import DNN_FEATURES, GAT_FEATURES


def read_features(path: str | Path) -> tuple[pd.DataFrame, str]:
    """Read genes and [N,13] features in model order; return the CSV SHA-256."""
    path = Path(path)
    contents = path.read_bytes()
    from io import BytesIO
    frame = pd.read_csv(BytesIO(contents))
    required = ("gene_ens", "gene_symbol") + GAT_FEATURES
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Feature CSV is missing columns: {', '.join(sorted(missing))}")
    if frame.empty or frame[["gene_ens", "gene_symbol"]].isna().any().any():
        raise ValueError("Feature CSV must contain genes with identifiers and symbols.")
    # Match prepare_gene_table's symbol deduplication before displaying targets.
    frame = frame.drop_duplicates("gene_symbol", keep="first").copy()
    if frame["gene_ens"].duplicated().any():
        raise ValueError("Feature CSV contains ambiguous duplicate Ensembl identifiers.")
    for name in ("gene_ens", "gene_symbol"):
        if not frame[name].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"{name} must contain nonempty text identifiers.")
    values = frame.loc[:, list(GAT_FEATURES)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.abs(values).max() > np.finfo(np.float32).max:
        raise ValueError("Features must be finite numbers representable as float32.")
    return frame.loc[:, list(required)].reset_index(drop=True), hashlib.sha256(contents).hexdigest()


def feature_rows(row: pd.Series) -> pd.DataFrame:
    """Return all 13 target attributes in the exact documented input order."""
    return pd.DataFrame({
        "Order": list(range(1, len(GAT_FEATURES) + 1)),
        "Feature": GAT_FEATURES,
        "Value": [float(row[name]) for name in GAT_FEATURES],
        "Branch": ["DNN + GAT" if name in DNN_FEATURES else "GAT only" for name in GAT_FEATURES],
    })


def graph_preview_dot(sample, symbols: dict[str, str], max_nodes: int = 25) -> str:
    """Draw a bounded actual-edge preview; omit self-loops and count duplicates.

    The inference graph is never changed. Full counts must be displayed beside
    this preview. ``sample`` is a GeneSample; torch/DGL remain lazy imports.
    """
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    ids = sample.node_gene_ids
    target = ids.index(sample.gene_id)
    kept = [target] + [i for i in range(len(ids)) if i != target][:max_nodes - 1]
    allowed = set(kept)
    lines = ['digraph { rankdir=LR; bgcolor="transparent";',
             'node [shape=box, style="rounded,filled", color="#9CA3AF", fontname="Helvetica", fontsize=11];',
             'edge [color="#9CA3AF", arrowsize=0.5];']
    for index in kept:
        label = symbols.get(ids[index], ids[index]) + (" (target)" if index == target else "")
        color = "#356596" if index == target else "#E5E7EB"
        text_color = "#FFFFFF" if index == target else "#26313D"
        lines.append(f'n{index} [label={json.dumps(label)}, fillcolor="{color}", fontcolor="{text_color}"];')
    sources, destinations = sample.graph.edges()
    counts: dict[tuple[int, int], int] = {}
    for source, destination in zip(sources.tolist(), destinations.tolist()):
        if source in allowed and destination in allowed and source != destination:
            counts[source, destination] = counts.get((source, destination), 0) + 1
    for (source, destination), count in counts.items():
        label = f' [label="×{count}"]' if count > 1 else ""
        lines.append(f"n{source} -> n{destination}{label};")
    lines.append("}")
    return "\n".join(lines)
