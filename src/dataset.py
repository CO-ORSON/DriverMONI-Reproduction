"""Processed BRCA classification samples, shared graph context, and batching."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .graph_utils import PPIGraph, load_humannet_edges, load_neighbor_cache


# Exact order: CV_multimodal_compare.py, __getitem__ line 513 and
# _prepare_subgraphs line 460 at ddb0a830158ba41c90019bd5ca92054c2ba0869b.
# 'deg' is differential expression, not graph degree. No re-scaling is applied.
DNN_FEATURES: tuple[str, ...] = (
    "synonymous_variant", "stop_gained", "missense_variant", "frameshift_variant",
    "splice", "inframe", "lost_stop and start", "deg", "related_pathway",
    "dir_pathway", "muta_count", "miss_ratio",
)
GAT_FEATURES: tuple[str, ...] = DNN_FEATURES + ("PPI",)
assert len(DNN_FEATURES) == 12
assert len(GAT_FEATURES) == 13


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def prepare_gene_table(
    features: pd.DataFrame,
    driver_gene_ids: Sequence[str],
    comparison_gene_ids: Sequence[str] | None = None,
    allowed_symbols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Apply upstream symbol deduplication, comparison filter, and binary labels.

    Return the full context table before target-gene CV splitting. Class 0 means
    absent from the supplied driver list, not a proven biological non-driver.
    """
    _require_columns(features, ("gene_ens", "gene_symbol") + GAT_FEATURES, "Feature table")
    if features[["gene_ens", "gene_symbol"]].isna().any().any():
        raise ValueError("Gene identifiers and symbols must not be missing")
    table = features.drop_duplicates(subset="gene_symbol", keep="first").copy()
    if (comparison_gene_ids is None) != (allowed_symbols is None):
        raise ValueError("Comparison gene IDs and allowed symbols must be supplied together")
    if comparison_gene_ids is not None:
        # Important: build this lookup from the original table, before symbol deduplication.
        lookup = features.drop_duplicates("gene_ens").set_index("gene_ens")
        missing = set(comparison_gene_ids) - set(lookup.index)
        if missing:
            raise ValueError(
                f"Comparison list requests {len(missing)} absent genes; "
                f"examples: {sorted(missing)[:5]}"
            )
        symbols = set(lookup.loc[list(comparison_gene_ids), "gene_symbol"]) & set(allowed_symbols)
        table = table.loc[table["gene_symbol"].isin(symbols)].copy()
    driver_genes = set(driver_gene_ids)
    table["label"] = table["gene_ens"].isin(driver_genes).astype(np.int64)
    if table.empty:
        raise ValueError("No target genes remain after filtering")
    return table.reset_index(drop=True)


@dataclass(frozen=True)
class GeneSample:
    """One target: DNN [12], graph nodes [N,13], class index, and stable gene IDs."""

    gene_features: Tensor
    graph: dgl.DGLGraph
    label: int
    gene_id: str
    node_gene_ids: tuple[str, ...]


@dataclass(frozen=True)
class GeneBatch:
    """Collated features [B,12], B component graphs, labels [B], and gene IDs."""

    gene_features: Tensor
    graph: dgl.DGLGraph
    labels: Tensor
    gene_ids: tuple[str, ...]

    def to(self, device: str | torch.device) -> "GeneBatch":
        """Move features, graph, and labels together; IDs remain Python strings."""
        return GeneBatch(
            self.gene_features.to(device), self.graph.to(device),
            self.labels.to(device), self.gene_ids,
        )


def collate_gene_samples(samples: Sequence[GeneSample]) -> GeneBatch:
    """Batch samples in their supplied order, retaining one target per graph."""
    if not samples:
        raise ValueError("Cannot collate an empty sample batch")
    return GeneBatch(
        torch.stack([sample.gene_features for sample in samples]),
        dgl.batch([sample.graph for sample in samples]),
        torch.tensor([sample.label for sample in samples], dtype=torch.long),
        tuple(sample.gene_id for sample in samples),
    )


class DriverMONIDataset(Dataset[GeneSample]):
    """One target per item, with graph features from the entire context table.

    Restrict ``target_gene_ids`` for a CV fold; keep ``gene_table`` shared between
    folds as upstream does. Target labels are never attached as node attributes.
    """

    def __init__(
        self,
        gene_table: pd.DataFrame,
        ppi: PPIGraph,
        neighbors: Mapping[str, Sequence[str]],
        target_gene_ids: Sequence[str] | None = None,
    ) -> None:
        _require_columns(gene_table, ("gene_ens", "label") + GAT_FEATURES, "Gene table")
        self.gene_table = gene_table.copy().reset_index(drop=True)
        self.context = self.gene_table.drop_duplicates("gene_ens").set_index("gene_ens")
        self.available_genes = set(self.context.index)
        self.target_gene_ids = tuple(
            self.gene_table["gene_ens"] if target_gene_ids is None else target_gene_ids
        )
        if not self.target_gene_ids:
            raise ValueError("Dataset must contain at least one target gene")
        if not self.available_genes <= set(ppi.gene_to_node):
            raise ValueError("Some feature genes have no global PPI node")
        if not set(self.target_gene_ids) <= self.available_genes:
            raise ValueError("Target genes must exist in the feature context")
        missing_cache = set(self.target_gene_ids) - set(neighbors)
        if missing_cache:
            raise ValueError(f"Neighbor cache is missing target genes: {sorted(missing_cache)[:5]}")
        if not self.context["label"].isin([0, 1]).all():
            raise ValueError("Labels must be binary class indices 0 or 1")
        try:
            values = self.context.loc[:, list(GAT_FEATURES)].to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("All model feature columns must be numeric") from error
        if not np.isfinite(values).all():
            raise ValueError("Model feature columns must not contain NaN or infinity")
        self.node_features = torch.tensor(values, dtype=torch.float32)
        if not torch.isfinite(self.node_features).all():
            raise ValueError("Model features overflow float32")
        self.gene_to_feature_row = {gene: index for index, gene in enumerate(self.context.index)}
        self.ppi = ppi
        self.neighbors = neighbors

    def __len__(self) -> int:
        return len(self.target_gene_ids)

    def __getitem__(self, index: int) -> GeneSample:
        target = self.target_gene_ids[index]
        graph = self.ppi.ego_graph(target, self.neighbors, self.available_genes)
        node_gene_ids = self.ppi.node_gene_ids(graph)
        feature_rows = [self.gene_to_feature_row[gene] for gene in node_gene_ids]
        graph.ndata["feat"] = self.node_features[feature_rows].clone()
        target_row = self.gene_to_feature_row[target]
        return GeneSample(
            gene_features=self.node_features[target_row, :len(DNN_FEATURES)].clone(),
            graph=graph,
            label=int(self.context.loc[target, "label"]),
            gene_id=target,
            node_gene_ids=node_gene_ids,
        )


def load_brca_dataset(
    upstream_root: str | Path,
    *,
    use_comparison_filter: bool = True,
    use_supplied_neighbors: bool = True,
    target_gene_ids: Sequence[str] | None = None,
    paths: Mapping[str, str | Path] | None = None,
) -> DriverMONIDataset:
    """Load original BRCA processed inputs without modifying the source checkout.

    Preserves cached neighborhoods by default. Set use_supplied_neighbors=False
    explicitly to reproduce the upstream cache-generation fallback instead.
    """
    root = Path(upstream_root).expanduser().resolve()
    filenames = {
        "features": "sample_preprocessed_file/BRCA_input_data.csv",
        "labels": "using_Data/BRCA_label.csv",
        "comparison_genes": "for_concat_data/after_concat_BRCA_genes2.csv",
        "gene_annotations": "gene_info_for_GOSemSim.csv",
        "ppi_edges": "for_ref/humannetv3_ens_ppi.csv",
        "neighbors": "PPI_graph/gene_n_humannet_BRCA.pickle",
    }
    if paths is not None:
        if set(paths) - set(filenames):
            raise ValueError(f"Unknown input paths: {sorted(set(paths) - set(filenames))}")
        filenames.update(paths)
    filenames = {key: Path(value).expanduser() for key, value in filenames.items()}
    required = ["features", "labels", "ppi_edges"]
    if use_comparison_filter:
        required += ["comparison_genes", "gene_annotations"]
    if use_supplied_neighbors:
        required.append("neighbors")
    missing = []
    for role in required:
        path = root / filenames[role]
        if not path.is_file():
            if role == "ppi_edges":
                archive = path.parent / "humannet_v2_and_v3.zip"
                if archive.is_file():
                    continue
                missing.append(f"{role}: {path} OR {archive}")
            else:
                missing.append(f"{role}: {path}")
    if missing:
        raise FileNotFoundError(
            "Missing BRCA inputs:\n  " + "\n  ".join(missing)
            + "\nObtain the pinned authors' files as described in data/README.md, "
              "or pass --data-root /path/to/DriverMONI."
        )
    features = pd.read_csv(root / filenames["features"])
    labels = pd.read_csv(root / filenames["labels"])
    _require_columns(labels, ("gene_label",), "Label table")
    comparison = symbols = None
    if use_comparison_filter:
        comparison_frame = pd.read_csv(root / filenames["comparison_genes"])
        annotations = pd.read_csv(root / filenames["gene_annotations"])
        _require_columns(comparison_frame, ("gene",), "Comparison list")
        _require_columns(annotations, ("Symbol",), "Gene annotations")
        comparison = comparison_frame["gene"].tolist()
        symbols = annotations["Symbol"].tolist()
    table = prepare_gene_table(features, labels["gene_label"].tolist(), comparison, symbols)
    ppi = PPIGraph(load_humannet_edges(root, filenames["ppi_edges"]), table["gene_ens"])
    if use_supplied_neighbors:
        neighbors = load_neighbor_cache(root / filenames["neighbors"])
    else:
        neighbors = ppi.discover_neighbors(table["gene_ens"])
    return DriverMONIDataset(table, ppi, neighbors, target_gene_ids)
