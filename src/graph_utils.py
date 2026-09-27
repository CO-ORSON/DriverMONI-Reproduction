"""HumanNet input graphs matching the active upstream HumanNet dataset.

Neighborhoods are undirected; message edges are directed CSV rows. Add one
self-loop per node before CSV edges, retaining duplicate rows and CSV self-loops.
Do not threshold LLS, add reverse edges, simplify edges, or filter by mutations.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import pickle
from typing import BinaryIO
from zipfile import ZipFile

import dgl
import pandas as pd
import torch


def load_humannet_edges(
    upstream_root: str | Path, relative_csv_path: str | Path = "for_ref/humannetv3_ens_ppi.csv",
) -> pd.DataFrame:
    """Read the unfiltered v3 CSV, also accepting the original bundled ZIP."""
    root = Path(upstream_root)
    csv_path = root / relative_csv_path
    if csv_path.is_file():
        return pd.read_csv(csv_path)
    archive = csv_path.parent / "humannet_v2_and_v3.zip"
    if not archive.is_file():
        raise FileNotFoundError(f"Missing HumanNet CSV {csv_path} or archive {archive}")
    with ZipFile(archive) as zipped:
        with zipped.open("humannet_v2_and_v3/humannetv3_ens_ppi.csv") as handle:
            return pd.read_csv(handle)


class _NeighborUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> object:
        raise pickle.UnpicklingError("Neighbor caches must not contain Python objects")


def load_neighbor_cache(path: str | Path) -> dict[str, list[str]]:
    """Read the authors' plain gene -> neighbor-list cache without class loading."""
    with Path(path).open("rb") as handle:
        cache = _read_neighbor_cache(handle)
    return cache


def _read_neighbor_cache(handle: BinaryIO) -> dict[str, list[str]]:
    cache = _NeighborUnpickler(handle).load()
    if not isinstance(cache, dict) or any(
        not isinstance(gene, str)
        or not isinstance(neighbors, list)
        or any(not isinstance(neighbor, str) for neighbor in neighbors)
        for gene, neighbors in cache.items()
    ):
        raise ValueError("Neighbor cache must be a dictionary of string IDs to string lists")
    return cache


class PPIGraph:
    """CPU global graph with stable gene-ID mappings and original edge multiplicity.

    Deterministic sorted IDs replace upstream set iteration. Graph equivalence
    is by gene identity and directed edge multiplicity, not arbitrary node IDs.
    """

    def __init__(self, edges: pd.DataFrame, feature_gene_ids: Iterable[str]) -> None:
        required = ["entrez_id_1", "entrez_id_2"]
        missing = set(required) - set(edges.columns)
        if missing:
            raise ValueError(f"HumanNet edge table is missing columns: {sorted(missing)}")
        # The misleading 'entrez' column names contain Ensembl IDs in this data.
        sources = edges["entrez_id_1"].tolist()
        destinations = edges["entrez_id_2"].tolist()
        features = list(feature_gene_ids)
        if any(not isinstance(gene, str) or not gene for gene in features + sources + destinations):
            raise ValueError("PPI endpoints and feature gene IDs must be nonempty strings")
        self.gene_ids = tuple(sorted(set(features) | set(sources) | set(destinations)))
        self.gene_to_node = {gene: index for index, gene in enumerate(self.gene_ids)}
        self.edge_rows = edges.loc[:, required].copy().reset_index(drop=True)
        count = len(self.gene_ids)
        # Exactly one synthetic loop per node, followed by every original CSV row.
        source_indices = list(range(count)) + [self.gene_to_node[gene] for gene in sources]
        destination_indices = list(range(count)) + [self.gene_to_node[gene] for gene in destinations]
        self.graph = dgl.graph((source_indices, destination_indices), num_nodes=count)
        self.graph.ndata["global_node_id"] = torch.arange(count)

    def discover_neighbors(self, target_gene_ids: Iterable[str]) -> dict[str, list[str]]:
        """Equivalent radius-one node sets to nx.ego_graph on undirected PPI.

        Includes the center when it occurs in the CSV graph; otherwise returns
        an empty list, like the upstream fallback. Ego extraction always adds it.
        """
        adjacency: dict[str, set[str]] = defaultdict(set)
        for source, destination in self.edge_rows.itertuples(index=False, name=None):
            adjacency[source].add(destination)
            adjacency[destination].add(source)
        return {
            gene: sorted(adjacency[gene] | {gene}) if gene in adjacency else []
            for gene in target_gene_ids
        }

    def ego_graph(
        self,
        target_gene: str,
        neighbors: Mapping[str, Sequence[str]],
        available_genes: set[str],
    ) -> dgl.DGLGraph:
        """Extract an induced ego graph, marking exactly one ``is_target`` node.

        Supplied neighbor lists are authoritative, even when empty. Remove
        neighbors without features and deduplicate nodes, but never edges.
        """
        if target_gene not in available_genes or target_gene not in self.gene_to_node:
            raise ValueError(f"Target gene lacks features or a global node: {target_gene}")
        if target_gene not in neighbors:
            raise ValueError(f"Neighbor cache has no entry for target gene {target_gene}")
        retained = set(neighbors[target_gene]) & available_genes
        nodes = [target_gene] + sorted(retained - {target_gene})
        node_indices = torch.tensor([self.gene_to_node[gene] for gene in nodes])
        graph = dgl.node_subgraph(self.graph, node_indices, store_ids=False)
        # Use actual returned IDs, not an assumption about subgraph reordering.
        target_mask = graph.ndata["global_node_id"] == self.gene_to_node[target_gene]
        if int(target_mask.sum()) != 1:
            raise RuntimeError("Subgraph extraction did not preserve one target gene")
        graph.ndata["is_target"] = target_mask
        source, destination = graph.edges()
        graph.edata["is_target_edge"] = target_mask[source] | target_mask[destination]
        return graph

    def node_gene_ids(self, graph: dgl.DGLGraph) -> tuple[str, ...]:
        """Return gene IDs in the graph's actual local node order."""
        return tuple(self.gene_ids[index] for index in graph.ndata["global_node_id"].tolist())
