"""Print BRCA graph inputs: python -m scripts.debug_brca --data-root PATH."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_brca_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "data/upstream")
    parser.add_argument("--gene", action="append", help="Target Ensembl ID; repeat to inspect multiple genes")
    parser.add_argument("--rebuild-neighbors", action="store_true", help="Explicitly use the upstream cache-generation fallback")
    parser.add_argument("--no-comparison-filter", action="store_true")
    args = parser.parse_args()
    dataset = load_brca_dataset(
        args.data_root, use_comparison_filter=not args.no_comparison_filter,
        use_supplied_neighbors=not args.rebuild_neighbors,
    )
    targets = args.gene
    if targets is None:
        targets = [dataset.target_gene_ids[0]]
        positives = dataset.gene_table.loc[dataset.gene_table["label"] == 1, "gene_ens"]
        if len(positives):
            targets.append(positives.iloc[0])
        empty = next((gene for gene in dataset.target_gene_ids if not dataset.neighbors[gene]), None)
        if empty is not None:
            targets.append(empty)
        targets = list(dict.fromkeys(targets))
    lookup = {gene: index for index, gene in enumerate(dataset.target_gene_ids)}
    missing = set(targets) - set(lookup)
    if missing:
        parser.error(f"Targets absent from this dataset: {sorted(missing)}")
    print(f"BRCA targets: {len(dataset)}; positives: {int(dataset.gene_table['label'].sum())}")
    print(f"Neighbor source: {'rebuilt' if args.rebuild_neighbors else 'supplied cache'}")
    for target in targets:
        sample = dataset[lookup[target]]
        symbol = dataset.context.loc[target, "gene_symbol"]
        print(f"\ntarget gene: {sample.gene_id} ({symbol})")
        print(f"number of nodes: {sample.graph.num_nodes()}")
        print(f"number of edges: {sample.graph.num_edges()}")
        print(f"DNN feature shape: {tuple(sample.gene_features.shape)}")
        print(f"node feature shape: {tuple(sample.graph.ndata['feat'].shape)}")
        print(f"label: {sample.label}")


if __name__ == "__main__":
    main()
