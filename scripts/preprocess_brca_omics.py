"""Acquire BRCA raw input snapshots and derive mutation/expression features."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.acquisition import FileCache
from src.preprocessing.brca_omics import run_omics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/brca_omics_sources.json")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--offline", action="store_true")
    modes.add_argument("--refresh", action="store_true")
    parser.add_argument("--lock-manifest", type=Path)
    parser.add_argument("--mutation-maf", type=Path, help="Explicit local MAF or MAF.gz")
    parser.add_argument("--expression-matrix", type=Path, help="Explicit local gene-by-sample TSV or TSV.gz; declare its unit/transform in the config")
    parser.add_argument("--annotation-gtf", type=Path, help="Explicit local annotation GTF or GTF.gz matching the declared release")
    parser.add_argument("--pathway-membership", type=Path, help="Optional mapped KEGG membership from the pathway entry point")
    parser.add_argument("--network-edges", type=Path, help="Optional mapped HumanNet CSV; pair with pathway-membership to export all 13 features")
    args = parser.parse_args()
    if bool(args.pathway_membership) != bool(args.network_edges):
        parser.error("--pathway-membership and --network-edges must be provided together")
    config = json.loads(args.config.expanduser().read_text())
    for role, path in (("mutation", args.mutation_maf), ("expression", args.expression_matrix), ("annotation", args.annotation_gtf)):
        if path:
            spec = config["sources"][role]
            spec.pop("url", None)
            spec.pop("member", None)
            spec["local_path"] = str(path.expanduser().resolve())
            spec["format"] = "gzip" if path.suffix == ".gz" else "text"
            config["profile"] = "explicit_local_inputs_with_declared_source_metadata"
    def project_path(value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path
    cache_dir = args.cache_dir.expanduser() if args.cache_dir else project_path(config["cache_dir"])
    output = args.output_dir.expanduser() if args.output_dir else project_path(config["output_dir"])
    lock = None if args.lock_manifest is None else json.loads(args.lock_manifest.expanduser().read_text())
    cache = FileCache(cache_dir, offline=args.offline, refresh=args.refresh, lock=lock)
    summary = run_omics(config, cache=cache, output_dir=output,
                        pathway_membership=None if args.pathway_membership is None else args.pathway_membership.expanduser(),
                        network_edges=None if args.network_edges is None else args.network_edges.expanduser())
    print(json.dumps(summary, indent=2))
    print(f"Artifacts: {output.resolve()}")


if __name__ == "__main__":
    main()
