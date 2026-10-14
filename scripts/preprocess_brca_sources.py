"""Prepare cached BRCA KEGG/HGNC sources and audit the supplied driver list."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.brca_sources import SourceCache, run_sources


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/brca_sources.json")
    parser.add_argument("--labels", type=Path, required=True, help="Authors' BRCA_label.csv; comparison only")
    parser.add_argument("--base-features", type=Path, help="Retain non-pathway columns and export the complete existing feature schema")
    parser.add_argument("--hgnc-snapshot", help="Override dated HGNC archive, YYYY-MM-DD")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--offline", action="store_true", help="Require checksummed cached sources; make no network calls")
    modes.add_argument("--refresh", action="store_true", help="Explicitly replace cached source snapshots")
    parser.add_argument("--lock-manifest", type=Path, help="Verify every source URL and checksum against an earlier manifest")
    parser.add_argument("--cgc-csv", type=Path, help="Optional lawfully obtained Cancer Gene Census export; not downloaded automatically")
    parser.add_argument("--cgc-version", help="Version of optional CGC export, e.g. v95")
    args = parser.parse_args()
    if bool(args.cgc_csv) != bool(args.cgc_version):
        parser.error("--cgc-csv and --cgc-version must be supplied together")
    config = json.loads(args.config.expanduser().read_text())
    expected = {"pathway_name", "hgnc_snapshot", "cache_dir", "output_dir"}
    if set(config) != expected:
        parser.error(f"Source config must contain exactly {sorted(expected)}")
    lock = None if args.lock_manifest is None else json.loads(args.lock_manifest.expanduser().read_text())
    cache = SourceCache(args.cache_dir.expanduser() if args.cache_dir else project_path(config["cache_dir"]),
                        offline=args.offline, refresh=args.refresh, lock=lock)
    output = args.output_dir.expanduser() if args.output_dir else project_path(config["output_dir"])
    summary = run_sources(
        cache=cache, output_dir=output, pathway_name=config["pathway_name"],
        hgnc_snapshot=args.hgnc_snapshot or config["hgnc_snapshot"],
        labels_path=args.labels.expanduser(),
        base_features_path=None if args.base_features is None else args.base_features.expanduser(),
        cgc_path=None if args.cgc_csv is None else args.cgc_csv.expanduser(), cgc_version=args.cgc_version,
    )
    print(json.dumps(summary, indent=2))
    print(f"Artifacts: {output.resolve()}")


if __name__ == "__main__":
    main()
