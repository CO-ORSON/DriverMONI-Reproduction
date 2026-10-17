"""Acquire BRCA sources and export one complete, auditable feature CSV."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.acquisition import FileCache
from src.preprocessing.brca import run_brca


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/brca_preprocessing.json")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--offline", action="store_true")
    modes.add_argument("--refresh", action="store_true")
    parser.add_argument("--lock-manifest", type=Path, help="Require the prior source URLs and SHA-256 values")
    args = parser.parse_args()
    config = json.loads(args.config.expanduser().read_text())
    expected = {"omics_config", "pathway_config", "cache_dir", "output_dir", "network", "labels"}
    if set(config) != expected:
        parser.error(f"Pipeline config must contain exactly {sorted(expected)}")
    omics = json.loads(project_path(config["omics_config"]).read_text())
    pathways = json.loads(project_path(config["pathway_config"]).read_text())
    lock = None if args.lock_manifest is None else json.loads(args.lock_manifest.expanduser().read_text())
    cache = FileCache(args.cache_dir.expanduser() if args.cache_dir else project_path(config["cache_dir"]),
                      offline=args.offline, refresh=args.refresh, lock=lock)
    output = args.output_dir.expanduser() if args.output_dir else project_path(config["output_dir"])
    summary = run_brca(config, omics, pathways, cache=cache, output_dir=output)
    print(json.dumps(summary, indent=2))
    print(f"Feature file: {(output / 'BRCA_input_data.csv').resolve()}")


if __name__ == "__main__":
    main()
