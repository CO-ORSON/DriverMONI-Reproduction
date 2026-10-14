"""Prepare linked oncology evidence and BRCA gene-status tables, without relabelling."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.acquisition import FileCache
from src.preprocessing.brca_driver_evidence import run_driver_evidence


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs/brca_driver_evidence.json')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--lock-manifest', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.expanduser().read_text())
    lock = json.loads(args.lock_manifest.expanduser().read_text()) if args.lock_manifest else None
    cache = FileCache(args.cache_dir.expanduser() if args.cache_dir else project_path(config['cache_dir']),
                      offline=args.offline, lock=lock)
    result = run_driver_evidence(config, cache=cache,
        output_dir=args.output_dir.expanduser() if args.output_dir else project_path(config['output_dir']),
        canonical_path=project_path(config['canonical_features']), reviewed_path=project_path(config['reviewed_assertions']), lock=lock)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
