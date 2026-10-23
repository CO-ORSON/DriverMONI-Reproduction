"""Run the complete BRCA experiment: python scripts/run_brca.py --config configs/brca.yaml."""

import argparse
import os
from pathlib import Path
import sys

# Support the requested direct-script command as well as python -m scripts.run_brca.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("DGLBACKEND", "pytorch")

from src.dataset import load_brca_dataset
from src.train import run_cross_validation
from src.utils import input_manifest, load_config, project_path, validate_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/brca.yaml")
    parser.add_argument("--data-root", type=Path, help="Override the upstream data folder")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--output", type=Path, help="Override the metrics JSON path")
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ["seed", "device"]:
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.data_root is not None:
        config["data"]["upstream_root"] = str(args.data_root.expanduser().resolve())
    if args.output is not None:
        config["output_path"] = str(args.output.expanduser().resolve())
    validate_config(config)
    data = config["data"]
    print("Loading processed BRCA data...", flush=True)
    dataset = load_brca_dataset(
        project_path(data["upstream_root"]),
        use_comparison_filter=data["use_comparison_filter"],
        use_supplied_neighbors=data["use_supplied_neighbors"],
        paths={key: data[key] for key in ["features", "labels", "comparison_genes", "gene_annotations", "ppi_edges", "neighbors"]},
    )
    selection = ("inner-validation AUC; each outer test fold is evaluated once after epoch selection"
                 if config["cross_validation"]["epoch_selection"] == "best_validation_roc_auc"
                 else "test-fold AUC (upstream behavior)")
    print(f"Loaded {len(dataset)} targets. Epoch selection: {selection}.", flush=True)
    run_cross_validation(
        dataset, config, project_path(config["output_path"]), input_files=input_manifest(data),
        log=lambda message: print(message, flush=True),
    )


if __name__ == "__main__":
    main()
