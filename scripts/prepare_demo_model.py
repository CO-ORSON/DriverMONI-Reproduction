"""Prepare a local, validation-selected fold-one checkpoint for the Streamlit app."""

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("DGLBACKEND", "pytorch")

from src.inference import load_inference_dataset, prepare_demo_checkpoint
from src.utils import input_manifest, load_config, project_path, validate_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/brca_validation.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "checkpoints/brca_demo.pt")
    parser.add_argument("--data-root", type=Path, help="Relocate the configured processed inputs")
    parser.add_argument("--device", default="cpu", help="cpu (default), auto, cuda or cuda:<index>")
    args = parser.parse_args()
    config = load_config(args.config)
    config["device"] = args.device
    if args.data_root is not None:
        config["data"]["upstream_root"] = str(args.data_root.expanduser().resolve())
    validate_config(config)
    if config["cross_validation"]["epoch_selection"] != "best_validation_roc_auc":
        parser.error("Use a validation-selection configuration: B or C, not the original/A test-selection protocol")
    print("Loading data and training one demo fold. Existing CV reports will not change.", flush=True)
    files = input_manifest(config["data"])
    dataset = load_inference_dataset(config)
    # Detect input replacement during the load before associating hashes with data.
    if files != input_manifest(config["data"]):
        raise RuntimeError("Input files changed while loading; retry with stable inputs")
    prepare_demo_checkpoint(dataset, config, project_path(args.output), input_files=files,
                            log=lambda message: print(message, flush=True))


if __name__ == "__main__":
    main()
