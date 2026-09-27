"""Run two epochs in three folds on 12 real BRCA targets, retaining full graph context."""

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("DGLBACKEND", "pytorch")

from src.dataset import DriverMONIDataset, load_brca_dataset
from src.train import run_cross_validation
from src.utils import input_manifest, load_config, project_path, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/upstream")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/brca_smoke_metrics.json")
    args = parser.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    config = load_config(PROJECT_ROOT / "configs/brca.yaml")
    config["device"] = "cpu"
    config["data"]["upstream_root"] = str(args.data_root.resolve())
    config["training"].update(epochs=2, batch_size=4)
    config["output_path"] = str(args.output.resolve())
    dataset = load_brca_dataset(args.data_root)
    targets = []
    for label in (0, 1):
        candidates = dataset.gene_table.loc[dataset.gene_table.label == label, "gene_ens"].tolist()
        if len(candidates) < 6:
            raise ValueError("Smoke test needs six targets per class")
        targets.extend(candidates[:6])
    subset = DriverMONIDataset(dataset.gene_table, dataset.ppi, dataset.neighbors, targets)
    report = run_cross_validation(subset, config, project_path(config["output_path"]),
                                  input_files=input_manifest(config["data"]),
                                  log=lambda message: print(message, flush=True))
    assert report["status"] == "complete" and len(report["folds"]) == 3
    held_out = [gene for fold in report["folds"] for gene in fold["test_gene_ids"]]
    assert len(held_out) == len(set(held_out)) == 12
    assert set(held_out) == set(targets)
    assert all([entry["training_mode_during_updates"] for entry in fold["epoch_history"]]
               == [True, False] for fold in report["folds"])
    report["smoke_test"] = {"targets": 12, "epochs_per_fold": 2,
                             "purpose": "Pipeline validation only; not a full BRCA performance estimate"}
    save_json(args.output, report)
    print("Smoke test passed: 12 real targets, three folds, two epochs per fold.")


if __name__ == "__main__":
    main()
