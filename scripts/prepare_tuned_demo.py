"""Import D's validation-selected fold-one checkpoint for local live predictions."""

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("DGLBACKEND", "pytorch")

from src.inference import prepare_tuned_demo_checkpoint
from src.utils import project_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "results/brca_tuning/fold_1/report.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "checkpoints/brca_demo_d.pt")
    parser.add_argument("--data-root", type=Path, help="Relocate identical processed input files")
    parser.add_argument("--device", default="cpu", help="cpu (default), auto, cuda or cuda:<index>")
    args = parser.parse_args()
    print("Importing the completed D fold-one winner. No training will run.", flush=True)
    prepare_tuned_demo_checkpoint(project_path(args.report), project_path(args.output),
                                 data_root=args.data_root, device=args.device,
                                 log=lambda message: print(message, flush=True))


if __name__ == "__main__":
    main()
