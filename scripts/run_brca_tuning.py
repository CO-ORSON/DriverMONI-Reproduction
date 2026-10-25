"""Run D's learning-rate/dropout search inside three outer BRCA folds."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import multiprocessing
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("DGLBACKEND", "pytorch")
# Workers use spawn and initialize a fresh runtime. Avoid macOS OpenMP's
# unnecessary fork-time initialization before exec (must precede torch import).
if sys.platform == "darwin":
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

import yaml

from src.dataset import load_brca_dataset
from src.tuning import aggregate_tuning_reports, run_tuning_fold, tuning_candidates
from src.utils import input_manifest, load_config, project_path, save_json, validate_config


def load_tuning_spec(path: str | Path) -> tuple[dict, dict]:
    """Read the separate search spec without relaxing reproduction validation."""
    with Path(path).expanduser().open() as handle:
        spec = yaml.safe_load(handle)
    required = {"base_config", "seed", "device", "search", "parallel_folds", "output_dir"}
    if not isinstance(spec, dict) or set(spec) != required:
        raise ValueError(f"Tuning configuration must contain exactly: {sorted(required)}")
    for name in ("base_config", "output_dir"):
        if not isinstance(spec[name], str) or not spec[name].strip():
            raise ValueError(f"{name} must be a nonempty path string")
    if type(spec["parallel_folds"]) is not int or not 1 <= spec["parallel_folds"] <= 3:
        raise ValueError("parallel_folds must be an integer from 1 to 3")
    base = load_config(project_path(spec["base_config"]))
    base["seed"], base["device"] = spec["seed"], spec["device"]
    validate_config(base)
    if (base["cross_validation"]["epoch_selection"] != "best_validation_roc_auc"
            or base["training"]["preserve_eval_mode_after_first_evaluation"]):
        raise ValueError("D requires C's separate validation selection and training mode every epoch")
    tuning_candidates(spec["search"])
    return spec, base


def _run_fold(number: int, base: dict, search: dict, output_dir: str,
              expected_inputs: list[dict], resume: bool) -> dict:
    """Independent process: load identical inputs and write only its own fold."""
    data = base["data"]
    before = input_manifest(data)
    if before != expected_inputs:
        raise RuntimeError("Input files changed before a fold started")
    dataset = load_brca_dataset(
        project_path(data["upstream_root"]),
        use_comparison_filter=data["use_comparison_filter"],
        use_supplied_neighbors=data["use_supplied_neighbors"],
        paths={key: data[key] for key in ("features", "labels", "comparison_genes",
                                        "gene_annotations", "ppi_edges", "neighbors")},
    )
    if input_manifest(data) != before:
        raise RuntimeError("Input files changed during dataset loading")
    return run_tuning_fold(
        dataset, base, search, number, Path(output_dir) / f"fold_{number}",
        input_files=before, resume=resume,
        log=lambda message: print(f"[fold {number}] {message}", flush=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/brca_tuning.yaml")
    parser.add_argument("--parallel-folds", type=int, choices=[1, 2, 3])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", help="cpu, auto, cuda, or cuda:<index>")
    parser.add_argument("--data-root", type=Path, help="Combined data root containing brca/ and upstream/")
    parser.add_argument("--output-dir", type=Path, help="New output folder; CLI paths are relative to the working directory")
    parser.add_argument("--resume", action="store_true", help="Reuse only matching, verified completed candidates")
    args = parser.parse_args()
    spec, base = load_tuning_spec(args.config)
    for name in ("seed", "device"):
        if getattr(args, name) is not None:
            base[name] = spec[name] = getattr(args, name)
    if args.data_root is not None:
        base["data"]["upstream_root"] = str(args.data_root.expanduser().resolve())
    if args.parallel_folds is not None:
        spec["parallel_folds"] = args.parallel_folds
    validate_config(base)
    # Avoid three memory-heavy training workers competing for one CUDA device.
    if base["device"] != "cpu" and spec["parallel_folds"] != 1:
        parser.error("Use --parallel-folds 1 for auto/CUDA; parallel folds are supported on CPU")
    output = args.output_dir.expanduser().resolve() if args.output_dir else project_path(spec["output_dir"])
    if not args.resume and output.exists() and any(output.iterdir()):
        parser.error("Output folder is not empty. Use --resume for the same run or select a new --output-dir")
    inputs = input_manifest(base["data"])
    candidates = tuning_candidates(spec["search"])
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    run = {"experiment": "D", "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
           "spec": deepcopy(spec), "base_config": deepcopy(base), "input_files": inputs,
           "candidate_count": len(candidates), "completed_folds": [], "resumed": args.resume}
    # Preserve a previous final/partial manifest until resume validation succeeds.
    pending_path = output / "run.pending.json"
    save_json(pending_path, run)
    print(f"D: {len(candidates)} candidates × 3 outer folds × {base['training']['epochs']} epochs; "
          f"{spec['parallel_folds']} independent CPU fold worker(s).", flush=True)
    print("Select by inner-validation AUC only; test the winner and the prespecified matched C control afterward.", flush=True)
    reports = []
    try:
        if spec["parallel_folds"] == 1:
            for number in (1, 2, 3):
                reports.append(_run_fold(number, base, spec["search"], str(output), inputs, args.resume))
                run["completed_folds"].append(number)
                save_json(pending_path, run)
        else:
            with ProcessPoolExecutor(max_workers=spec["parallel_folds"],
                                     mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = {pool.submit(_run_fold, number, base, spec["search"], str(output), inputs, args.resume): number
                           for number in (1, 2, 3)}
                for future in as_completed(futures):
                    reports.append(future.result())
                    run["completed_folds"].append(futures[future])
                    run["completed_folds"].sort()
                    save_json(pending_path, run)
        reports.sort(key=lambda report: report["fold"])
        summary = aggregate_tuning_reports(reports, ddof=base["evaluation"]["std_ddof"])
        save_json(output / "summary.json", summary)
        run.update(status="complete", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                   elapsed_seconds=time.perf_counter() - started)
        save_json(pending_path, run)
        pending_path.replace(output / "run.json")
        print(f"Completed all folds. Summary: {output / 'summary.json'}", flush=True)
    except BaseException as error:
        run.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                   error=f"{type(error).__name__}: {error}", elapsed_seconds=time.perf_counter() - started)
        save_json(pending_path, run)
        # A rejected resume must not destroy the previous run's original manifest.
        if not (output / "run.json").exists():
            pending_path.replace(output / "run.json")
        raise


if __name__ == "__main__":
    main()
