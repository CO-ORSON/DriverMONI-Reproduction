"""Configuration, reproducibility, device selection, and JSON/provenance helpers."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re
from numbers import Real
from typing import Any

import dgl
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    """Resolve relative configured paths against the project, not shell cwd."""
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    """Read YAML and reject unsupported algorithm settings instead of ignoring them."""
    with Path(path).expanduser().open() as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate the BRCA-only source-compatible pipeline's supported configuration."""
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    required = {"cancer", "seed", "device", "reproducibility", "data", "model", "graph",
                "training", "cross_validation", "evaluation", "output_path", "upstream"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    unknown = set(config) - required
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    section_keys = {
        "upstream": {"repository", "commit"},
        "reproducibility": {"deterministic_algorithms", "cpu_threads"},
        "data": {"upstream_root", "features", "labels", "comparison_genes", "gene_annotations",
                 "ppi_edges", "neighbors", "use_comparison_filter", "use_supplied_neighbors"},
        "training": {"optimizer", "learning_rate", "weight_decay", "betas", "eps", "amsgrad",
                     "scheduler", "early_stopping", "epochs", "batch_size", "loss", "class_weights",
                     "shuffle_train", "shuffle_test", "num_workers", "drop_last",
                     "preserve_eval_mode_after_first_evaluation"},
        "cross_validation": {"n_splits", "stratified", "shuffle", "epoch_selection"},
        "evaluation": {"metrics", "positive_class", "probability_threshold", "zero_division", "std_ddof"},
    }
    for section, keys in section_keys.items():
        values = config[section]
        if not isinstance(values, dict):
            raise ValueError(f"{section} must be a YAML mapping")
        if set(values) != keys:
            raise ValueError(f"Invalid {section} keys: missing={sorted(keys - set(values))}, "
                             f"unknown={sorted(set(values) - keys)}")
    for key in ("output_path", *[f"data.{key}" for key in section_keys["data"]
                                if not key.startswith("use_")]):
        value = config["data"][key[5:]] if key.startswith("data.") else config[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty path string")
    if (config["upstream"]["repository"] != "https://github.com/BML-cbnu/DriverMONI"
            or config["upstream"]["commit"] != "ddb0a830158ba41c90019bd5ca92054c2ba0869b"):
        raise ValueError("upstream metadata must identify the implemented pinned reference revision")
    if not isinstance(config["device"], str) or not re.fullmatch(r"auto|cpu|cuda(?::[0-9]+)?", config["device"]):
        raise ValueError("device must be auto, cpu, cuda, or cuda:<nonnegative index>")
    if config["cancer"] != "BRCA":
        raise ValueError("Only BRCA is implemented")
    _validate_seed(config["seed"])
    expected_model = {
        "dnn": {"input_dim": 12, "hidden_dims": [256, 128], "output_dim": 64, "dropout": [0.7, 0.6]},
        "gat": {"input_dim": 13, "attention_dim": 64, "output_dim": 32, "attention_activation": "sigmoid"},
        "fusion": {"extra_gat_projection_dim": 32, "concatenated_dim": 96, "num_classes": 2},
    }
    if config["model"] != expected_model:
        raise ValueError("model must match the implemented upstream BRCA architecture")
    if config["graph"] != {"hops": 1, "neighbor_discovery": "undirected", "message_edges": "csv_direction",
                           "self_loops": True, "edge_threshold": None}:
        raise ValueError("Unsupported graph settings; the source-compatible graph is fixed")
    training = config["training"]
    fixed = {"optimizer": "adam", "loss": "weighted_nll", "class_weights": "inverse_training_fold_frequency",
             "scheduler": None, "early_stopping": None, "num_workers": 0, "drop_last": False}
    for key, expected in fixed.items():
        if key not in training or training[key] != expected:
            raise ValueError(f"training.{key} must be {expected!r} in this implementation")
    for key, minimum in [("epochs", 1), ("batch_size", 2)]:
        if type(training.get(key)) is not int or training[key] < minimum:
            raise ValueError(f"training.{key} must be an integer >= {minimum}")
    for key in ["learning_rate", "eps"]:
        if not _finite_number(training[key]) or training[key] <= 0:
            raise ValueError(f"training.{key} must be positive and finite")
    if not _finite_number(training["weight_decay"]) or training["weight_decay"] < 0:
        raise ValueError("training.weight_decay must be nonnegative and finite")
    betas = training.get("betas", [])
    if not isinstance(betas, list) or len(betas) != 2 or any(not _finite_number(beta) or not 0 <= beta < 1 for beta in betas):
        raise ValueError("training.betas must contain two numbers in [0, 1)")
    for section, keys in [(training, ["amsgrad", "shuffle_train", "shuffle_test", "preserve_eval_mode_after_first_evaluation"]),
                          (config["data"], ["use_comparison_filter", "use_supplied_neighbors"]),
                          (config["reproducibility"], ["deterministic_algorithms"])]:
        for key in keys:
            if type(section.get(key)) is not bool:
                raise ValueError(f"{key} must be a YAML boolean")
    threads = config["reproducibility"].get("cpu_threads")
    if threads is not None and (type(threads) is not int or threads < 1):
        raise ValueError("cpu_threads must be null or a positive integer")
    cv = config["cross_validation"]
    if cv.get("n_splits") != 3 or cv.get("stratified") is not True or cv.get("shuffle") is not True:
        raise ValueError("Use three-fold shuffled stratified CV")
    if cv.get("epoch_selection") != "best_test_roc_auc":
        raise ValueError("Only the documented upstream best_test_roc_auc selection is implemented")
    evaluation = config["evaluation"]
    if (type(evaluation["positive_class"]) is not int or evaluation["positive_class"] != 1
            or not _finite_number(evaluation["probability_threshold"])
            or not 0 <= evaluation["probability_threshold"] <= 1):
        raise ValueError("Use positive class 1 and a probability threshold in [0,1]")
    if type(evaluation["zero_division"]) is not int or evaluation["zero_division"] not in [0, 1]:
        raise ValueError("evaluation.zero_division must be 0 or 1")
    if type(evaluation["std_ddof"]) is not int or evaluation["std_ddof"] not in [0, 1]:
        raise ValueError("evaluation.std_ddof must be 0 or 1")
    expected_metrics = {"auc", "accuracy", "precision", "recall", "f1", "balanced_accuracy"}
    metrics = evaluation["metrics"]
    if not isinstance(metrics, list) or not all(isinstance(metric, str) for metric in metrics) or len(metrics) != len(expected_metrics) or set(metrics) != expected_metrics:
        raise ValueError(f"evaluation.metrics must contain {sorted(expected_metrics)}")


def _finite_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and bool(np.isfinite(value))


def _validate_seed(seed: int) -> None:
    # DGL 2.2 accepts a signed C int, narrower than NumPy/PyTorch's seed ranges.
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in [0, 2**31) for DGL compatibility")


def seed_everything(seed: int, deterministic_algorithms: bool, cpu_threads: int | None) -> None:
    """Seed once before CV; do not reset RNG state between epochs or folds."""
    _validate_seed(seed)
    if deterministic_algorithms:
        workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if workspace not in {":4096:8", ":16:8"}:
            raise ValueError("Deterministic execution requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)
    torch.backends.cudnn.deterministic = deterministic_algorithms
    torch.backends.cudnn.benchmark = False
    if cpu_threads is not None:
        torch.set_num_threads(cpu_threads)


def select_device(value: str) -> torch.device:
    """Resolve auto to CUDA when available, otherwise CPU; reject invalid requests."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are auto, cpu, cuda, and cuda:<index>")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {device.index} is unavailable")
        # Fail before training if PyTorch has CUDA but the DGL wheel does not.
        try:
            dgl.graph(([0], [0])).to(device)
        except Exception as error:
            raise RuntimeError("CUDA requires a compatible CUDA-enabled DGL installation; "
                               "use --device cpu or install matching CUDA PyTorch/DGL wheels") from error
    return device


def save_json(path: str | Path, report: dict[str, Any]) -> None:
    """Atomically save strict JSON; partial experiments carry an explicit status."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def runtime_metadata() -> dict[str, Any]:
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["torch", "dgl", "numpy", "pandas", "scikit-learn"]},
        "cpu_threads": torch.get_num_threads(),
        "cuda_version": torch.version.cuda,
    }


def input_manifest(data_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Hash consumed local input files (the whole archive if reading its member)."""
    root = project_path(data_config["upstream_root"])
    keys = ["features", "labels", "ppi_edges"]
    if data_config["use_comparison_filter"]:
        keys += ["comparison_genes", "gene_annotations"]
    if data_config["use_supplied_neighbors"]:
        keys += ["neighbors"]
    manifest = []
    for key in keys:
        path = root / Path(data_config[key]).expanduser()
        if key == "ppi_edges" and not path.is_file():
            path = path.parent / "humannet_v2_and_v3.zip"
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest.append({"role": key, "path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    return manifest
