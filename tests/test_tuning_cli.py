"""Real subprocess/parallel-worker integration of the standalone D entry point."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from scripts.run_brca_tuning import load_tuning_spec
from src.utils import load_config
from tests.test_validation_training import make_dataset

ROOT = Path(__file__).resolve().parents[1]


def test_published_search_has_six_candidates_and_unchanged_c_base():
    spec, base = load_tuning_spec(ROOT / "configs/brca_tuning.yaml")
    original = load_config(ROOT / "configs/brca_validation_train_mode.yaml")
    expected = deepcopy(original)
    expected.update(seed=42, device="cpu")
    assert base == expected
    assert spec["search"] == {"learning_rates": [.001, .0003, .0001],
                              "dropout_pairs": [[.7, .6], [.5, .3]], "selection_metric": "auc"}
    assert spec["parallel_folds"] == 3


@pytest.mark.parametrize("mutation", ["unknown", "missing", "workers_bool", "workers_zero", "bad_seed", "source_mode"])
def test_search_spec_rejects_ignored_or_invalid_settings(tmp_path, mutation):
    spec, base = load_tuning_spec(ROOT / "configs/brca_tuning.yaml")
    if mutation == "unknown":
        spec["unknown"] = True
    elif mutation == "missing":
        del spec["search"]
    elif mutation == "workers_bool":
        spec["parallel_folds"] = True
    elif mutation == "workers_zero":
        spec["parallel_folds"] = 0
    elif mutation == "bad_seed":
        spec["seed"] = True
    else:
        base["training"]["preserve_eval_mode_after_first_evaluation"] = True
        path = tmp_path / "base.yaml"
        path.write_text(yaml.safe_dump(base))
        spec["base_config"] = str(path)
    path = tmp_path / "search.yaml"
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError):
        load_tuning_spec(path)


def test_parallel_cli_smoke_and_sequential_resume(tmp_path):
    """Spawn real workers from a foreign cwd; resume produces identical fold data."""
    data = tmp_path / "data"
    data.mkdir()
    dataset = make_dataset()
    table = dataset.gene_table
    table.drop(columns="label").to_csv(data / "features.csv", index=False)
    pd.DataFrame({"gene_label": table.loc[table["label"] == 1, "gene_ens"]}).to_csv(data / "labels.csv", index=False)
    dataset.ppi.edge_rows.to_csv(data / "ppi.csv", index=False)
    spec, base = load_tuning_spec(ROOT / "configs/brca_tuning.yaml")
    base["training"].update(epochs=2, batch_size=4)
    base["data"].update(upstream_root=str(data), features="features.csv", labels="labels.csv", ppi_edges="ppi.csv")
    base_file = tmp_path / "base.yaml"
    base_file.write_text(yaml.safe_dump(base))
    spec.update(base_config=str(base_file), parallel_folds=3, output_dir=str(tmp_path / "result"))
    spec["search"]["learning_rates"] = [.001]
    config_file = tmp_path / "search.yaml"
    config_file.write_text(yaml.safe_dump(spec))
    command = [sys.executable, str(ROOT / "scripts/run_brca_tuning.py"), "--config", str(config_file)]
    run = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=90)
    assert run.returncode == 0, run.stdout + run.stderr
    output = tmp_path / "result"
    first = json.loads((output / "summary.json").read_text())
    assert first["status"] == "complete"
    assert len(first["folds"]) == 3
    assert all(len(fold["candidate_results"]) == 2 for fold in first["folds"])
    before = {path.parent.name: path.read_bytes() for path in output.glob("fold_*/report.json")}
    resumed = subprocess.run(command + ["--resume", "--parallel-folds", "1"], cwd=tmp_path,
                             capture_output=True, text=True, timeout=90)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert json.loads((output / "summary.json").read_text()) == first
    # No completed fold should be evaluated or rewritten during an unchanged resume.
    for path in output.glob("fold_*/report.json"):
        assert path.read_bytes() == before[path.parent.name]
    rejected = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert rejected.returncode != 0 and "not empty" in rejected.stderr
