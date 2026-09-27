"""Regression coverage for fresh-checkout CLI, config, backend and seed failures."""

import os
from pathlib import Path
import importlib
import runpy
import sys

import pytest

from src.dataset import load_brca_dataset
from src.utils import load_config, seed_everything, validate_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return load_config(ROOT / "configs/brca.yaml")


@pytest.mark.parametrize("script", ["debug_brca.py", "run_brca.py", "smoke_test.py"])
def test_direct_scripts_work_outside_project_without_backend_setting(script, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("DGLBACKEND", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [script, "--help"])
    monkeypatch.setattr(sys, "path", [value for value in sys.path if value not in ("", str(ROOT))])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(ROOT / "scripts" / script), run_name="__main__")
    assert exit_info.value.code == 0
    assert "--data-root" in capsys.readouterr().out
    assert str(ROOT) in sys.path


def test_library_import_sets_backend_without_home_configuration(tmp_path, monkeypatch):
    import src
    monkeypatch.delenv("DGLBACKEND", raising=False)
    monkeypatch.setenv("DGLDEFAULTDIR", str(tmp_path / "must-not-create"))
    importlib.reload(src)
    assert os.environ["DGLBACKEND"] == "pytorch"
    assert not (tmp_path / "must-not-create").exists()


def test_incompatible_explicit_backend_explained(monkeypatch):
    import src
    monkeypatch.setenv("DGLBACKEND", "tensorflow")
    with pytest.raises(RuntimeError, match="DGLBACKEND=pytorch"):
        importlib.reload(src)


@pytest.mark.parametrize("seed", [-1, 2**31, 2**32 - 1, True])
def test_invalid_dgl_seeds_fail_before_runtime_calls(config, seed):
    config["seed"] = seed
    with pytest.raises(ValueError, match="seed"):
        validate_config(config)
    with pytest.raises(ValueError, match="seed"):
        seed_everything(seed, True, 1)


def test_largest_supported_dgl_seed(config):
    config["seed"] = 2**31 - 1
    validate_config(config)
    seed_everything(config["seed"], True, 1)


@pytest.mark.parametrize("section,key,value", [
    ("training", "learnng_rate", 0.02),
    ("training", "learning_rate", "0.001"),
    ("training", "betas", None),
    ("data", "features", ""),
    ("reproducibility", "cpu_threds", 2),
    ("evaluation", "probability_threshold", "0.5"),
    ("evaluation", "metrics", None),
])
def test_invalid_config_values_give_actionable_errors(config, section, key, value):
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_missing_nested_key_and_unsupported_device(config):
    del config["data"]["labels"]
    with pytest.raises(ValueError, match="labels"):
        validate_config(config)
    config = load_config(ROOT / "configs/brca.yaml")
    config["device"] = "mps"
    with pytest.raises(ValueError, match="device"):
        validate_config(config)


def test_missing_inputs_list_every_role_and_acquisition_instructions(tmp_path):
    with pytest.raises(FileNotFoundError) as error:
        load_brca_dataset(tmp_path)
    message = str(error.value)
    for role in ("features", "labels", "comparison_genes", "gene_annotations", "ppi_edges", "neighbors"):
        assert role in message
    assert "data/README.md" in message and "--data-root" in message
    with pytest.raises(FileNotFoundError) as error:
        load_brca_dataset(tmp_path, use_comparison_filter=False, use_supplied_neighbors=False)
    assert "neighbors:" not in str(error.value)
    assert "comparison_genes:" not in str(error.value)
