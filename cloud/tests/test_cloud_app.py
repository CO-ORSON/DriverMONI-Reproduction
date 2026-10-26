"""Standalone review-app tests, runnable without the scientific test conftest.

Run: python -m pytest cloud/tests -q
Only cloud/requirements.txt and pytest are needed; Torch/DGL are not imported.
"""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

PROJECT = Path(__file__).resolve().parents[2]
REVIEW_PAGES = ["Experiments", "Gene explorer", "Model architecture", "About this app"]


@pytest.fixture
def deployment_root(tmp_path: Path) -> Path:
    """Copy only files a review deployment receives, excluding local artifacts."""
    paths = [
        "cloud/app.py", "streamlit_app.py", ".streamlit/config.toml",
        "src/__init__.py", "src/app_data.py", "src/app_results.py", "src/feature_schema.py",
        "data/brca/BRCA_input_data.csv",
    ] + [f"results/published/{name}.json" for name in ("original", "a", "b", "c", "d")]
    for relative in paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / relative, destination)
    return tmp_path


def _open(root: Path) -> AppTest:
    app = AppTest.from_file(root / "cloud/app.py", default_timeout=30).run()
    assert not app.exception
    assert not app.error
    return app


def _text(app) -> str:
    return "\n".join(str(element.value) for name in ("markdown", "caption", "info", "warning", "title")
                     for element in getattr(app, name))


def test_clean_deployment_opens_d_and_keeps_experiments_ordered(deployment_root):
    assert not (deployment_root / "checkpoints").exists()
    assert not (deployment_root / "data/upstream").exists()
    assert list((deployment_root / "results").iterdir()) == [deployment_root / "results/published"]
    app = _open(deployment_root)
    assert app.radio(key="page").options == REVIEW_PAGES
    assert app.radio(key="page").value == "Experiments"
    assert app.selectbox(key="experiment_choice").value == "D"
    assert [label.split(" · ")[0] for label in app.selectbox(key="experiment_choice").options] == [
        "Original", "A", "B", "C", "D",
    ]
    assert app.metric[0].value == "0.921"
    assert "Live prediction" not in app.radio(key="page").options
    assert "Setup" not in app.radio(key="page").options


def test_gene_explorer_uses_shared_features_without_full_reports(deployment_root):
    app = _open(deployment_root)
    app.radio(key="page").set_value("Gene explorer").run()
    assert not app.exception
    assert not app.error
    frame = pd.read_csv(deployment_root / "data/brca/BRCA_input_data.csv")
    row = frame.loc[frame.gene_symbol.eq("BRCA1")].iloc[0]
    app.selectbox(key="explorer_gene").set_value(row.gene_ens).run()
    assert not app.exception
    assert not app.error
    assert "BRCA1" in [element.value for element in app.subheader]
    shown = next(element.value for element in app.dataframe if "Feature" in element.value.columns)
    assert len(shown) == 13
    assert shown["Value"].tolist() == [float(row[name]) for name in shown["Feature"]]
    assert not app.metric
    assert not app.button
    assert "Setup" not in _text(app)


def test_all_review_pages_render_without_model_artifacts(deployment_root):
    app = _open(deployment_root)
    for page in REVIEW_PAGES:
        app.radio(key="page").set_value(page).run()
        assert not app.exception, page
        assert not app.error, page
    assert "About" in " ".join(element.value for element in app.title)
    assert "training" in _text(app).lower() or "prediction" in _text(app).lower()


def test_missing_feature_file_is_explained_without_breaking_results(deployment_root):
    (deployment_root / "data/brca/BRCA_input_data.csv").unlink()
    app = _open(deployment_root)
    assert app.metric[0].value == "0.921"
    app.radio(key="page").set_value("Gene explorer").run()
    assert not app.exception
    assert "unavailable" in _text(app).lower() or "missing" in _text(app).lower()
    app.radio(key="page").set_value("Model architecture").run()
    assert not app.exception
    assert not app.error


def test_entrypoint_does_not_import_scientific_runtime(deployment_root):
    """A separate interpreter also works when the full project's tests run first."""
    code = f'''import sys
from streamlit.testing.v1 import AppTest
app = AppTest.from_file({str(deployment_root / "cloud/app.py")!r}, default_timeout=30).run()
for page in {REVIEW_PAGES!r}:
    app.radio(key="page").set_value(page).run()
    assert not app.exception, str(app.exception)
assert not any(name == "torch" or name.startswith("torch.") or name == "dgl" or name.startswith("dgl.")
               for name in sys.modules), "Review pages imported model dependencies"
print("All review pages rendered without importing Torch or DGL")
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=deployment_root,
                            text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "without importing Torch or DGL" in result.stdout


def test_cloud_dependencies_and_server_binding_are_deployable():
    requirements = [line.strip() for line in (PROJECT / "cloud/requirements.txt").read_text().splitlines()
                    if line.strip() and not line.startswith("#")]
    assert requirements == ["streamlit==1.66.0", "numpy==1.26.4", "pandas==3.0.6",
                            "pyarrow==18.1.0", "graphviz==0.21"]
    config = tomllib.loads((PROJECT / ".streamlit/config.toml").read_text())
    assert "address" not in config.get("server", {}), "Do not pin the review deployment to loopback"
