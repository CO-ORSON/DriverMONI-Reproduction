"""Exercise the actual Streamlit pages without training or local model artifacts."""

import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from src.app_results import DEFAULT_REPORT_FILENAMES, METRIC_NAMES
from src.feature_schema import DNN_FEATURES, GAT_FEATURES

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def app_root(tmp_path: Path) -> Path:
    """A fresh clone has the updated CSV and five published compact results."""
    paths = ["data/brca/BRCA_input_data.csv"] + [
        f"results/published/{name}.json" for name in ("original", "a", "b", "c", "d")
    ]
    for relative in paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / relative, destination)
    return tmp_path


def _open(root: Path) -> AppTest:
    code = f"from pathlib import Path\nfrom streamlit_app import main\nmain(Path({str(root)!r}))"
    app = AppTest.from_string(code, default_timeout=20).run()
    assert not app.exception
    return app


def _messages(elements) -> str:
    return "\n".join(element.value for element in elements)


def _page(root: Path, name: str) -> AppTest:
    app = _open(root)
    app.radio(key="page").set_value(name).run()
    assert not app.exception
    return app


def _write_report(root: Path, experiment: str, *, filename_experiment: str | None = None) -> dict:
    """Synthetic twelve-gene CV with real CSV identifiers and distinct B/C scores.

    This is a UI fixture, not an asserted scientific result. Scores perfectly
    classify two drivers and two class-0 targets in each outer fold, with
    disjoint fitting/validation/test subsets and 25% validation holdouts.
    No ignored local report is required.
    """
    csv_path = root / ("data/upstream/sample_preprocessed_file/BRCA_input_data.csv"
                       if experiment == "original" else "data/brca/BRCA_input_data.csv")
    ids = pd.read_csv(csv_path)["gene_ens"].iloc[:12].tolist()
    positive = {"original": .6, "A": .7, "B": .9, "C": .8}[experiment]
    metrics = dict.fromkeys(METRIC_NAMES, 1.0)
    validation = experiment not in ("original", "A")
    report = {
        "status": "complete", "cancer": "BRCA", "num_targets": 12,
        "dataset_counts": {"driver": 6, "non_driver": 6},
        "input_files": [{"role": "features", "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest()}],
        "config": {
            "seed": 42,
            "data": {"use_comparison_filter": experiment == "original",
                     "use_supplied_neighbors": experiment == "original"},
            "training": {"epochs": 1, "preserve_eval_mode_after_first_evaluation": experiment != "C"},
            "cross_validation": {"n_splits": 3, "stratified": True, "shuffle": True,
                                 "epoch_selection": "best_validation_roc_auc" if validation else "best_test_roc_auc"},
            "evaluation": {"positive_class": 1, "probability_threshold": .5, "zero_division": 0},
        },
        "protocol": {"std_ddof": 0, "test_fold_used_for_epoch_selection": not validation,
                     "source_preserve_eval_mode": experiment != "C"},
        "summary": {name: {"mean": 1.0, "std": 0.0} for name in METRIC_NAMES},
        "folds": [],
    }
    if validation:
        report["config"]["cross_validation"]["validation_fraction"] = .25
        report["protocol"].update({"validation_fraction": .25, "test_evaluations_per_fold": 1,
                                   "refit_on_outer_training": False,
                                   "class_weight_population": "inner training targets only"})

    def predictions(genes: list[str]) -> list[dict]:
        return [{"gene_id": gene, "label": index % 2, "driver_probability": .1 if index % 2 == 0 else positive}
                for index, gene in enumerate(genes)]

    for index in range(3):
        test = ids[4 * index:4 * index + 4]
        remaining = [gene for gene in ids if gene not in test]
        val = remaining[:2] if validation else []
        fit = remaining[2:] if validation else remaining
        fold = {
            "fold": index + 1, "selected_epoch": 1, "metrics": metrics.copy(),
            "test_gene_ids": test, "train_gene_ids": fit,
            "test_counts": {"driver": 2, "non_driver": 2},
            "train_counts": {"driver": len(fit) // 2, "non_driver": len(fit) // 2},
            "class_weights": {"driver": 1.0, "non_driver": 1.0},
            "predictions": predictions(test),
            "epoch_history": [{"epoch": 1, "training_mode_during_updates": True,
                               "validation_metrics" if validation else "metrics": metrics.copy()}],
        }
        if validation:
            fold.update({"validation_gene_ids": val, "validation_counts": {"driver": 1, "non_driver": 1},
                         "validation_predictions": predictions(val), "selected_validation_metrics": metrics.copy()})
        report["folds"].append(fold)
    path = root / "results" / DEFAULT_REPORT_FILENAMES[filename_experiment or experiment]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report))
    return report


def _write_d_report(root: Path) -> dict:
    """UI-only D fixture: six tied settings, with the first candidate selected."""
    from tests.test_app_results import _d_report

    base = _write_report(root, "C")
    for fold in base["folds"]:
        for rows in (fold["predictions"], fold["validation_predictions"]):
            for row in rows:
                if row["label"]:
                    row["driver_probability"] = .85
    report = _d_report(base)
    path = root / "results" / DEFAULT_REPORT_FILENAMES["D"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report))
    return report


@pytest.mark.parametrize("page,title", [
    ("Gene explorer", "Explore a gene"),
    ("Live prediction", "Run a gene prediction"),
    ("Experiments", "Compare the experiments"),
    ("Model architecture", "Inside DriverMONI"),
    ("Setup", "Set up the local demo"),
])
def test_fresh_clone_all_pages_render(app_root: Path, page: str, title: str) -> None:
    app = _open(app_root)
    app.radio(key="page").set_value(page).run()
    assert not app.exception
    assert title in [element.value for element in app.title]
    assert not app.error
    if page == "Gene explorer":
        assert "No per-gene report" in _messages(app.info)
    elif page == "Live prediction":
        assert "checkpoint has not been prepared" in _messages(app.info)
        assert "prepare_demo_model.py" in _messages(app.code)
    elif page == "Experiments":
        assert "not a fair ranking" in _messages(app.warning)
        comparison = next(element.value for element in app.dataframe if "Experiment" in element.value.columns)
        assert comparison["Experiment"].str.split(" · ").str[0].tolist() == ["Original", "A", "B", "C", "D"]
        assert "Did tuning help?" in [element.value for element in app.subheader]
        assert "compared with 0.876" in _messages(app.markdown)
        assert "AUC fell in the other two" in _messages(app.info)


def test_experiments_are_ordered_and_d_uses_its_matched_control(app_root: Path) -> None:
    from streamlit_app import EXPERIMENT_NAMES, PAGES

    app = _open(app_root)
    assert app.radio(key="page").value == "Experiments"
    assert app.radio(key="page").options == list(PAGES)
    assert app.selectbox(key="experiment_choice").options == [
        EXPERIMENT_NAMES[name] for name in ("original", "A", "B", "C", "D")
    ]
    assert app.selectbox(key="experiment_choice").value == "D"
    assert [metric.value for metric in app.metric] == ["0.921", "0.138", "0.524", "0.207"]
    chart = app.get("vega_lite_chart")[0]
    encoding = json.loads(chart.proto.spec)["encoding"]
    assert encoding["y"]["stack"] is False
    assert encoding["xOffset"]["field"] == encoding["color"]["field"]
    assert encoding["color"]["scale"]["domain"] == ["C · matched control", "D · tuned settings"]
    assert "Side-by-side" in chart.proto.alt
    tuning = next(element.value for element in app.dataframe if "Learning rate" in element.value.columns)
    assert tuning["Learning rate"].tolist() == [.001, .001, .0003]
    assert tuning["Selected epoch"].tolist() == [1, 2, 7]
    assert "historical" in _messages(app.caption).lower() or "earlier standalone C" in _messages(app.caption)
    for name, expected_auc in [("original", "0.917"), ("A", "0.932"), ("B", "0.919"), ("C", "0.906")]:
        app.selectbox(key="experiment_choice").set_value(name).run()
        assert not app.exception
        assert app.metric[0].value == expected_auc
        if name in ("original", "A"):
            assert "used test performance" in _messages(app.warning)


def test_experiment_results_do_not_require_a_feature_csv(app_root: Path) -> None:
    (app_root / "data/brca/BRCA_input_data.csv").unlink()
    app = _open(app_root)
    assert not app.error
    assert app.metric[0].value == "0.921"


@pytest.mark.parametrize("problem", ["missing", "malformed_json", "wrong_feature_hash", "tampered_metrics"])
def test_unusable_published_results_are_actionable(app_root: Path, problem: str) -> None:
    path = app_root / "results/published/d.json"
    if problem == "missing":
        path.unlink()
    elif problem == "malformed_json":
        path.write_text("{ invalid")
    else:
        report = json.loads(path.read_text())
        if problem == "wrong_feature_hash":
            next(item for item in report["provenance"]["input_files"] if item["role"] == "features")["sha256"] = "0" * 64
        else:
            report["tuned_summary"]["auc"]["mean"] = .1
        path.write_text(json.dumps(report))
    app = _open(app_root)
    assert "The recorded results could not be loaded" in _messages(app.error)
    assert not app.metric
    app.radio(key="page").set_value("Model architecture").run()
    assert not app.exception
    assert not app.error


def test_gene_selection_preserves_exact_feature_order_and_values(app_root: Path) -> None:
    source = pd.read_csv(app_root / "data/brca/BRCA_input_data.csv").drop_duplicates("gene_symbol")
    row = source.loc[source["gene_symbol"].eq("BRCA1")].iloc[0]
    app = _page(app_root, "Gene explorer")
    assert any("BRCA1" in option and row["gene_ens"] in option for option in app.selectbox(key="explorer_gene").options)
    app.selectbox(key="explorer_gene").set_value(row["gene_ens"]).run()
    assert not app.exception
    shown = app.dataframe[0].value
    assert shown["Feature"].tolist() == list(GAT_FEATURES)
    assert shown["Value"].tolist() == [float(row[name]) for name in GAT_FEATURES]
    assert shown["Branch"].tolist() == ["DNN + GAT"] * len(DNN_FEATURES) + ["GAT only"]
    assert "No per-gene report" in _messages(app.info)
    assert len(app.metric) == 0


@pytest.mark.parametrize("problem", ["missing", "missing_column", "nonfinite"])
def test_unavailable_or_invalid_csv_is_actionable(app_root: Path, problem: str) -> None:
    path = app_root / "data/brca/BRCA_input_data.csv"
    if problem == "missing":
        path.unlink()
    else:
        frame = pd.read_csv(path).iloc[:3].copy()
        if problem == "missing_column":
            frame = frame.drop(columns=[GAT_FEATURES[-1]])
        else:
            frame.loc[frame.index[0], GAT_FEATURES[0]] = float("inf")
        frame.to_csv(path, index=False)
    app = _page(app_root, "Gene explorer")
    if problem == "missing":
        assert "feature CSV for this experiment is unavailable" in _messages(app.info)
    else:
        assert "Cannot read the feature dataset" in _messages(app.error)
    app.radio(key="page").set_value("Model architecture").run()
    assert not app.exception
    assert not app.error


def test_switching_saved_experiments_changes_scores_and_explains_selection(app_root: Path) -> None:
    report = _write_report(app_root, "B")
    _write_report(app_root, "C")
    _write_report(app_root, "A")
    _write_d_report(app_root)
    gene = report["folds"][0]["test_gene_ids"][1]
    app = _page(app_root, "Gene explorer")
    app.selectbox(key="saved_experiment").set_value("B").run()
    app.selectbox(key="explorer_gene").set_value(gene).run()
    assert not app.exception
    assert app.metric[0].value == "0.9000"
    assert "held out from fitting and model selection" in _messages(app.info)
    assert "SAVED RESULT · Test round 1 of 3" in _messages(app.caption)
    app.selectbox(key="saved_experiment").set_value("C").run()
    assert not app.exception
    assert app.metric[0].value == "0.8000"
    app.selectbox(key="saved_experiment").set_value("D").run()
    assert not app.exception
    assert app.metric[0].value == "0.8500"
    app.selectbox(key="saved_experiment").set_value("A").run()
    assert not app.exception
    assert app.metric[0].value == "0.7000"
    assert "used test results to choose" in _messages(app.warning)


@pytest.mark.parametrize("problem", ["malformed_json", "wrong_hash", "tampered_metrics"])
def test_invalid_report_never_displays_a_prediction(app_root: Path, problem: str) -> None:
    report = _write_report(app_root, "B")
    path = app_root / "results" / DEFAULT_REPORT_FILENAMES["B"]
    if problem == "malformed_json":
        path.write_text("{ invalid JSON")
    else:
        if problem == "wrong_hash":
            report["input_files"][0]["sha256"] = "0" * 64
        else:
            report["summary"]["auc"]["mean"] = .2
        path.write_text(json.dumps(report))
    app = _page(app_root, "Gene explorer")
    app.selectbox(key="saved_experiment").set_value("B").run()
    assert "Cannot use this results file" in _messages(app.error)
    assert len(app.metric) == 0
    assert len(app.dataframe[0].value) == 13


def test_report_named_c_must_actually_use_c_protocol(app_root: Path) -> None:
    report = _write_report(app_root, "B", filename_experiment="C")
    app = _page(app_root, "Gene explorer")
    app.selectbox(key="explorer_gene").set_value(report["folds"][0]["test_gene_ids"][1]).run()
    app.selectbox(key="saved_experiment").set_value("C").run()
    assert not app.exception
    assert "Cannot use this results file" in _messages(app.error)
    assert len(app.metric) == 0


def test_saved_experiment_order_and_missing_original_data(app_root: Path) -> None:
    from streamlit_app import EXPERIMENT_NAMES

    app = _page(app_root, "Gene explorer")
    assert app.selectbox(key="saved_experiment").value == "A"
    assert app.selectbox(key="saved_experiment").options == [
        EXPERIMENT_NAMES[name] for name in ("original", "A", "B", "C", "D")
    ]
    app.selectbox(key="saved_experiment").set_value("original").run()
    assert not app.exception
    assert "feature CSV for this experiment is unavailable" in _messages(app.info)
    assert not app.metric
    app.selectbox(key="saved_experiment").set_value("D").run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 13


def test_original_gene_scores_use_the_original_feature_file(app_root: Path) -> None:
    original_csv = app_root / "data/upstream/sample_preprocessed_file/BRCA_input_data.csv"
    original_csv.parent.mkdir(parents=True)
    original_frame = pd.read_csv(app_root / "data/brca/BRCA_input_data.csv").iloc[:12].copy()
    original_frame.loc[original_frame.index[1], GAT_FEATURES[0]] = .123456
    original_frame.to_csv(original_csv, index=False)
    original = _write_report(app_root, "original")
    _write_report(app_root, "A")
    gene = original["folds"][0]["test_gene_ids"][1]
    app = _page(app_root, "Gene explorer")
    app.selectbox(key="saved_experiment").set_value("original").run()
    app.selectbox(key="explorer_gene").set_value(gene).run()
    assert not app.exception
    assert not app.error
    assert app.metric[0].value == "0.6000"
    assert app.dataframe[0].value["Value"].iloc[0] == .123456
    assert "used test results to choose" in _messages(app.warning)
    app.selectbox(key="saved_experiment").set_value("A").run()
    assert not app.exception
    assert app.metric[0].value == "0.7000"
    updated = pd.read_csv(app_root / "data/brca/BRCA_input_data.csv").set_index("gene_ens").loc[gene]
    assert app.dataframe[0].value["Value"].iloc[0] == updated[GAT_FEATURES[0]]


def test_report_named_d_requires_the_tuning_protocol(app_root: Path) -> None:
    _write_report(app_root, "C", filename_experiment="D")
    app = _page(app_root, "Gene explorer")
    app.selectbox(key="saved_experiment").set_value("D").run()
    assert not app.exception
    assert "requires the recorded validation hyperparameter search" in _messages(app.error)
    assert not app.metric


class _DisplayDataset:
    """A display-only one-gene graph; inference behavior is tested elsewhere."""

    def __init__(self, table: pd.DataFrame) -> None:
        self.gene_table = table
        self.target_gene_ids = tuple(table["gene_ens"])

    def __getitem__(self, index: int):
        gene = self.target_gene_ids[index]
        graph = SimpleNamespace(edges=lambda: (pd.Series([0]), pd.Series([0])))
        return SimpleNamespace(gene_id=gene, node_gene_ids=(gene,), graph=graph)


def _live_ui_with_previous_result(
    root: Path, monkeypatch: pytest.MonkeyPatch, *, experiment: str = "B", both: bool = False,
) -> AppTest:
    """Reach a visible live result through the actual buttons with fake inference."""
    from src import inference

    table = pd.read_csv(root / "data/brca/BRCA_input_data.csv").iloc[:1].copy()
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    for name in (("B", "D") if both else (experiment,)):
        checkpoint = checkpoint_dir / ("brca_demo_d.pt" if name == "D" else "brca_demo.pt")
        checkpoint.write_bytes(b"UI-only sentinel: loader is mocked")

    def load_bundle(path, **kwargs):
        name = "D" if Path(path).name == "brca_demo_d.pt" else "B"
        metadata = {"experiment": name, "selected_epoch": 1 if name == "D" else 5,
                    "dataset_sha256": "a" * 64}
        if name == "D":
            metadata["selected_hyperparameters"] = {"candidate_id": "candidate_02",
                                                     "learning_rate": .001, "dropout_pair": [.5, .3]}
        return SimpleNamespace(metadata=metadata, dataset=_DisplayDataset(table))

    result = {"gene_id": table.iloc[0]["gene_ens"], "gene_symbol": table.iloc[0]["gene_symbol"],
              "split": "held_out_test", "driver_score": .91, "label": 1,
              "threshold": .5, "num_nodes": 1, "num_edges": 1}
    monkeypatch.setattr(inference, "load_inference_bundle", load_bundle)
    monkeypatch.setattr(inference, "predict_gene", lambda bundle, gene: dict(
        result, driver_score=.81 if bundle.metadata["experiment"] == "D" else .91))
    app = _open(root)
    app.radio(key="page").set_value("Live prediction").run()
    assert app.selectbox(key="live_experiment").value == ("D" if both or experiment == "D" else "B")
    app.selectbox(key="live_experiment").set_value(experiment).run()
    app.button(key="load_model").click().run()
    app.button(key="run_prediction").click().run()
    assert not app.exception
    assert "Model ready" in _messages(app.success)
    assert app.metric[0].value == ("0.8100" if experiment == "D" else "0.9100")
    assert "LIVE MODEL OUTPUT · Held out for testing" in _messages(app.caption)
    return app


@pytest.mark.parametrize("reason", ["Input checksum changed", "Model source checksum changed"])
def test_failed_model_reload_clears_old_model_and_result(
    app_root: Path, monkeypatch: pytest.MonkeyPatch, reason: str,
) -> None:
    from src import inference

    app = _live_ui_with_previous_result(app_root, monkeypatch)

    def reject_load(*args, **kwargs):
        raise ValueError(reason)

    monkeypatch.setattr(inference, "load_inference_bundle", reject_load)
    app.button(key="load_model").click().run()
    assert not app.exception
    assert f"Could not load the demo model: {reason}" in _messages(app.error)
    assert not app.success
    assert not app.metric
    assert "inference_bundle" not in app.session_state
    assert "live_output" not in app.session_state


def test_switching_live_models_clears_previous_weights_and_score(app_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit_app import EXPERIMENT_NAMES

    app = _live_ui_with_previous_result(app_root, monkeypatch, experiment="D", both=True)
    assert app.selectbox(key="live_experiment").options == [EXPERIMENT_NAMES[name] for name in ("B", "D")]
    for experiment, expected_score in (("B", "0.9100"), ("D", "0.8100")):
        app.selectbox(key="live_experiment").set_value(experiment).run()
        assert not app.exception
        assert not app.metric
        assert not app.success
        assert "inference_bundle" not in app.session_state
        assert "live_output" not in app.session_state
        app.button(key="load_model").click().run()
        assert not app.metric
        assert app.session_state["inference_bundle"].metadata["experiment"] == experiment
        app.button(key="run_prediction").click().run()
        assert not app.exception
        assert app.metric[0].value == expected_score


def test_switching_to_missing_d_model_does_not_reuse_b_score(app_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _live_ui_with_previous_result(app_root, monkeypatch)
    app.selectbox(key="live_experiment").set_value("D").run()
    assert not app.exception
    assert "checkpoint has not been prepared for this experiment" in _messages(app.info)
    assert "prepare_tuned_demo.py" in _messages(app.code)
    assert not app.metric
    assert "inference_bundle" not in app.session_state
    assert "live_output" not in app.session_state


def test_mislabeled_live_checkpoint_is_rejected(app_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src import inference

    app = _live_ui_with_previous_result(app_root, monkeypatch, experiment="D", both=True)
    wrong_bundle = SimpleNamespace(metadata={"experiment": "B"})
    monkeypatch.setattr(inference, "load_inference_bundle", lambda *args, **kwargs: wrong_bundle)
    app.button(key="load_model").click().run()
    assert not app.exception
    assert "saved model belongs to a different experiment" in _messages(app.error)
    assert not app.metric
    assert "inference_bundle" not in app.session_state
    assert "live_output" not in app.session_state


def test_failed_prediction_removes_previous_successful_score(app_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src import inference

    app = _live_ui_with_previous_result(app_root, monkeypatch)

    def reject_prediction(*args, **kwargs):
        raise ValueError("Target graph unavailable")

    monkeypatch.setattr(inference, "predict_gene", reject_prediction)
    app.button(key="run_prediction").click().run()
    assert not app.exception
    assert "Prediction could not run: Target graph unavailable" in _messages(app.error)
    assert "Model ready" in _messages(app.success)
    assert not app.metric
    assert "LIVE MODEL OUTPUT" not in _messages(app.caption)
    assert "live_output" not in app.session_state
