"""Local BRCA research app: python -m streamlit run streamlit_app.py."""
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("DGLBACKEND", "pytorch")

import pandas as pd
import streamlit as st
from src.app_data import feature_rows, graph_preview_dot, read_features
from src.app_results import (
    DEFAULT_REPORT_FILENAMES, EXPERIMENT_ORDER, gene_prediction,
    load_published_experiments, load_report, validate_experiment,
)

PAGES = ("Experiments", "Gene explorer", "Live prediction", "Model architecture", "Setup")
REVIEW_PAGES = ("Experiments", "Gene explorer", "Model architecture", "About this app")
EXPERIMENT_NAMES = {
    "original": "Original · Paper reproduction", "A": "A · Updated data",
    "B": "B · Separate validation", "C": "C · Training active every round",
    "D": "D · Tuned learning settings",
}
DESCRIPTIONS = {
    "original": "Recreates the authors’ model and training behavior using their processed data.",
    "A": "Uses the team’s newly prepared data with the original training behavior.",
    "B": "Uses a separate validation group of genes to choose the training epoch, keeping the test genes for the final evaluation.",
    "C": "Builds on B and keeps dropout and batch normalization in training mode during every training round (epoch).",
    "D": "Builds on C and tries six combinations of learning rate and dropout. Validation genes choose the settings before the test genes are scored.",
}
METRIC_LABELS = {
    "auc": "Ranking quality (AUC)", "accuracy": "Overall correctness (accuracy)",
    "precision": "Positive prediction reliability (precision)",
    "recall": "Known drivers found (recall)", "f1": "Precision–recall balance (F1)",
    "balanced_accuracy": "Class-balanced correctness",
}
PREPARE_COMMAND = (
    "python scripts/prepare_demo_model.py --config configs/brca_validation.yaml "
    "--output checkpoints/brca_demo.pt --device cpu"
)
PREPARE_D_COMMAND = (
    "python scripts/prepare_tuned_demo.py --report results/brca_tuning/fold_1/report.json "
    "--output checkpoints/brca_demo_d.pt --device cpu"
)
CHECKPOINTS = {"B": "checkpoints/brca_demo.pt", "D": "checkpoints/brca_demo_d.pt"}


def file_key(path: Path) -> tuple[int, int]:
    """Invalidate cached reads when a file is replaced or modified."""
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


@st.cache_data(show_spinner=False)
def cached_features(path: str, signature: tuple[int, int]):
    return read_features(path)


@st.cache_data(show_spinner=False)
def cached_report(path: str, signature: tuple[int, int], feature_hash: str):
    return load_report(path, expected_feature_sha256=feature_hash)


def select_gene(frame: pd.DataFrame, key: str) -> str:
    ids = frame["gene_ens"].tolist()
    symbols = dict(zip(ids, frame["gene_symbol"], strict=True))
    preferred = frame.loc[frame["gene_symbol"].eq("TP53"), "gene_ens"]
    default = ids.index(preferred.iloc[0]) if not preferred.empty else 0
    if st.session_state.get(key) not in ids:
        st.session_state.pop(key, None)
    return st.selectbox("Find a gene", ids, index=default, key=key,
                        format_func=lambda gene: f"{symbols[gene]} · {gene}",
                        help="Type a gene name, such as TP53 or BRCA1, or its Ensembl identifier.")


def show_feature_table(row: pd.Series) -> None:
    st.subheader("What the model reads")
    st.write("Measurements describe mutations, gene activity, pathways and network information.")
    with st.expander("See the 13 input measurements"):
        st.dataframe(feature_rows(row), hide_index=True, width="stretch",
                     column_config={"Value": st.column_config.NumberColumn(format="%.6g")})
        st.caption("Exact model order: the gene branch uses rows 1–12; the network branch uses all 13. The app applies no extra scaling.")
        st.caption("‘deg’ is the expression feature. ‘PPI’ is a prepared network attribute, not the number of neighbors.")


def show_score(score: float, label: int, threshold: float) -> None:
    left, right = st.columns(2)
    left.metric("Driver score · 0 to 1", f"{score:.4f}")
    right.metric("Model prediction", "Driver" if score >= threshold else "Not flagged as a driver")
    st.write("A higher score means the model ranks this gene more strongly as a possible cancer driver.")
    st.caption("This is a model score, not the probability that a gene causes cancer.")
    st.caption(f"Decision cutoff: {threshold:g}. Reference label: " +
               ("listed driver." if label == 1 else "not in the supplied driver list; this does not prove it is a non-driver."))


def render_gene_explorer(root: Path, *, review_mode: bool = False) -> None:
    st.title("Explore a gene")
    st.write("Browse the measurements used by the model. This shared viewer does not calculate predictions."
             if review_mode else "See a gene’s measurements and saved score. This page does not run the model again.")
    experiment = st.selectbox("Recorded experiment", list(EXPERIMENT_ORDER), index=1,
                              format_func=EXPERIMENT_NAMES.get, key="saved_experiment")
    st.caption(DESCRIPTIONS[experiment])
    relative = ("data/upstream/sample_preprocessed_file/BRCA_input_data.csv"
                if experiment == "original" else "data/brca/BRCA_input_data.csv")
    path = root / relative
    if not path.is_file():
        st.info("The authors’ original feature CSV is not included in this shared viewer. Choose A, B, C or D to explore the team’s updated measurements."
                if review_mode and experiment == "original" else
                "The feature CSV for this experiment is unavailable."
                + (" The repository owner can restore the published data file." if review_mode else
                   " See Setup for its location, or choose another experiment."))
        return
    try:
        frame, feature_hash = cached_features(str(path), file_key(path))
    except (OSError, ValueError, TypeError) as error:
        st.error(f"Cannot read the feature dataset: {error}")
        return
    report = None
    report_path = root / "results" / DEFAULT_REPORT_FILENAMES[experiment]
    if review_mode:
        st.caption("Individual prediction scores are available in the local app. A–D share the same input measurements.")
    elif report_path.is_file():
        try:
            report = cached_report(str(report_path), file_key(report_path), feature_hash)
            validate_experiment(report, experiment)
        except (OSError, ValueError, KeyError, TypeError) as error:
            st.error(f"Cannot use this results file: {error}")
            report = None
    else:
        st.info("No per-gene report is saved locally for this experiment. You can still browse measurements and view the Experiments page.")
        st.caption("Setup explains how to restore a full report. Published summaries do not contain individual scores.")
    if report is not None and experiment == "original":
        target_ids = {gene for fold in report["folds"] for gene in fold["test_gene_ids"]}
        frame = frame[frame["gene_ens"].isin(target_ids)]
        if frame.empty:
            st.error("No experiment targets are present in this feature table.")
            return
    st.caption("Data: authors’ processed BRCA dataset." if experiment == "original"
               else "Data: the team’s updated BRCA dataset, shared by A, B, C and D.")
    gene = select_gene(frame, "explorer_gene")
    row = frame.set_index("gene_ens").loc[gene]
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader(str(row["gene_symbol"]))
        st.caption(gene)
        if report is not None:
            prediction = gene_prediction(report, gene)
            if prediction is None:
                st.warning("This gene is absent from the selected report.")
            else:
                st.caption(f"SAVED RESULT · Test round {prediction['fold']} of 3 · Training epoch {prediction['selected_epoch']}")
                show_score(prediction["driver_probability"], prediction["label"], prediction["threshold"])
                if prediction["test_fold_used_for_epoch_selection"]:
                    st.warning("This experiment used test results to choose its training epoch, so its reported performance may be optimistic.")
                else:
                    st.info("This gene was held out from fitting and model selection for this saved score. Precomputed measurements and network context are shared across the splits.")
                st.download_button("Download this recorded score", json.dumps(
                    {"gene_symbol": row["gene_symbol"], "feature_sha256": feature_hash,
                     "experiment": experiment, **prediction}, indent=2),
                    file_name=f"{gene}_{experiment}_recorded_score.json", mime="application/json")
    with right:
        show_feature_table(row)


def clear_live_model() -> None:
    for key in ("inference_bundle", "bundle_key", "live_output"):
        st.session_state.pop(key, None)


def render_live_prediction(root: Path) -> None:
    st.title("Run a gene prediction")
    st.write("Choose a saved model and let it score a gene using its measurements and protein connections.")
    experiment = st.selectbox("Live model", ["B", "D"],
                              index=1 if (root / CHECKPOINTS["D"]).is_file() else 0,
                              format_func=EXPERIMENT_NAMES.get, key="live_experiment")
    st.caption(DESCRIPTIONS[experiment])
    checkpoint = root / CHECKPOINTS[experiment]
    cache_key = (experiment, str(checkpoint), file_key(checkpoint) if checkpoint.is_file() else None)
    if st.session_state.get("bundle_key") != cache_key:
        clear_live_model()
    if not checkpoint.is_file():
        st.info("A trained demo checkpoint has not been prepared for this experiment on this computer yet.")
        with st.expander("How to add this model"):
            st.code(PREPARE_D_COMMAND if experiment == "D" else PREPARE_COMMAND, language="bash")
            st.write("D imports the winning model already saved by tuning. B’s preparation command trains its first fold. See Setup for the inputs.")
        return
    if st.button("Load trained model", type="primary", key="load_model"):
        clear_live_model()
        try:
            with st.spinner("Checking the saved model and its input data…"):
                from src.inference import load_inference_bundle
                bundle = load_inference_bundle(checkpoint, data_root=root / "data", device="cpu")
                if bundle.metadata["experiment"] != experiment:
                    raise ValueError("The saved model belongs to a different experiment. Use the matching checkpoint.")
                st.session_state["inference_bundle"] = bundle
                st.session_state["bundle_key"] = cache_key
        except (ImportError, OSError, ValueError, RuntimeError, EOFError, KeyError) as error:
            st.error(f"Could not load the demo model: {error}")
            st.info("See Setup for the environment, input files and model preparation.")
    bundle = st.session_state.get("inference_bundle")
    if bundle is None:
        st.caption("The model runs locally on your computer’s CPU. Loading it does not start training.")
        return
    metadata = bundle.metadata
    st.success(f"Model ready · Experiment {experiment} · Test round 1")
    st.caption("One saved model from the three-round experiment; it has not been refitted on all genes.")
    with st.expander("Model details"):
        st.write(f"Selected training epoch: {metadata['selected_epoch']}. Execution device: CPU.")
        if metadata.get("selected_hyperparameters"):
            st.json(metadata["selected_hyperparameters"])
    table = bundle.dataset.gene_table
    gene = select_gene(table, "live_gene")
    if st.button("Run prediction", type="primary", key="run_prediction"):
        st.session_state.pop("live_output", None)
        try:
            from src.inference import predict_gene
            with st.spinner("Reading the gene’s measurements and connections…"):
                st.session_state["live_output"] = predict_gene(bundle, gene)
        except (ValueError, RuntimeError) as error:
            st.error(f"Prediction could not run: {error}")
    result = st.session_state.get("live_output")
    if result is None or result["gene_id"] != gene:
        return
    st.subheader(result["gene_symbol"])
    roles = {"fit": "Used for training", "validation": "Used for model selection",
             "held_out_test": "Held out for testing"}
    st.caption(f"LIVE MODEL OUTPUT · {roles[result['split']]}")
    if result["split"] == "fit":
        st.warning("The model learned from this gene. Its score is a demonstration, not a test on an unseen target.")
    elif result["split"] == "validation":
        st.warning("This gene helped select the model. Its score is not a held-out test result.")
    else:
        st.info("This target was held out from fitting and model selection. Precomputed measurements and network context are shared across the splits.")
    show_score(result["driver_score"], result["label"], result["threshold"])
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("The gene’s local network")
        st.write("The highlighted gene is connected to nearby genes through the protein interaction network.")
        c1, c2 = st.columns(2)
        c1.metric("Genes in the full network", result["num_nodes"])
        c2.metric("Directed connections", result["num_edges"])
        sample = bundle.dataset[bundle.dataset.target_gene_ids.index(gene)]
        symbols = dict(zip(table["gene_ens"], table["gene_symbol"], strict=True))
        st.graphviz_chart(graph_preview_dot(sample, symbols), width="stretch")
        st.caption("For readability, this preview shows up to 25 genes. Prediction uses the full graph.")
        with st.expander("Network and tensor details"):
            st.write("The preview preserves directions, hides self-loops and marks repeated edges with ×count. Model input includes all self-loops and duplicate edges.")
            st.table(pd.DataFrame({
                "Stage": ["Gene input", "Gene embedding", "Network input", "Network messages", "Target embedding", "Fusion", "Class scores"],
                "Shape": ["1 × 12", "1 × 64", f"{result['num_nodes']} × 13", f"{result['num_nodes']} × 64", "1 × 32", "1 × 96", "1 × 2"],
            }))
    with right:
        show_feature_table(table.set_index("gene_ens").loc[gene])
    st.download_button("Download this live prediction", json.dumps(
        {"source": "live_fold_one_model", "experiment": metadata["experiment"],
         "selected_epoch": metadata["selected_epoch"], "dataset_sha256": metadata["dataset_sha256"],
         "shared_graph_feature_context": True, **result}, indent=2),
        file_name=f"{gene}_{experiment}_live_prediction.json", mime="application/json")


def render_experiments(root: Path) -> None:
    st.title("Compare the experiments")
    st.write("DriverMONI looks for genes that may help drive breast cancer. These experiments test changes to its data and learning procedure.")
    try:
        summary = load_published_experiments(root / "results/published")
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.error(f"The recorded results could not be loaded: {error}")
        return
    runs = {run["experiment"]: run for run in summary["experiments"]}
    experiment = st.selectbox("Choose an experiment", list(EXPERIMENT_ORDER), index=4,
                              format_func=EXPERIMENT_NAMES.get, key="experiment_choice")
    run = runs[experiment]
    st.subheader(EXPERIMENT_NAMES[experiment])
    st.write(DESCRIPTIONS[experiment])
    st.caption(f"{run['num_targets']:,} target genes · {run['dataset_counts']['driver']} labelled drivers · 3 test rounds · Random seed {run['seed']}")
    for column, key in zip(st.columns(4), ("auc", "precision", "recall", "f1")):
        column.metric(METRIC_LABELS[key], f"{run['summary'][key]['mean']:.3f}")
    st.caption("Average across three test rounds. All scores run from 0 to 1; higher is better for each measure.")
    if experiment == "D":
        control = run["matched_control"]
        delta = run["summary"]["auc"]["mean"] - control["summary"]["auc"]["mean"]
        st.subheader("Did tuning help?")
        st.write(f"D’s average AUC was {run['summary']['auc']['mean']:.3f}, compared with "
                 f"{control['summary']['auc']['mean']:.3f} for C run under the same tuning-experiment conditions "
                 f"(a change of {delta:+.3f}).")
        st.write("This comparison matches the data splits and starting random states, making it the relevant control for D.")
        keys = ("auc", "precision", "recall", "f1")
        chart_labels = {"auc": "AUC", "precision": "Precision", "recall": "Recall", "f1": "F1"}
        st.bar_chart(pd.DataFrame({
            "C · matched control": {chart_labels[key]: control["summary"][key]["mean"] for key in keys},
            "D · tuned settings": {chart_labels[key]: run["summary"][key]["mean"] for key in keys},
        }), color=["#7B8490", "#356596"], stack=False, sort=False,
            x_label="Metric", y_label="Average score", height=300,
            alt="Side-by-side comparison of D and its matched C control. D has higher mean AUC, precision, recall and F1 in this recorded run.")
        st.info("The average AUC gain came from one test round; AUC fell in the other two. This is a promising result from one random seed, not proof of a consistent improvement.")
        with st.expander("What was tuned?"):
            st.write("Learning rate controls the size of each learning step. Dropout temporarily switches off some internal units during training to reduce reliance on individual units.")
            st.write("Six combinations were tried. Validation results chose one combination per round; the outer test genes were used only after selection.")
            st.dataframe(pd.DataFrame([{
                "Test round": fold["fold"], "Learning rate": fold["learning_rate"],
                "Dropout": str(fold["dropout_pair"]), "Selected epoch": fold["selected_epoch"],
            } for fold in run["folds"]]), hide_index=True, width="stretch")
            st.caption("This matched C control uses different random-state handling from the earlier standalone C run.")
    elif experiment in ("original", "A"):
        st.warning("This experiment used test performance to choose its training epoch. Its reported scores may be optimistic; compare them with B–D cautiously.")
    elif experiment == "B":
        st.write("B separates model selection from testing. It preserves the original code’s switch to evaluation mode after the first epoch.")
    else:
        st.write("C keeps dropout and batch normalization active throughout training. The recorded standalone C run did not improve AUC or F1 over B.")
    with st.expander("What do these scores mean?"):
        st.markdown(
            "- **AUC:** how well labelled drivers rank above other genes; 0.5 is random ranking.\n"
            "- **Precision:** among genes flagged as drivers, the fraction in the supplied driver list.\n"
            "- **Recall:** the fraction of labelled drivers the model finds.\n"
            "- **F1:** a balance between precision and recall.\n"
            "- **Accuracy:** the fraction of all predictions matching supplied labels. With few drivers, this can look high even when driver detection is poor.\n"
            "- **Balanced accuracy:** averages correctness separately for the two classes."
        )
        st.caption("Class 0 means absent from the supplied driver list, not proven to be a non-driver. AUC uses continuous scores; the other measures use the recorded 0.5 cutoff.")
    with st.expander("All experiments · Original → A → B → C → D"):
        st.dataframe(pd.DataFrame([{
            "Experiment": EXPERIMENT_NAMES[item["experiment"]],
            **{label: f"{item['summary'][key]['mean']:.4f} ± {item['summary'][key]['std']:.4f}"
               for key, label in METRIC_LABELS.items()},
        } for item in runs.values()]), hide_index=True, width="stretch")
        st.warning("These are not a fair ranking of model quality: Original uses different data, Original/A select epochs on test results, and D includes an extra search.")
        st.caption("Mean ± population standard deviation across three rounds describes round-to-round variation, not a confidence interval. Historical C here is distinct from D’s matched control.")
    with st.expander("Results for each test round"):
        st.dataframe(pd.DataFrame([{
            "Test round": fold["fold"], "Selected epoch": fold["selected_epoch"],
            **{METRIC_LABELS[key]: fold["metrics"][key] for key in METRIC_LABELS},
        } for fold in run["folds"]]), hide_index=True, width="stretch")
        if experiment == "D":
            st.caption("Matched C control on the same test genes:")
            st.dataframe(pd.DataFrame([{
                "Test round": fold["fold"],
                **{METRIC_LABELS[key]: fold["metrics"][key] for key in METRIC_LABELS},
            } for fold in run["matched_control"]["folds"]]), hide_index=True, width="stretch")
    with st.expander("Source and reproducibility details"):
        st.write("These are recorded results. Opening this page does not retrain or rescore the model.")
        st.caption(f"Full report: {run['source_filename']}")
        st.caption(f"Report SHA-256: {run['report_sha256']}")
        st.caption(f"Feature SHA-256: {run['feature_sha256']}")
        st.write("Published files contain summaries, settings and input checksums. Per-gene scores and trained weights are separate local artifacts.")
    path = root / "results/published" / f"{experiment.lower()}.json"
    st.download_button("Download this experiment’s results", path.read_bytes(),
                       file_name=path.name, mime="application/json")


def render_architecture() -> None:
    st.title("Inside DriverMONI")
    st.write("The model combines two views of the same gene: its own measurements and information from its protein interaction network.")
    st.graphviz_chart('''digraph {
        rankdir=LR; bgcolor="transparent";
        node [shape=box, style="rounded,filled", fillcolor="#E5E7EB", color="#9CA3AF", fontcolor="#26313D", fontname="Helvetica"];
        edge [color="#6B7280"];
        gene [label="Gene measurements"]; dnn [label="Gene branch\\nPatterns in 12 measurements"];
        network [label="Gene and its connections"]; gat [label="Network branch\\nPatterns across neighboring genes"];
        fusion [label="Combine both views", fillcolor="#356596", color="#356596", fontcolor="#FFFFFF"]; score [label="Driver score"];
        gene -> dnn -> fusion; network -> gat -> fusion -> score;
    }''', width="stretch")
    st.write("The network branch considers the target gene and its immediate neighbors. After information passes through the network, the model reads the target gene’s representation and combines it with the gene branch.")
    st.info("Original and A–D share the same layer structure. D tunes learning rate and dropout; it does not add a new architecture.")
    with st.expander("Technical architecture and layer dimensions"):
        st.code(
            "Gene branch (DNN)\n"
            "12 → Linear(256) → BatchNorm → ReLU → Dropout\n"
            "   → Linear(128) → BatchNorm → ReLU → Dropout\n"
            "   → Linear(64)  → BatchNorm → ReLU\n\n"
            "Network branch (GAT)\n"
            "13 → bias-free projection(64)\n"
            "   → LeakyReLU attention → sigmoid → sum messages\n"
            "   → extract target → BatchNorm → ReLU\n"
            "   → Linear(32) → BatchNorm → ReLU\n"
            "   → extra Linear(32) → BatchNorm → ReLU\n\n"
            "Fusion: concatenate 64 + 32 → Linear(96, 2) → softmax scores", language="text")
        st.write("Original/A/B/C use dropout 0.7 and 0.6. D searches those values and the pair 0.5 and 0.3. During prediction dropout is disabled.")
        st.write("Attention uses unnormalized sigmoid weights. The final class scores separately use softmax over two logits.")
        st.caption("Neighborhood discovery is undirected; message edges retain CSV directions and duplicates, with self-loops. The extra 32 → 32 projection follows the active authors’ code and is absent from the simplified paper diagram.")


def render_setup(root: Path) -> None:
    st.title("Set up the local demo")
    st.write("Use the project’s Python 3.12 environment. Live predictions run locally on CPU.")
    st.code("python -m pip install -r requirements-app.txt\npython -m streamlit run streamlit_app.py", language="bash")
    st.subheader("What works immediately?")
    st.write("Published results for Original and A–D come with the project. The team’s updated CSV supports browsing genes. Individual saved scores and live models need additional local files.")
    with st.expander("File locations and availability"):
        files = {
            "Team’s BRCA features": "data/brca/BRCA_input_data.csv",
            "Authors’ BRCA features": "data/upstream/sample_preprocessed_file/BRCA_input_data.csv",
            **{f"Published results · {key}": f"results/published/{key.lower()}.json" for key in EXPERIMENT_ORDER},
            **{f"Live model · {key}": path for key, path in CHECKPOINTS.items()},
            **{f"Full report · {key}": f"results/{name}" for key, name in DEFAULT_REPORT_FILENAMES.items()},
        }
        st.dataframe(pd.DataFrame([{"Resource": name, "Location": path, "Available": (root / path).is_file()}
                                  for name, path in files.items()]), hide_index=True, width="stretch")
    st.subheader("Add a model for live predictions")
    st.write("Place the authors’ labels and HumanNet network under data/upstream/ as explained in docs/streamlit_app.md.")
    with st.expander("D · Use the model already saved by tuning"):
        st.code(PREPARE_D_COMMAND, language="bash")
        st.write("Requires the completed D fold-one report and candidate checkpoints. Imports and checks the winning model without training again.")
    with st.expander("B · Prepare a separate demo model"):
        st.code(PREPARE_COMMAND, language="bash")
        st.write("Trains B’s first fold and saves its validation-selected model. This does not replace the three-fold results.")
    st.subheader("Add recorded scores for individual genes")
    st.write("Reuse completed full reports if available. Otherwise run the appropriate experiment from the project folder:")
    with st.expander("Experiment commands"):
        st.code(
            "python scripts/run_brca.py --config configs/brca.yaml --device cpu\n"
            "python scripts/run_brca.py --config configs/brca_updated_data.yaml --device cpu\n"
            "python scripts/run_brca.py --config configs/brca_validation.yaml --device cpu\n"
            "python scripts/run_brca.py --config configs/brca_validation_train_mode.yaml --device cpu\n"
            "python scripts/run_brca_tuning.py --config configs/brca_tuning.yaml --device cpu", language="bash")
    st.caption("Weights and full reports stay local and are ignored by Git. Starting the app never starts training. See docs/streamlit_app.md for PyCharm instructions.")


def render_about() -> None:
    st.title("About this app")
    st.write("This shared viewer lets our group review the DriverMONI reproduction without installing Python or running an experiment.")
    st.markdown(
        "- **Experiments:** recorded results for Original and A–D, with explanations of the scores.\n"
        "- **Gene explorer:** the team’s prepared BRCA measurements in the model’s feature order.\n"
        "- **Model architecture:** how the gene and network branches combine."
    )
    st.info("Live prediction is available in the local app. This online viewer displays saved experiment summaries and input measurements; it does not train a model or generate new predictions.")
    st.write("The original processed inputs, trained model files and full per-gene reports are obtained or generated separately. The published result summaries include settings and input checksums for traceability.")
    st.link_button("Code, data instructions and experiment guides", "https://github.com/CO-ORSON/DriverMONI-Reproduction")
    st.caption("DriverMONI was proposed by Jeongho Park and Kyuri Jo. This is a group reproduction and experiment viewer, not the authors’ official application.")


def main(project_root: Path = PROJECT_ROOT, *, review_mode: bool = False) -> None:
    """Explicit root enables fresh-clone and missing-data tests."""
    root = Path(project_root).resolve()
    st.set_page_config(page_title="DriverMONI · BRCA Explorer", page_icon="🧬", layout="wide")
    pages = REVIEW_PAGES if review_mode else PAGES
    if st.session_state.get("page") not in pages:
        st.session_state.pop("page", None)
    with st.sidebar:
        st.title("DriverMONI")
        st.caption("Shared results viewer" if review_mode else "Understanding cancer-driver prediction")
        page = st.radio("Explore", pages, key="page")
        st.divider()
        st.caption("BRCA = breast cancer. A driver gene can contribute to cancer development.")
        st.caption("Group research demonstration. Not a clinical diagnostic tool.")
        st.link_button("Project on GitHub", "https://github.com/CO-ORSON/DriverMONI-Reproduction")
    if page == "Experiments":
        render_experiments(root)
    elif page == "Gene explorer":
        render_gene_explorer(root, review_mode=review_mode)
    elif page == "Live prediction":
        render_live_prediction(root)
    elif page == "Model architecture":
        render_architecture()
    elif page == "About this app":
        render_about()
    else:
        render_setup(root)


if __name__ == "__main__":
    main()
