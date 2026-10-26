# Local Streamlit app

The app lets the group explore the prepared BRCA features, inspect saved
cross-validation scores, compare experiments, and run the model for an existing
gene. It uses the existing DriverMONI architecture and graph construction.
Running the app does not start an experiment or modify the original, A, B, C or D
results.

This guide covers the full app on your own computer, including live predictions.
A separate online viewer shows published results, updated gene measurements and
the architecture without training or live inference. Follow the
[Community Cloud deployment guide](deploy_streamlit.md) to share that viewer.
Actual hosted deployment still needs to be checked after launch.

## Install and open

Use the project's **Python 3.12** environment. In the project root, install the
app requirements and launch it:

```bash
python -m pip install -r requirements-app.txt
python -m streamlit run streamlit_app.py --server.address 127.0.0.1
```

`requirements-app.txt` includes the existing model requirements and adds
Streamlit and a compatible PyArrow pin for its tables. Keep PyArrow 18.1.0:
the tested Arrow 26 wheel requires NumPy 2 at runtime, while the model requires
NumPy 1.26.4. Keep the pinned PyTorch, DGL and torchdata versions together. The
validated scientific environment is macOS arm64 with CPU; see the
[environment guide](../README.md#environment-setup) for platform limits.

Open the local address printed by Streamlit, usually `http://localhost:8501`.
Keep its process running while using the browser. Stop it from your IDE or with
Ctrl+C in the terminal when finished.
The explicit server address keeps this full app local to your computer. The
shared theme configuration leaves the address unset so it also works for the
separate hosted viewer.

## Open it in PyCharm

1. Open the reproduction project folder in PyCharm.
2. Select the project's **Python 3.12 interpreter** in the interpreter settings.
   Use the same environment that runs the model tests.
3. Open PyCharm's Terminal in the project root and paste:

   ```bash
   python -m pip install -r requirements-app.txt
   ```

4. Open **Run → Edit Configurations**, add a **Python** configuration and name it
   `DriverMONI app`. Choose **Module name** instead of Script path; menu wording
   may vary between PyCharm versions.
5. Set these fields:

   | Field | Value |
   | --- | --- |
   | Module name | `streamlit` |
   | Parameters | `run streamlit_app.py --server.address 127.0.0.1` |
   | Working directory | The `DriverMONI-Reproduction` project root |
   | Interpreter | The project's Python 3.12 environment |

6. Click **Run**, then open the local browser address shown in the Run panel.

This is a Python module configuration, not a pytest configuration. If PyCharm's
Terminal uses a different interpreter, activate the project environment there
before installing packages, or install `requirements-app.txt` through PyCharm's
package manager for the selected interpreter.

## What each page does

| Page | Purpose | Additional local files |
| --- | --- | --- |
| **Experiments** | Start here: read results and simple explanations in order: original reproduction, A, B, C, D. | None; the five compact result snapshots are tracked. |
| **Gene explorer** | Choose an experiment, find a gene and inspect its inputs; see a saved test score when a full report is available. | The selected experiment's feature CSV; a full report for saved gene scores. |
| **Live prediction** | Choose the B or D demonstration model and calculate a score for an existing gene. | Original labels and HumanNet, plus that model's local checkpoint. |
| **Model architecture** | Explain the DNN, custom sigmoid GAT, and fusion layers. | None. |
| **Setup** | Show the input, saved-report and model preparation steps. | None. |

A fresh clone includes `data/brca/BRCA_input_data.csv` and the five snapshots in
`results/published/`: `original.json`, `a.json`, `b.json`, `c.json` and `d.json`.
The Experiments page opens first, with D selected to show the newest experiment;
the experiment choices remain ordered from original through D. Its results are
available even without the feature CSV or model files. The original reproduction
uses a different feature file at
`data/upstream/sample_preprocessed_file/BRCA_input_data.csv`; Gene explorer
explains when that file is missing and lets you choose another experiment.

Full per-gene reports and model checkpoints are local generated files, ignored
by Git. The app explains missing files instead of generating random predictions.
The older `app_assets/experiment_summary.json` contains only A/B/C and is no
longer the source for the updated Experiments page.

## Add saved cross-validation scores

After following the [B input setup](brca_validation_experiment.md#inputs-and-environment),
run the full B experiment if you do not already have its completed report:

```bash
python scripts/run_brca.py --config configs/brca_validation.yaml --device cpu
```

The report is `results/brca_validation_metrics.json`. The app reads saved scores;
viewing them does not retrain the model. Reuse completed local reports instead
of rerunning solely to open the app:

| Experiment | Full local report |
| --- | --- |
| Original reproduction | `results/brca_metrics.json` |
| A | `results/brca_updated_data_metrics.json` |
| B | `results/brca_validation_metrics.json` |
| C | `results/brca_validation_train_mode_metrics.json` |
| D | `results/brca_tuning/summary.json` |

For B, C and D, each gene's saved test score comes from the model for the outer
fold that held that gene out. D uses the setting and epoch selected within that
fold using validation results. The full report contains predictions from
**three different trained models**, one per fold. These are not scores from one
model fitted to all genes.

The tracked result snapshots contain aggregate results and provenance, not
all per-gene predictions or any trained weights. They should not be used as a
substitute for a completed full report when inspecting individual scores.

To intentionally refresh the published snapshots after completing the five
experiments, run `python scripts/export_experiment_results.py`. It checks the
full reports and exports compact results into `results/published/`; it does not
retrain models. Review changed snapshots before sharing them. Opening the app
does not refresh these tracked files. See the [results index](../results/README.md)
for the differences between the experiments and what the exporter checks.

## Prepare a model for live prediction

Use the same prepared CSV and original auxiliary files as B:

```text
data/
├── brca/BRCA_input_data.csv
└── upstream/
    ├── using_Data/BRCA_label.csv
    └── for_ref/humannet_v2_and_v3.zip
```

The [prepared-data input guide](brca_updated_data_experiment.md#place-the-files)
links to the pinned authors' files and lists their checksums. No ZIP extraction
is needed. Do not replace the canonical CSV with a different preprocessing
output while using a checkpoint prepared from this one.

### D: use the already trained selected model

If the completed D fold report and its saved candidate checkpoint are available,
run this from the project root:

```bash
python scripts/prepare_tuned_demo.py --report results/brca_tuning/fold_1/report.json --output checkpoints/brca_demo_d.pt --device cpu
```

This prepares the model selected by validation in **D's first outer fold**.
It does not retrain or choose a model using test scores. It checks the source
report, input files, model code and saved weights, then checks that the exported
model reproduces the saved validation and test scores. The local outputs are
`checkpoints/brca_demo_d.pt` and its companion JSON preparation record.
Reuse an existing prepared D model instead of exporting it again. The exporter
refuses to overwrite an existing output or B's `brca_demo.pt` file.

In the completed D run, this model uses learning rate **0.001**, dropout
**(0.5, 0.3)** and selected epoch **1**. Those values describe this fold's saved
winner; they are not universal best settings or a new full-data model. If the
saved D artifacts are unavailable, reproduce D first using
`python scripts/run_brca_tuning.py --config configs/brca_tuning.yaml`; the compact
published summary cannot replace the saved weights and full fold report.

### B: retain the existing demonstration model

To prepare B if it is not already available:

```bash
python scripts/prepare_demo_model.py --config configs/brca_validation.yaml --output checkpoints/brca_demo.pt --device cpu
```

This separate command replays **B's first outer fold** for its configured ten
epochs. It uses the inner validation set to select the earliest maximum-AUC
epoch, then saves the complete selected state, including BatchNorm buffers, with
a JSON record of its preparation. It does not replace the official full CV
report. The outputs are `checkpoints/brca_demo.pt` and
`checkpoints/brca_demo.json`; both stay local.

The B and D demos are each one fold's validation-selected model, not new
full-data fits or ensembles. B retains the authors' evaluation-mode updates
after epoch one. D restores training mode during every training epoch and uses
the settings selected by D's validation search. The app uses separate paths:
`checkpoints/brca_demo.pt` for B and `checkpoints/brca_demo_d.pt` for D, preserving
the B artifact. It checks the embedded experiment identity before showing scores.

Live prediction lists B then D. It initially selects D if that checkpoint exists,
otherwise B. Switching models clears the previously loaded model and score;
press **Load trained model** and then **Run prediction** for the new choice.

In PyCharm, create a second **Python** run configuration with:

| Field | Value |
| --- | --- |
| Script path | `scripts/prepare_tuned_demo.py` in this project |
| Parameters | `--report results/brca_tuning/fold_1/report.json --output checkpoints/brca_demo_d.pt --device cpu` |
| Working directory | The project root |
| Interpreter | The same Python 3.12 environment |

After export, return to the app's **Live prediction** page. A selected gene is
looked up in the fixed feature table, its graph is built using the existing
dataset pipeline, and the model runs in evaluation mode. CPU is the demo's
default execution device. Raw mutations, patient samples and arbitrary gene
feature uploads are outside this app's scope.

## Read the scores correctly

- **Model score:** the positive-class softmax output. It is not a calibrated
  probability that a gene causes cancer, and it is not a patient diagnosis.
- **Benchmark label:** class 1 means the gene appears in the original driver
  list; class 0 means it is absent from that list. Class 0 does not prove the gene
  is biologically a non-driver. The later driver-evidence tables do not replace
  the labels used to train these models.
- **Live training membership:** a gene may belong to the exported model's
  fitting, validation, or test subset. Only its test-subset genes were held out
  from both fitting and epoch selection for that model. The shared graph and
  precomputed feature context still span target splits.
- **Saved versus live:** the live fold-1 model and the full-CV explorer need not
  give the same score for genes whose saved score came from fold 2 or 3.
- **Decision threshold:** the experiments use a score of at least 0.5 for the
  positive-class decision. AUC uses continuous scores. The score is more
  informative than presenting the threshold decision as biological truth.

The Experiments page explains the metrics in everyday language:

- **Ranking quality (AUC):** how well higher scores separate labelled drivers
  from the other targets; this uses the scores before a decision threshold.
- **Precision:** among targets predicted positive, the share labelled as drivers.
- **Recall:** the share of labelled drivers that the model finds.
- **F1:** a combined measure of precision and recall.
- **Accuracy:** the share of all targets assigned the correct benchmark label.
- **Balanced accuracy:** the average success rate for the two benchmark classes.

For presentations, show AUC alongside precision, recall and F1. Accuracy alone
can look high because only 42 of 16,786 updated-data targets are labelled drivers.
The original reproduction uses 15,028 targets and different input data.
Original/A select epochs using test-fold AUC; B/C/D use validation subsets.
These differences prevent a simple ranking of every row as better or worse.

D's main comparison is with its **matched fixed C control**, run with D's
random-number handling. It is separate from the historical C result. In this
run, D's mean AUC rose from 0.8763 to 0.9213 and mean F1 from 0.1657 to 0.2072,
while accuracy declined. The AUC gain came from fold 1; folds 2 and 3 declined.
This is promising one-seed evidence, not an improvement in every fold or proof
of general superiority. Fold standard deviation is not a confidence interval.
See the [D guide](brca_hyperparameter_tuning.md) and
[results index](../results/README.md) for full values and limitations.

## If something is missing

| Symptom | Action |
| --- | --- |
| `No module named streamlit` | Install `requirements-app.txt` in the interpreter selected by the run configuration. |
| DGL, PyTorch or another model dependency cannot import | Use Python 3.12 and install the complete app requirements together. Avoid independently upgrading DGL or PyTorch. Review the platform guidance in the main README. |
| Features are available, but no saved gene scores appear | Generate or restore the selected experiment's full JSON report under `results/`; compact snapshots do not contain per-gene scores. |
| Original reproduction has no feature table | Obtain its original CSV at `data/upstream/sample_preprocessed_file/BRCA_input_data.csv`, or choose A–D to browse the included updated CSV. |
| Live prediction has no model | Prepare the selected B or D checkpoint using its command above. Starting Streamlit does not train a model. |
| D's saved model or fold report is missing | Restore the completed local D output directory, or rerun D before preparing its demo model. The published JSON contains no weights. |
| Original label or HumanNet input is missing | Put the pinned files under `data/upstream` using the layout above. These files are not included in the repository. |
| A checkpoint or report is rejected after changing inputs | Use the inputs recorded for that artifact, or prepare a separate artifact for the changed data. Do not relabel an old result as a new run. |
| A relative path cannot be found | Use the reproduction project root as the working directory for the commands and PyCharm configurations in this guide. |

The full app remains a local research demonstration. The separate
[`cloud/app.py`](../cloud/app.py) entrypoint shares published results and updated
gene measurements online. It omits live prediction and local model setup;
model checkpoints, full per-gene reports and the authors' original feature CSV
are not part of that upload. See the [deployment guide](deploy_streamlit.md)
for the exact branch, entrypoint and Python settings. Hosting live inference
would require a separate deployment with its model and network inputs.

## Updated interface verification

Full project verification on **26 October 2026**: **294 passed, 2 skipped** in
58.05 seconds, including original-source parity checks. The skips cover unavailable
CUDA and the optional cached-source preprocessing replay. The model, training
code, experiment configurations, published results and existing B checkpoint
matched their pre-update checksums (21 protected files).

The imported D checkpoint selected candidate 02, epoch 1. All 44 weight/buffer
tensors match the saved tuning winner exactly. Replaying its 2,798 validation
and 5,596 test predictions produced zero score difference. Live BRCA1 and TP53
scores also match their saved fold-one scores. Importing D does not retrain it.

On **26 October 2026**, the updated interface passed **29 Streamlit AppTest
cases** in 6.51 seconds:

- All five pages and the ordered Original/A/B/C/D choices render from a fresh
  temporary project containing only committed input/result files.
- Displayed D summary values and selected settings match the published snapshot;
  the interface distinguishes D's matched control from historical C. The chart
  places scores side by side rather than stacking unrelated model scores.
- Experiment results remain usable without a feature CSV. Gene explorer uses
  the correct file for Original versus A–D and preserves all 13 feature values
  in their documented order.
- Missing, malformed, mismatched and mislabeled result files are rejected with
  readable messages. Synthetic per-gene reports check score switching through
  A/B/C/D without depending on ignored local reports.
- Switching between B and D clears old model weights and scores. Missing models,
  mismatched model identity, loading failures and prediction failures do not
  leave a stale successful prediction visible.

These interface tests use synthetic per-gene reports and mocked live inference
where needed; their scores are test fixtures, not new scientific measurements.
Model-export and numerical-inference checks are separate from AppTest.

Run the interface tests from the project root:

```bash
KMP_INIT_AT_FORK=FALSE python -m pytest -c pytest.ini tests/test_streamlit_app.py -q
```

The environment setting supports the restricted macOS runtime used for local
verification; it does not change the scientific model.

## Earlier B demonstration verification

Validated locally on 24 October 2026 with Python 3.12, CPU, Streamlit 1.66.0,
PyArrow 18.1.0 and the unchanged pinned scientific dependencies:

- Full project suite: **202 passed, 2 skipped** (CUDA unavailable and the optional
  cached-source preprocessing replay). Run `python -m pytest -c pytest.ini tests
  --brca-data-root /path/to/pinned/DriverMONI` for original-source parity checks.
- Sixteen Streamlit AppTest cases cover all pages, missing/corrupt inputs,
  experiment identity, feature order and clearing stale results after failures.
- The actual exported B fold-one checkpoint selected epoch 5. Its splits,
  class weights, epoch losses, validation scores and test scores exactly match
  the existing B fold-one report. Five single-gene held-out predictions agreed
  within `3.3e-9`; batch-size-dependent floating-point rounding is expected.
- Browser checks exercised TP53 and BRCA1 predictions, gene search, graph
  preview, feature display and the downloadable BRCA1 JSON.

Local training logs, the test log and `results/streamlit_verification.json`
record the checks. They and the checkpoint remain Git-ignored; the app does not
modify the baseline, A/B/C/D configurations or existing CV reports. These earlier
checks describe the B-only app before the D integration; they do not claim that
the new D interface or export was covered by that historical test count.
