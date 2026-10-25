# D: BRCA learning-rate and dropout tuning

Experiment D tests a small search over learning rate and DNN dropout using C's
training behavior: call `model.train()` before every training epoch, and use
`eval()` for validation and testing. It keeps the original reproduction and
experiments A, B and C separate. It does not change the Streamlit application or
the model used by that application.

The purpose is to test whether these settings improve outer-test performance.
Lower dropout or a smaller learning rate is not assumed to be better. Select
settings using validation AUC and report the resulting test metrics even if they
are worse than the control.

## Search settings

`configs/brca_tuning.yaml` uses `configs/brca_validation_train_mode.yaml` as its
base configuration. The candidate order is fixed:

| Candidate | Adam learning rate | DNN dropout pair |
| --- | --- | --- |
| 1: fixed C control | 0.001 | (0.7, 0.6) |
| 2 | 0.001 | (0.5, 0.3) |
| 3 | 0.0003 | (0.7, 0.6) |
| 4 | 0.0003 | (0.5, 0.3) |
| 5 | 0.0001 | (0.7, 0.6) |
| 6 | 0.0001 | (0.5, 0.3) |

The dropout pair supplies the probabilities at the DNN's two existing dropout
layers, in their existing order. Layer dimensions, BatchNorm/ReLU ordering,
custom sigmoid-attention GAT, target extraction and fusion remain unchanged.
There is no change to GAT dropout or attention, and no new model layer is added.

Each candidate runs all ten epochs with batch size 32, weighted NLL loss, Adam
with zero weight decay, and no scheduler or early stopping. The class weights
are `N_fit / (2 * n_fit_class)` and use fitting targets only. The prediction
threshold remains 0.5. Threshold tuning is outside this experiment.

## Data and installation

Use the project's Python 3.12 environment and install the dependencies in
`requirements.txt`; see the [environment setup](../README.md#environment-setup).
No new training dependency is needed for D. CPU is the established environment.
CUDA requires compatible PyTorch and DGL installations; this guide does not
establish numerical identity across devices.

Use the same prepared inputs as B and C:

```text
data/
├── brca/BRCA_input_data.csv
└── upstream/
    ├── using_Data/BRCA_label.csv
    └── for_ref/humannet_v2_and_v3.zip
```

The [prepared-data guide](brca_updated_data_experiment.md#place-the-files) gives
download links and placement instructions. ZIP extraction is not required.
The current dataset has 16,786 target genes, including 42 original driver
labels. The remaining 16,744 genes receive benchmark class 0; that does not
establish that they are biologically non-drivers. The same feature order,
precomputed values, label mapping and rebuilt PPI neighborhoods are retained.

## Run the experiment

From the project root:

```bash
python scripts/run_brca_tuning.py --config configs/brca_tuning.yaml
```

The default configuration uses seed 42 and CPU, with three outer folds running
in separate processes and one compute thread per process. Each fold trains its
six candidates sequentially. The complete search comprises 18 candidate runs,
or 180 training epochs. The fixed C control is already one candidate, so it
does not require an additional training run.

For a machine with less available memory, run one fold at a time:

```bash
python scripts/run_brca_tuning.py --config configs/brca_tuning.yaml --parallel-folds 1
```

To resume an interrupted run with the same inputs and settings:

```bash
python scripts/run_brca_tuning.py --config configs/brca_tuning.yaml --resume
```

Use a new output directory for a separate run or a changed seed:

```bash
python scripts/run_brca_tuning.py --config configs/brca_tuning.yaml --seed 43 --output-dir results/brca_tuning_seed43
```

Other overrides are `--device` and `--data-root`. The data root must contain both
`brca/` and `upstream/`; an authors-only checkout is not that combined directory.
Start with the project root as the working directory for these commands.
Keep each seed's complete results, including unsuccessful candidates. The
initial experiment is one seed only; the example above is a separate follow-up,
not a run automatically included by the default configuration.

### PyCharm

Create a **Python** run configuration:

- Interpreter: the project's Python 3.12 environment with requirements installed.
- Script: `scripts/run_brca_tuning.py` in this project.
- Parameters: `--config configs/brca_tuning.yaml`.
- Working directory: this reproduction project's root.

Press **Run**. To resume, add `--resume` to Parameters. To reduce simultaneous
work, add `--parallel-folds 1`. The runner sets the DGL PyTorch backend.

### macOS runtime and tests

On macOS, the tuning runner sets `KMP_INIT_AT_FORK=FALSE`, unless already set,
before importing PyTorch. This avoids OpenMP fork-time initialization problems
when starting worker processes. It is a runtime compatibility setting, not a
scientific model change. Scikit-learn uses the same setting as an OpenMP
workaround in its [initialization code](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/__init__.py).

The test runner does not pass through this training entry point. In the
restricted macOS environment used for this project, run tests from the project
root with the setting supplied explicitly:

```bash
KMP_INIT_AT_FORK=FALSE python -m pytest -c pytest.ini tests -q
```

For PyCharm's pytest run configuration, add `KMP_INIT_AT_FORK=FALSE` under
**Environment variables**. Test cases requiring optional data or CUDA can be
skipped when those dependencies are unavailable. Running the tests does not
perform the full 180-epoch experiment.

## Selection and evaluation procedure

1. Construct three outer splits with
   `StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)`.
2. Within each outer training fold, reserve a stratified 25% for validation.
   All six candidates use the same fitting, validation and test target IDs for
   that fold. The 25% holdout is a project choice inherited from B and C.
3. Train each candidate for ten epochs on the fitting targets only. Evaluate
   validation AUC after each epoch. Retain the entire state at the earliest
   maximum validation AUC, including BatchNorm buffers.
4. Select the candidate with the highest retained validation AUC. Break an
   exact tie by the candidate order above. The test fold has no role in choosing
   a candidate, its epoch or the classification threshold.
5. Freeze that selection, restore its saved state, and evaluate the outer test
   fold. Also evaluate the prespecified fixed C control's validation-selected
   state. If the winner is the control, reuse that test result.
6. Report per-fold and mean ± population standard deviation for AUC, accuracy,
   precision, recall, F1 and balanced accuracy, for both the selected D procedure
   and the fixed C control. Test AUC uses the positive-class probability.

Each outer fold may select different hyperparameters. D estimates the behavior
of this selection procedure; it does not identify one universal setting or
train a single final model on all genes. There is no refit on the full outer
training subset after selection and no full-dataset refit for deployment.

At seed 42, the expected target counts match B and C:

| Fold | Fitting drivers / class 0 | Validation drivers / class 0 | Test drivers / class 0 |
| --- | --- | --- | --- |
| 1 | 21 / 8,371 | 7 / 2,791 | 14 / 5,582 |
| 2 | 21 / 8,372 | 7 / 2,791 | 14 / 5,581 |
| 3 | 21 / 8,372 | 7 / 2,791 | 14 / 5,581 |

## Why D includes a matched C control

The historical C runner seeds once before cross-validation. Different amounts
of random-number consumption can therefore change later model initialization
and batch order. D instead resets the seed for every candidate in a fold and
uses separate generators for the fitting, validation and test data loaders.
This makes the comparison within D more directly controlled.

For one-based fold number `fold`, the model initialization seed is
`fold_seed = (seed + fold - 1) % 2**31`. Loader seeds are `fold_seed` for fitting,
`(fold_seed + 10000) % 2**31` for validation and
`(fold_seed + 20000) % 2**31` for testing. Candidates in the same fold therefore
start with matched parameter initialization and fitting-batch order. Their
updates can still differ because dropout probabilities and learning rates
differ.

These random-number controls are specific to D and its control. The historical
C implementation and recorded scores are unchanged. Compare D primarily with
the fixed C control rerun under D's random-number handling, not directly with
the historical C score. Historical B and C results provide context, and A's
test-based epoch selection makes A unsuitable as an unbiased comparator.

## Outputs and resuming

The default output directory is separate from all earlier reports:

```text
results/brca_tuning/
├── run.json
├── summary.json
├── fold_1/
│   ├── report.json
│   └── candidate_XX.pt
├── fold_2/
│   ├── report.json
│   └── candidate_XX.pt
└── fold_3/
    ├── report.json
    └── candidate_XX.pt
```

`candidate_XX.pt` represents the saved validation-selected state for each
candidate; multiple candidate files are present in each fold directory.
The run metadata records the effective experiment and its provenance. Fold
reports preserve candidate histories and selection evidence along with final
selected-model and control predictions. The summary combines completed folds.
Treat an incomplete run as incomplete rather than reporting partial folds as
the complete experiment.

Resume requires matching configuration, relevant source, runtime, inputs and
dataset fingerprints. This prevents silently combining incompatible runs.
After changing those dependencies, use a new output directory. Preserve the
complete existing directory when resuming; the saved states are needed to
restore the selected models.

Generated reports and checkpoints are local, Git-ignored experiment artifacts.
They do not update the application bundle or replace B's local demo model.
The separately exported compact snapshot
[`results/published/d.json`](../results/published/d.json) is committed for team
review; see the [results index](../results/README.md) for all five experiments.

## Interpretation and limitations

Use outer-test metrics to assess the prespecified comparison after selection.
The highest validation score is a selection statistic, not the final reported
performance. This search optimizes validation ROC-AUC; it can improve AUC while
reducing F1, precision or recall. Accuracy is dominated by the many class-0 genes.
Do not choose a different winner after inspecting test F1.

The initial run uses one seed and only seven validation positives per fold.
Searching six candidates and ten epochs can exploit noise in that small
validation set. Repeat the full selection procedure across prespecified seeds
before claiming a consistent improvement. Three-fold standard deviation is not
a confidence interval, and changing the search after observing test results
adds another source of selection bias.

The outer splits and their earlier A/B/C outcomes have already been examined in
this project. Although this runner does not use outer-test results for its
selection, these are not pristine external test data. Precomputed normalized
features and full graph-feature context remain shared across target splits.
The experiment does not establish fully inductive generalization, independent
biological validation or clinical usefulness.

## Results

The full seed-42 CPU experiment completed on **25 October 2026, Singapore
time**. Both `results/brca_tuning/run.json` and
`results/brca_tuning/summary.json` have `status: complete`. All six candidates
completed ten epochs in each of three outer folds: **18 candidate runs and
180 training epochs**. Three fold processes ran with one compute thread each.
The recorded environment was Python 3.12.14, PyTorch 2.2.2, DGL 2.2.0,
NumPy 1.26.4, pandas 3.0.6 and scikit-learn 1.9.1 on macOS arm64.

### Validation-selected settings

Each row is the candidate selected independently within that fold using only
validation AUC. Test AUC and F1 below are measured after restoring that selected
state; they did not determine the selection.

| Fold | Learning rate | Dropout pair | Selected epoch | Validation AUC | Test AUC | Test F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.001 | (0.5, 0.3) | 1 | 0.883554 | 0.947062 | 0.117647 |
| 2 | 0.001 | (0.5, 0.3) | 2 | 0.928341 | 0.911434 | 0.347826 |
| 3 | 0.0003 | (0.7, 0.6) | 7 | 0.936275 | 0.905483 | 0.156250 |

The lower dropout pair was selected in two folds, while fold 3 selected the
original dropout pair with a lower learning rate. These results do not establish
one universally best setting. No single final model was chosen or fitted on the
full dataset.

### Comparison with the matched fixed C control

Values are **mean ± population standard deviation** across the three outer
test folds. The control uses learning rate 0.001 and dropout (0.7, 0.6), with
its epoch selected separately by validation AUC. Its selected epochs were
10, 5 and 1. Both sides use D's per-fold seeds and separate loader generators;
these control scores are **not the historical C scores**.

| Test metric | Matched fixed C control | Validation-tuned D |
| --- | --- | --- |
| ROC-AUC | 0.876285 ± 0.066010 | 0.921326 ± 0.018359 |
| Accuracy | 0.989872 ± 0.001705 | 0.986954 ± 0.008027 |
| Precision | 0.105259 ± 0.021140 | 0.138249 ± 0.080319 |
| Recall | 0.404762 ± 0.121405 | 0.523810 ± 0.121405 |
| F1 | 0.165683 ± 0.035087 | 0.207241 ± 0.100650 |
| Balanced accuracy | 0.698051 ± 0.060210 | 0.755963 ± 0.058561 |

D increased mean AUC by **0.045041** and mean F1 by **0.041558** in this run.
Mean precision, recall and balanced accuracy also increased, while mean accuracy
fell by **0.002918**. The gains were not consistent across folds:

| Fold | Control test AUC | D test AUC | D minus control AUC |
| --- | --- | --- | --- |
| 1 | 0.783884 | 0.947062 | +0.163178 |
| 2 | 0.933998 | 0.911434 | -0.022564 |
| 3 | 0.910973 | 0.905483 | -0.005491 |

The mean AUC gain is driven by fold 1; AUC declined in folds 2 and 3. F1 shows a
different tradeoff: it declined in fold 1 and increased in folds 2 and 3. D's
mean F1 therefore should not be described as an improvement in every fold.
The larger F1 standard deviation also limits a claim of greater stability.

This is a **preliminary improvement in mean AUC and F1 against the matched
control on this one-seed experiment**, not proof of a generally better model.
Only seven labelled drivers were available for validation in each fold, and
the small grid trained every candidate for a fixed ten epochs. The lower
learning rates may have different convergence needs; this run does not answer
whether a longer schedule would change their ranking. Repeat the complete
prespecified selection procedure across seeds before making a broader
performance claim, retaining the earlier evaluation limitations.

### Verification and saved evidence

- Publication test suite, run from a clean copy containing only the files in
  this pull request and the existing repository: **187 passed, 2 skipped**.
  This includes the result-export checks and excludes the separate local
  Streamlit work. The skipped checks were CUDA and the optional cached
  raw-source preprocessing replay.
- An independent audit recalculated all six metrics from each of the 18 saved
  selected-validation prediction sets and six final winner/control test
  prediction sets. These agreed with the saved reports to absolute tolerance
  `1e-12`; aggregate means, population standard deviations and paired
  differences also agreed.
- The audit checked the 16,786 targets, split membership and counts, fitting-only
  class weights, all 180 epoch records and validation-based selection decisions.
  It checked hashes for all 18 checkpoint files, complete model-state structure,
  BatchNorm counters, matched initial states, relevant source, inputs and runtime.
- The audit did **not** replay full training or inference from the checkpoints.
  Its receipt is `results/brca_tuning_verification.json`. A final hash check
  confirmed that the 36 protected baseline and application files retained their
  pre-experiment contents.

The detailed reports, predictions, checkpoints and audit receipt remain local
and Git-ignored. A compact result snapshot is committed in
[`results/published/d.json`](../results/published/d.json), including fold metrics,
selected settings, the matched control and provenance without per-gene records,
checkpoints or machine-specific absolute paths. The
[results index](../results/README.md) compares the published original/A/B/C/D
snapshots and explains their different data and protocols. Streamlit and its
existing local B model are unchanged.

After all corresponding detailed reports exist locally, regenerate the compact
snapshots from the project root with:

```bash
python scripts/export_experiment_results.py
```

The [export script](../scripts/export_experiment_results.py) validates report
consistency and aggregates; it does not replay training or replace the separate
prediction-metric audit described above. A compact snapshot is not sufficient
to replay a run without obtaining the documented inputs.

The separate D implementation consists of
[`configs/brca_tuning.yaml`](../configs/brca_tuning.yaml),
[`scripts/run_brca_tuning.py`](../scripts/run_brca_tuning.py),
[`src/tuning.py`](../src/tuning.py), and the tests
[`test_tuning_core.py`](../tests/test_tuning_core.py),
[`test_tuning_protocol.py`](../tests/test_tuning_protocol.py) and
[`test_tuning_cli.py`](../tests/test_tuning_cli.py), together with this guide.
