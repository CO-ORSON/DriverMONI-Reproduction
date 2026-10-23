# B: BRCA validation-based epoch selection

B implements **suggestion 2 only**: select the epoch using a separate validation
subset within each outer training fold. It retains the authors' model and their
unusual evaluation-mode updates after epoch one. It is an experimental evaluation
protocol, not a claim of higher predictive performance or exact source parity.

## Which experiment to run

| Configuration | Input data | Epoch selection | Training mode |
| --- | --- | --- | --- |
| `configs/brca.yaml` | Original processed data and filtered targets | Test-fold AUC | Source behavior |
| `configs/brca_updated_data.yaml` (A) | Prepared CSV, all 16,786 targets | Test-fold AUC | Source behavior |
| `configs/brca_validation.yaml` (B) | Same prepared CSV as A | Inner validation AUC | Source behavior |
| `configs/brca_validation_train_mode.yaml` (C) | Same prepared CSV as B | Same inner validation as B | Training mode restored each epoch |

The original configurations and output filenames remain unchanged. The separate
[C experiment](brca_training_mode_experiment.md) uses B's validation protocol and
restores training mode each epoch. This document and its results describe B.

## Inputs and environment

Use the Python 3.12 environment in the [main README](../README.md#environment-setup).
Use the uploaded `data/brca/BRCA_input_data.csv` without rerunning preprocessing.
B requires the same two auxiliary files as A:

```text
data/
├── brca/BRCA_input_data.csv
└── upstream/
    ├── using_Data/BRCA_label.csv
    └── for_ref/humannet_v2_and_v3.zip
```

The [prepared-data input guide](brca_updated_data_experiment.md#place-the-files)
provides download links, placement instructions and expected checksums. No ZIP
extraction is needed. B retains all 16,786 targets, the original 42 positive
labels, the 12/13-feature ordering, rebuilt one-hop neighborhoods, and source edge
and self-loop behavior. The other 16,744 targets are class 0 under the benchmark
convention; they are not established biological non-drivers.

The teammate's manifest refresh and byte-identical CSV replay are documented in
[BRCA source preparation](brca_source_preparation.md). B consumes the fixed CSV;
it does not perform or independently certify raw-source replay.

## Run from the project root

```bash
python scripts/run_brca.py --config configs/brca_validation.yaml --device cpu
```

This writes `results/brca_validation_metrics.json`. A later run replaces that
file. To retain another run, choose a different output:

```bash
python scripts/run_brca.py --config configs/brca_validation.yaml --device cpu --output results/brca_validation_my_run.json
```

The YAML data base is `data`, so `--data-root` must point to a directory containing
both `brca/` and `upstream/`. Do not pass the authors-only checkout for this
configuration. YAML data/output bases are project-relative; CLI paths are relative
to the current working directory. See [configuration rules](configuration.md).

### PyCharm

Create a **Python** run configuration, not a pytest configuration:

- Interpreter: the existing Python 3.12 environment with project requirements.
- Script: this project's `scripts/run_brca.py`.
- Parameters: `--config configs/brca_validation.yaml --device cpu`.
- Working directory: this reproduction project's root.

The runner sets the DGL PyTorch backend. CPU is the validated environment;
compatible CUDA installations can use the configuration's `auto` device setting
by omitting `--device cpu`.

## Selection procedure and explicit choices

1. Use the same outer `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`
   assignments as A, given the same data and seed.
2. Within each outer training fold, reserve 25% using stratified
   `train_test_split(test_size=0.25, random_state=seed)`. The **25% fraction is a
   project choice**, not a value found in the authors' paper or implementation.
3. Compute class weights on the remaining fitting targets only:
   `N_fit / (2 * n_fit_class)`. Validation/test labels do not determine weights.
4. Train for ten epochs. After each epoch evaluate validation AUC and retain the
   earliest maximum together with the complete model state, including BatchNorm
   running statistics. The outer test fold is not evaluated during selection.
5. Restore the selected state and evaluate the outer test fold once. There is no
   early stopping and no refit using the full outer training fold.

Expected counts for the current CSV and seed 42:

| Fold | Outer train total | Fitting drivers / class 0 | Validation drivers / class 0 | Test drivers / class 0 |
| --- | --- | --- | --- | --- |
| 1 | 11,190 | 21 / 8,371 | 7 / 2,791 | 14 / 5,582 |
| 2 | 11,191 | 21 / 8,372 | 7 / 2,791 | 14 / 5,581 |
| 3 | 11,191 | 21 / 8,372 | 7 / 2,791 | 14 / 5,581 |

Adam at 0.001, batch size 32, weighted float64 NLL, zero weight decay, all model
dimensions/dropout values and the graph algorithm stay unchanged. There is no
scheduler. `preserve_eval_mode_after_first_evaluation: true` remains enabled:
only epoch one updates BatchNorm running statistics and applies dropout;
subsequent epochs still update learned parameters in evaluation mode. Suggestion
1 is deliberately absent from B.

Validation loading is unshuffled, an explicit project choice. The existing train
and test loader shuffle settings remain enabled. The saved protocol records the
inner split seed and validation loader order.

## Read the results

Require JSON `status: complete` before reporting a completed experiment. The
report saves configuration, input hashes, dependency versions, all outer/inner
target assignments, class counts, fitting-only weights, validation history and
selected validation predictions separately from final test predictions/metrics.
The aggregate mean and population standard deviation use the three outer **test**
folds. No model checkpoint file is saved.

Report AUC, precision, recall and F1 together. Validation has only seven labelled
drivers per fold, and test has fourteen, so scores and selected epochs can be
sensitive to the split. Fold standard deviation is not a confidence interval.

This protocol removes outer-test-based epoch selection. It does not eliminate
every possible information-sharing issue: precomputed normalized features and
the full graph-feature context are still shared across target splits. It is not
a fully inductive experiment and does not establish raw preprocessing parity.

For the [B-versus-C comparison](brca_training_mode_experiment.md), hold data,
all inner/outer splits, the configured seed and other settings fixed. The seed
is set once before CV; active dropout consumes additional random numbers, so
identical per-fold initial weights and minibatch orders are not guaranteed across
B and C. Comparing A's test-selected AUC against B is not a fair
test of model improvement; B also uses fewer fitting targets. Retain A as the
source-compatible reproduction record.

## Completed local experiment

Completed **23 October 2026 at 02:33 Singapore time**, in **15.3 minutes** on
macOS arm64 CPU, using seed 42 and the configuration above. All three folds and
all 30 training epochs completed. The environment was Python 3.12.14, PyTorch
2.2.2, DGL 2.2.0, NumPy 1.26.4, pandas 3.0.6 and scikit-learn 1.9.1.

| Fold | Selected epoch | Selected validation AUC | Test AUC | Test precision | Test recall | Test F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 5 | 0.848902 | 0.916287 | 0.080000 | 0.428571 | 0.134831 |
| 2 | 1 | 0.905973 | 0.939988 | 0.136364 | 0.428571 | 0.206897 |
| 3 | 2 | 0.875928 | 0.900197 | 0.071429 | 0.142857 | 0.095238 |

| Test metric | Mean ± population standard deviation across three folds |
| --- | --- |
| ROC-AUC | 0.918824 ± 0.016343 |
| Accuracy | 0.990409 ± 0.003005 |
| Precision | 0.095931 ± 0.028804 |
| Recall | 0.333333 ± 0.134687 |
| F1 | 0.145655 ± 0.046222 |
| Balanced accuracy | 0.662695 ± 0.066192 |

These are the final outer-test metrics from the validation-selected states.
This run establishes an executable B reference; it does not establish a model
improvement over A. The reduction in fitting targets and change in selection
protocol must accompany any presentation of the two experiments.

Verification completed:

- Full suite: **133 passed, 2 skipped** in 25.15 seconds. The skipped checks are
  CUDA and optional cached-source preprocessing replay.
- A controlled baseline run before and after the implementation change matched
  exactly in all non-timing report fields, including losses and predictions.
- Dedicated tests prove validation-only selection, earliest tie handling, full
  parameter/buffer restoration, and one outer-test evaluation per fold.
- All inner/outer assignments were reconstructed from the input data. Outer
  assignments match the existing A experiment; training, validation and test
  targets are pairwise disjoint within each fold, and every target is tested once.
- Class counts, fitting-only weights, source training-mode behavior, all ten
  epochs, finite losses/scores and constant learning rate were checked.
- All six test and selected-validation metrics were independently recalculated
  from saved predictions, as were test-fold summary means and standard deviations.
- Consumed input hashes and the effective configuration match the files. The
  original configurations, model, dataset loader, graph construction and metric
  implementation are unchanged.

Run the suite from the project root, supplying the full external authors' checkout:

```bash
python -m pytest tests -q --brca-data-root=/path/to/original/DriverMONI
```

The local run also has `results/brca_validation_run.log`,
`results/brca_validation_tests.log` and `results/brca_validation_verification.json`.
The verification receipt records source-file hashes and the audit checks. These
logs and receipt were produced during this review; the standard runner creates
the metrics JSON, not those extra files. Generated results remain ignored by Git;
the tables above are the shareable completion record. No full-run repeat or
independent raw-source replay was performed for B.
