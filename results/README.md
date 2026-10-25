# BRCA experiment results

Start with the compact result snapshots committed to this repository:
[original reproduction](published/original.json), [A](published/a.json),
[B](published/b.json), [C](published/c.json) and [D](published/d.json).
Each contains per-fold metrics, selected settings, aggregate metrics, dataset
counts and provenance. D includes its matched fixed C control. These are
completed experimental results, not smoke-test scores.

## What each experiment measures

| Experiment | Data and target count | Selection and training behavior |
| --- | --- | --- |
| Original reproduction | Authors' processed data; 15,028 genes | Select the best epoch using each test fold's AUC; preserve authors' train/eval behavior |
| A | Updated prepared data; 16,786 genes | Same source-compatible procedure as the original reproduction |
| B | Updated prepared data; 16,786 genes | Select epoch using an inner validation holdout; preserve authors' train/eval behavior |
| C | Same updated data and splits as B | Select epoch using validation; restore training mode every epoch |
| Matched C control in D | Same updated data; 16,786 genes | Fixed learning rate 0.001 and dropout (0.7, 0.6); C's training behavior with D's per-fold seeds and separate loader generators |
| D | Same data and random-number controls as its matched control | Select learning rate, dropout and epoch using validation only; restore training mode every epoch |

Every experiment has **42 labelled drivers**; the other targets are benchmark
class 0, not established biological non-drivers. All snapshots use seed 42,
three outer folds and ten epochs per candidate. The original reproduction and
A keep the authors' unusual behavior: training mode in epoch one, then
evaluation mode during later parameter updates. B preserves this behavior;
C and D restore training mode every epoch.

**This is not a ranking of interchangeable benchmarks.** The original
reproduction uses different data and graph context from A–D. The original
reproduction and A use the test folds to select epochs. B/C/D reserve 25% of
each outer training fold for validation, reducing the fitting population and
changing the selection procedure. D also changes random-number handling, so
its primary comparator is the matched control inside `published/d.json`, not
the historical C result.

## Completed results

Values are **mean ± population standard deviation** across three final outer
fold results (`std_ddof: 0`). They are not confidence intervals. AUC uses the
positive-class probability; threshold-based metrics use 0.5. Original and A
values are measured at their test-selected epochs and have that selection bias.

| Experiment | ROC-AUC | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Original reproduction | 0.917385 ± 0.008582 | 0.102249 ± 0.069126 | 0.523810 ± 0.121405 | 0.150854 ± 0.074647 |
| A | 0.932188 ± 0.020985 | 0.060602 ± 0.020310 | 0.523810 ± 0.089087 | 0.106439 ± 0.029981 |
| B | 0.918824 ± 0.016343 | 0.095931 ± 0.028804 | 0.333333 ± 0.134687 | 0.145655 ± 0.046222 |
| C | 0.905708 ± 0.034354 | 0.066514 ± 0.031041 | 0.357143 ± 0.303046 | 0.103845 ± 0.051883 |
| Matched C control in D | 0.876285 ± 0.066010 | 0.105259 ± 0.021140 | 0.404762 ± 0.121405 | 0.165683 ± 0.035087 |
| D | 0.921326 ± 0.018359 | 0.138249 ± 0.080319 | 0.523810 ± 0.121405 | 0.207241 ± 0.100650 |

| Experiment | Accuracy | Balanced accuracy |
| --- | --- | --- |
| Original reproduction | 0.977243 ± 0.012144 | 0.751162 ± 0.054469 |
| A | 0.975634 ± 0.008423 | 0.750288 ± 0.041631 |
| B | 0.990409 ± 0.003005 | 0.662695 ± 0.066192 |
| C | 0.985642 ± 0.005933 | 0.672181 ± 0.149281 |
| Matched C control in D | 0.989872 ± 0.001705 | 0.698051 ± 0.060210 |
| D | 0.986954 ± 0.008027 | 0.755963 ± 0.058561 |

### Main D comparison

Compared with its matched fixed C control, D increased mean AUC by **0.045041**
and mean F1 by **0.041558**, while accuracy fell by **0.002918**. The mean AUC
gain is driven by fold 1: paired AUC changes were **+0.163178, -0.022564 and
-0.005491**. F1 fell in fold 1 and rose in folds 2 and 3. This is preliminary
single-seed evidence, not a consistent improvement across folds or proof of a
generally better model.

D selected learning rate 0.001 with dropout (0.5, 0.3) in folds 1 and 2, and
0.0003 with dropout (0.7, 0.6) in fold 3. Their selected epochs were 1, 2 and 7.
It did not choose one globally optimal setting or refit a final deployment
model. See [the complete D guide](../docs/brca_hyperparameter_tuning.md) for
selection rules, per-fold results and verification.

Only seven labelled drivers are available in each validation subset. D searches
six candidates with a fixed ten-epoch budget. Repeat the complete selection
procedure across prespecified seeds before claiming a stable gain. Precomputed
features and graph context remain shared across target splits, and the group
has already examined earlier results on these targets; these are not pristine
external test data or independent biological validation.

## Run and export results

From the project root, after following the documented environment and data setup:

```bash
python scripts/run_brca_tuning.py --config configs/brca_tuning.yaml
```

D writes detailed fold reports and checkpoints under `results/brca_tuning/`,
plus `run.json` and `summary.json`. See its guide for resuming an interrupted
run or selecting a new output directory.

To regenerate all five compact snapshots from completed local reports:

```bash
python scripts/export_experiment_results.py
```

The exporter uses Python's standard library, reads the reports listed below and
writes `results/published/{original,a,b,c,d}.json`. It does not retrain models or
change the detailed source reports. A fresh clone contains the compact snapshots,
not the ignored detailed reports; to re-export, first reproduce the corresponding
full runs or obtain those detailed reports locally.

| Experiment | Configuration | Detailed local report | Guide |
| --- | --- | --- | --- |
| Original reproduction | `configs/brca.yaml` | `results/brca_metrics.json` | [README](../README.md) |
| A | `configs/brca_updated_data.yaml` | `results/brca_updated_data_metrics.json` | [A guide](../docs/brca_updated_data_experiment.md) |
| B | `configs/brca_validation.yaml` | `results/brca_validation_metrics.json` | [B guide](../docs/brca_validation_experiment.md) |
| C | `configs/brca_validation_train_mode.yaml` | `results/brca_validation_train_mode_metrics.json` | [C guide](../docs/brca_training_mode_experiment.md) |
| D and its matched control | `configs/brca_tuning.yaml` | `results/brca_tuning/summary.json` | [D guide](../docs/brca_hyperparameter_tuning.md) |

The original/A/B/C runner overwrites its configured output path on a repeated
run; use `--output` to preserve a separate run. D uses its own output directory
and explicit resume checks. Check each detailed JSON's status before using it;
the exporter rejects incomplete or smoke-test reports.

## Published summaries and local evidence

The committed JSON snapshots preserve metrics, class counts, selected epochs
and hyperparameters, protocol information, relevant configuration/runtime
details, input hashes and the source report's SHA-256. They omit per-gene
predictions, gene split membership, epoch histories, checkpoints, logs, source
input files and machine-specific absolute paths. They are compact records for
the team to inspect, not complete replay artifacts.

Detailed reports, predictions, checkpoints, logs and review receipts remain
local and Git-ignored. They preserve the additional evidence needed for a full
audit: exact target assignments, per-epoch histories and per-gene labels/scores.
Input files must be obtained through the documented data setup rather than
reconstructed from a compact result snapshot.

The exporter checks report structure, full-run coverage, selection consistency
and aggregate arithmetic. It does **not** recompute metrics from prediction
scores or replay training. A separate review independently recomputed all six
final metrics from the detailed saved scores for the five experiments, including
D's matched control, to absolute tolerance `1e-12`. That review is distinct from
the exporter. The original reproduction and A do not have a full protocol-audit
receipt, and publishing these snapshots did not involve a fresh training replay.
D's more extensive audit is described in its guide; it also did not replay full
training or checkpoint inference.

## Reading the validation-based reports

B/C reports distinguish the outer training fold from its fitting and validation
subsets. Class weights come from fitting targets only. Epoch histories contain
validation metrics; selected-validation predictions and final-test predictions
are separate. The summary aggregates final **test** results, not validation
results. Every target appears in one outer test fold. All ten epochs execute,
and the restored state includes the selected epoch's BatchNorm buffers; there
is no refit on the full outer training fold.

B records training mode only in epoch one; C records training mode in every
epoch. Their shared top-level seed does not guarantee identical later-fold
initial weights or batch orders because additional dropout consumes random
numbers. D resets each candidate's fold seed and uses separate data-loader
generators, which is why its fixed C control is reported separately. For D,
only the validation-selected winner and prespecified control are evaluated on
each test fold after selection is frozen.

`scripts/smoke_test.py` writes `brca_smoke_metrics.json` using only 12 real targets,
two epochs per fold and batch size 4. It checks execution; its metrics are not
published benchmark results. No result export updates Streamlit or its local
model bundle.
