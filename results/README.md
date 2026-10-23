# BRCA experiment results

The runner writes `brca_metrics.json` here by default. It overwrites that path on
a new experiment; use `--output` to retain separate runs. Generated files are
excluded from Git. Check the JSON's `status` before interpreting results:
`running`, `complete`, `failed`, or `interrupted`.

The separate `configs/brca_updated_data.yaml` experiment writes
`brca_updated_data_metrics.json`. It uses all 16,786 newly prepared genes and
rebuilt neighborhoods with the same model/training settings. Its population and
graph context differ from the baseline; interpret it as a distinct experiment.
See [its guide and provenance limits](../docs/brca_updated_data_experiment.md).

The **B experiment**, `configs/brca_validation.yaml`, writes
`brca_validation_metrics.json`. It uses the same prepared dataset, but selects
epochs on an inner validation subset and evaluates each outer test fold once
after restoring the selected model. The original training-mode behavior remains
enabled. See [the B guide](../docs/brca_validation_experiment.md).

The **C experiment**, `configs/brca_validation_train_mode.yaml`, writes
`brca_validation_train_mode_metrics.json`. It keeps B's validation protocol and
calls `model.train()` before every training epoch, restoring training-mode
dropout and BatchNorm throughout optimization. Validation and test inference
remain in evaluation mode. See [the C guide](../docs/brca_training_mode_experiment.md).

`scripts/smoke_test.py` writes `brca_smoke_metrics.json` separately. It runs only
12 real targets, two epochs per fold, batch size 4, and three folds on CPU; the
`smoke_test` field identifies its limited scope. Do not use its scores as a full
BRCA benchmark. The completed full-run summary is recorded in the root README so
teammates can see it without redistributing per-gene input or prediction records.

The report contains:

- Effective configuration, seed, device, dependency versions, source revision,
  input paths, SHA-256 hashes, and execution timestamps.
- Full target-gene fold assignments and train/test driver/non-driver counts.
- Training-fold class weights and per-epoch mean batch loss, learning rate,
  train/eval mode, metrics, and elapsed time.
- Each fold's selected epoch, metrics, and per-gene true labels/driver scores.
- Unweighted mean and configurable standard deviation across the three folds.

Metrics are `auc`, `accuracy`, `precision`, `recall`, `f1`, and
`balanced_accuracy`. AUC uses positive-class probabilities. Other metrics use
probability >= 0.5 by default. `std_ddof: 0` means population standard deviation
across the three fold scores; it is not a confidence interval.

The source-compatible protocol selects each fold's epoch by held-out AUC and
reports every metric at that same epoch. It also preserves evaluation-mode
parameter updates after epoch one. The held-out folds therefore participate in
epoch selection; these scores are not an untouched-test-set estimate. Neither a
separate validation split nor early stopping is silently introduced.

## B and C output interpretation

With `best_validation_roc_auc`, the report additionally distinguishes the outer
training fold from its fitting and validation subsets. Class weights come from
the fitting subset only. The epoch history contains validation metrics; selected
validation metrics/predictions are recorded separately from the final test
metrics/predictions. Each test target occurs in exactly one outer test fold.

The summary aggregates the three final **test** results, not validation results.
All ten epochs execute; the restored model includes the selected epoch's
BatchNorm state. No checkpoint file or full-outer-training refit is produced.

Compare C against B using the same data, inner/outer target splits, seed and
other settings. Verify that each C epoch records training mode, while B records
training mode only in epoch one. The shared top-level seed does not promise
identical batch orders or later-fold initial weights: C's additional dropout
consumes random numbers, and the runner seeds once before CV. Report any changes
across AUC, precision, recall and F1 without assuming they all improve.

A difference between A's test-selected result and B's result cannot establish a
model improvement: epoch selection and fitting sample counts differ. The current
graph context and preprocessing remain shared across target splits in B and C.
Each validation subset has only seven drivers; treat one three-fold experiment
as limited evidence rather than a guaranteed general performance gain.
