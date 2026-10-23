# C: BRCA validation selection with training mode restored each epoch

C combines validation-based epoch selection with restoring `model.train()` at
the start of every training epoch. It is a separate experimental protocol. The
DNN, custom sigmoid-attention GAT, fusion classifier, data and graph construction
are unchanged. It does not replace the original reproduction or establish a
performance improvement by itself.

## What differs from B

| Setting | B | C |
| --- | --- | --- |
| Configuration | `configs/brca_validation.yaml` | `configs/brca_validation_train_mode.yaml` |
| Epoch selection | Inner validation AUC | Same |
| `training.preserve_eval_mode_after_first_evaluation` | `true` | `false` |
| Mode during parameter updates | Training in epoch 1; evaluation in epochs 2–10 | Training in all ten epochs |
| Dropout during updates | Active in epoch 1 only | Active in every epoch |
| BatchNorm running statistics | Updated in epoch 1 only | Updated in every epoch |
| Final outer-test evaluation | Once, after restoring the validation-selected state | Same |
| Output | `results/brca_validation_metrics.json` | `results/brca_validation_train_mode_metrics.json` |

These two configuration files differ only in the training-mode switch and output
path. The source-compatible configurations, `configs/brca.yaml` and
`configs/brca_updated_data.yaml` (A), remain available. C uses a training switch
already supported by the runner; it does not add a new architecture or optimizer.

The authors' loop calls `eval()` for evaluation and does not restore `train()`
before subsequent training epochs. B preserves that behavior for source parity.
C explicitly changes it so dropout and BatchNorm use their training behavior
whenever parameter updates are performed. Validation and final test inference
still use evaluation mode, with dropout disabled and BatchNorm statistics frozen.

## Inputs and environment

Use the Python 3.12 environment from the [main README](../README.md#environment-setup)
and the same files as B:

```text
data/
├── brca/BRCA_input_data.csv
└── upstream/
    ├── using_Data/BRCA_label.csv
    └── for_ref/humannet_v2_and_v3.zip
```

Use the uploaded CSV without rerunning preprocessing. The
[prepared-data guide](brca_updated_data_experiment.md#place-the-files) provides
download links, exact placement and expected checksums. ZIP extraction is not
required. C keeps all 16,786 target genes and the original 42 labelled drivers.
The other 16,744 genes receive benchmark class 0; they are not established
biological non-drivers.

## Run from the project root

```bash
python scripts/run_brca.py --config configs/brca_validation_train_mode.yaml --device cpu
```

The report is saved to `results/brca_validation_train_mode_metrics.json`, leaving
B's output separate. A repeated C run overwrites C's report. To keep another run:

```bash
python scripts/run_brca.py --config configs/brca_validation_train_mode.yaml --device cpu --output results/brca_validation_train_mode_my_run.json
```

The configuration's data base is `data`, containing both `brca/` and `upstream/`.
An optional `--data-root` must point to that combined directory, not an authors-only
checkout. YAML data/output paths are project-relative; CLI paths are relative to
the working directory. See [configuration rules](configuration.md).

### PyCharm

Create a **Python** run configuration:

- Interpreter: Python 3.12 with the project requirements installed.
- Script: this project's `scripts/run_brca.py`.
- Parameters: `--config configs/brca_validation_train_mode.yaml --device cpu`.
- Working directory: this reproduction project's root.

The runner selects DGL's PyTorch backend. CPU is the established environment.
A compatible PyTorch/DGL CUDA installation can use the YAML's `auto` device
selection by omitting `--device cpu`; cross-device numerical identity is not
guaranteed.

## Protocol and comparison rules

1. Use the same input snapshot and seed 42 as B. Outer splits are
   `StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)`.
2. Reserve the same stratified 25% of each outer training fold for validation.
   This fraction is a project choice. Class weights use only the remaining
   fitting targets: `N_fit / (2 * n_fit_class)`.
3. Train all ten epochs, calling `model.train()` before each one. Retain Adam
   at 0.001, zero weight decay, batch size 32, weighted float64 NLL, the existing
   shuffle settings and all architecture values. There is no scheduler or early
   stopping.
4. Evaluate validation AUC in evaluation mode after each epoch. Keep the complete
   model state, including BatchNorm buffers, at each strict AUC improvement;
   the earliest maximum wins ties.
5. Restore the selected state and evaluate the outer test fold once. Do not
   select a second epoch or threshold using the test results. There is no refit
   on the full outer training fold and no checkpoint file is written.
6. Compare C against B's final outer-test metrics on the same target partitions.
   Report AUC, precision, recall and F1 together, including per-fold results and
   mean ± population standard deviation. Retain unchanged B results even if C
   is worse on a metric.

Expected target counts for both B and C on the current data and seed:

| Fold | Fitting drivers / class 0 | Validation drivers / class 0 | Test drivers / class 0 |
| --- | --- | --- | --- |
| 1 | 21 / 8,371 | 7 / 2,791 | 14 / 5,582 |
| 2 | 21 / 8,372 | 7 / 2,791 | 14 / 5,581 |
| 3 | 21 / 8,372 | 7 / 2,791 | 14 / 5,581 |

The same top-level seed reproduces the target splits. It does **not** guarantee
identical batch orders or per-fold initial model weights between B and C: dropout
in later C epochs consumes random numbers that B does not, and the existing
runner seeds once before CV. BatchNorm statistics also evolve differently.
This experiment keeps the existing random-number handling rather than introducing
per-fold or separate-loader generators solely for C. Interpret any result as the
effect of the configured training procedure, not a perfectly matched sequence
of stochastic updates. Repeated paired seeds would strengthen a later comparison.

## Interpretation and limitations

The model architecture and data do not become new because the mode changes.
C tests whether training-mode dropout and BatchNorm throughout optimization help
under B's independent epoch-selection procedure. Higher AUC, precision, recall
or F1 is not guaranteed, and one metric can improve while another declines.
The switch changes dropout and BatchNorm together; this experiment does not
isolate their individual contributions.
A's test-selected score is not an unbiased comparator for a performance claim;
use B as C's primary reference.

Only seven labelled drivers are available in each validation subset and fourteen
in each test fold. Selected epochs and scores can therefore be sensitive to the
split; three-fold standard deviation is not a confidence interval. Precomputed
normalized features and full graph-feature context remain shared across target
splits. C does not establish fully inductive evaluation, raw preprocessing parity
or independent biological validation.

The JSON must have `status: complete` before reporting an experiment. Its schema
matches B: validation histories and selected validation predictions are separate
from final test predictions and metrics. Check the saved
`preserve_eval_mode_after_first_evaluation` flag and the training-mode value in
each epoch record (`training_mode_during_updates: true`) when verifying C.

## Completed local experiment

Completed **23 October 2026 at 11:36:50 Singapore time**, in **15.2 minutes**
(911.65 seconds) on macOS arm64 CPU. The saved report has `status: complete`;
all three folds and all 30 training epochs completed with seed 42. The runtime
matches B: Python 3.12.14, PyTorch 2.2.2, DGL 2.2.0, NumPy 1.26.4,
pandas 3.0.6 and scikit-learn 1.9.1, with one CPU thread.

| Fold | Selected epoch | Selected validation AUC | Test AUC | Test accuracy | Test precision | Test recall | Test F1 | Test balanced accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 5 | 0.800788 | 0.866267 | 0.993924 | 0.083333 | 0.142857 | 0.105263 | 0.569458 |
| 2 | 2 | 0.914163 | 0.949996 | 0.980340 | 0.093220 | 0.785714 | 0.166667 | 0.883271 |
| 3 | 2 | 0.932436 | 0.900863 | 0.982663 | 0.022989 | 0.142857 | 0.039604 | 0.563813 |

The comparison below uses the existing, unchanged B report and C's final
outer-test results. Values are **mean ± population standard deviation** across
the three outer test folds, using the same input snapshot and target assignments.

| Test metric | B: validation selection, source mode behavior | C: validation selection, training mode every epoch |
| --- | --- | --- |
| ROC-AUC | 0.918824 ± 0.016343 | 0.905708 ± 0.034354 |
| Accuracy | 0.990409 ± 0.003005 | 0.985642 ± 0.005933 |
| Precision | 0.095931 ± 0.028804 | 0.066514 ± 0.031041 |
| Recall | 0.333333 ± 0.134687 | 0.357143 ± 0.303046 |
| F1 | 0.145655 ± 0.046222 | 0.103845 ± 0.051883 |
| Balanced accuracy | 0.662695 ± 0.066192 | 0.672181 ± 0.149281 |

**C did not improve mean AUC or F1 in this run.** Its slightly higher mean recall
is dominated by fold 2: C finds 2, 11 and 2 of the 14 test drivers across the
three folds, whereas B finds 6, 6 and 2. C's recall varies much more between
folds, and its mean precision is lower. This is evidence about this single-seed
experiment, not a statistically general conclusion that either training procedure
is superior. The joint dropout/BatchNorm change and different random-number
consumption do not isolate a cause for the score differences. Retain both
experiments and report this result without claiming a predictive improvement.

Verification completed:

- Full suite: **137 passed, 2 skipped** in 25.93 seconds. The skipped checks are
  CUDA and optional cached-source preprocessing replay.
- The B report is unchanged. The only configuration differences are C's mode
  switch and separate output path; no production Python code was changed for C.
- Runtime versions and the consumed input-file manifests match B, with file
  hashes verified. The preprocessing inputs and existing model remain unchanged.
- All outer and inner target assignments were reconstructed and match B exactly.
  Fitting, validation and test subsets are disjoint within each fold; all 16,786
  targets occur in an outer test fold exactly once.
- Class counts and fitting-only weights match B. All 30 C epochs have finite
  losses and scores, use training mode during updates and retain learning rate
  0.001. The first fold's first-epoch loss and validation metrics match B exactly.
- Each selected epoch is the earliest maximum validation AUC. All six final
  test and selected-validation metrics were recalculated from saved predictions,
  along with summary means and population standard deviations.
- Tests with real model hooks verify Dropout and BatchNorm modes, evolving
  BatchNorm statistics during C training, and restoration of the complete
  validation-selected state before final test evaluation.

Run the suite from the project root, supplying the full external authors' checkout:

```bash
python -m pytest tests -q --brca-data-root=/path/to/original/DriverMONI
```

The standard runner produces `results/brca_validation_train_mode_metrics.json`.
This review additionally produced:

- `results/brca_validation_train_mode_run.log`
- `results/brca_validation_train_mode_tests.log`
- `results/brca_validation_train_mode_verification.json`
- `results/brca_b_vs_c_comparison.json`

The verification receipt and comparison JSON were generated during review; the
standard runner does not create them. All generated results and logs are ignored
by Git. The tables in this guide are the shareable completion record. No model
checkpoint was retained, and no independent full C replay, raw-source replay or
clinical validation was performed.
