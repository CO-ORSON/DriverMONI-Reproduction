# Training protocol: source versus reproduction settings

Verified against `CV_multimodal_compare.py` at upstream revision
`ddb0a830158ba41c90019bd5ca92054c2ba0869b`, especially `test()` lines 349-387 and
`train_data()` lines 1149-1271. The paper alone was not used to infer settings.

| Setting | Actual authors' code | Clean default |
| --- | --- | --- |
| Loss | `nll_loss(log_softmax(logits, dim=1, dtype=float), labels, weight=weights)` | Same, explicit float64 log-softmax/weights and weighted mean reduction. |
| Class weights | `(1 / n_class) * (N_train / 2)` | Same, computed from each training fold only. |
| Optimizer | Adam with only `lr=0.001` supplied | Adam, same LR and explicit library defaults. |
| Adam defaults | Betas `(0.9,0.999)`, epsilon `1e-8`, AMSGrad false | Same, visible in YAML. |
| Weight decay | Not passed; Adam's default is 0 | `weight_decay: 0.0`. |
| LR schedule | No scheduler or LR updates | Constant LR; `scheduler: null`. |
| Epochs | 10 | 10. |
| Batch size | 32 | 32. |
| DataLoaders | Train/test shuffled; workers 0; drop_last false | Same. |
| Early stopping | None; all epochs run | `early_stopping: null`. |
| CV | `StratifiedKFold(n_splits=3, shuffle=True)` | Same plus `random_state=seed`. |
| Seed | No seed/random_state specified | 42, a new configurable reproduction default. |
| Epoch selection | Highest test-fold AUC, strict improvement | Same; earliest epoch wins ties. |
| Threshold | Driver probability >= 0.5 | Same. |
| Metrics | ROC curve area and balanced accuracy | Both retained, plus accuracy/precision/recall/F1. |
| Model mode | First epoch train; `test()` sets eval and train mode is never restored | Preserved by default with an explicit switch. |

The supplied main paper reports Adam 0.001, class weighting, and three-fold
stratified CV. It does not establish this source's ten-epoch/batch-size settings,
test-AUC epoch selection, or persistent evaluation-mode behavior. Do not describe
the latter as paper-specified methodology.

## Added settings, explicitly declared

- `seed: 42`: seeds Python `random`, NumPy, PyTorch, CUDA where available, and DGL.
  Seed once before CV, not per fold or epoch. The splitter gets its own explicit
  random_state. The authors did not specify a seed.
- `deterministic_algorithms: true`: enables supported deterministic PyTorch
  behavior, fixes cuDNN settings, and sets the cuBLAS workspace configuration.
  Unsupported deterministic operations raise rather than silently switching.
  CPU tests do not establish exact cross-platform/CUDA reproducibility.
- `cpu_threads: 1`: an execution/reproducibility choice. The authors did not set
  a thread count; null leaves the runtime default unchanged.
- `device: auto`: CUDA when available, otherwise CPU. Explicit CUDA requests
  fail clearly if either PyTorch or DGL cannot use it. Upstream hard-codes CUDA.
- `zero_division: 0`: undefined precision/recall/F1 cases return zero. These added
  metrics and this convention were not specified by the source.
- `std_ddof: 0`: population standard deviation across the three fold scores.
  Upstream saves individual fold values without computing mean/std. This is not
  a confidence interval; set 1 explicitly for sample standard deviation.

## Source-compatible per-fold execution

This section applies to `best_test_roc_auc`, used by the original-data and
prepared-data reproduction configurations. B is specified separately below.

1. Split target genes with stratification. Neighborhood features retain the full
   filtered context table; labels are not node features.
2. Print class counts, derive training-fold weights, instantiate a fresh model
   and Adam optimizer, and construct train/test loaders.
3. Run all configured epochs with weighted NLL. AUC always uses
   `softmax(logits, dim=1)[:,1]`, never thresholded labels.
4. Preserve eval mode after the first evaluation by default: dropout is disabled
   and BN running statistics stop updating, while gradient/optimizer updates
   continue. `preserve_eval_mode_after_first_evaluation: false` explicitly chooses
   a corrected protocol, recorded as a departure from the source baseline.
5. Select the best held-out AUC epoch and report all metrics from that epoch.
   No independent selection by F1/accuracy or extra validation split is introduced.
6. Persist fold assignments, epoch history, selected predictions, counts, weights,
   and settings. After three folds, calculate unweighted means/std and mark the
   JSON complete. Failed runs never masquerade as completed experiments.

The first evaluated epoch initializes the best result, avoiding the authors'
uninitialized-result edge case if all AUC values are exactly zero. For positive
AUC, selection matches the original. Singleton training batches are rejected
rather than silently dropped, because source BatchNorm cannot train on them.
The default BRCA folds do not produce singleton training batches.

## B: validation-based epoch selection

`configs/brca_validation.yaml` implements suggestion 2 only. It uses the same
prepared data and outer folds as `brca_updated_data.yaml`; it does not change
the DNN/GAT architecture, features, labels or graph construction.

1. Create the same three shuffled stratified outer folds with `random_state=seed`.
2. Split each outer training fold with stratified
   `train_test_split(test_size=0.25, random_state=seed)`. These inner fitting and
   validation subsets are disjoint from each other and the outer test targets.
   The fraction is a new project choice, not a setting supplied by the authors.
3. Derive class weights **only from the fitting subset**, using
   `N_fit / (2 * n_fit_class)`, and initialize a fresh model and Adam optimizer.
4. Run all ten epochs with the existing batch size, optimizer, loss and learning
   rate. Evaluate the validation subset after each epoch. Preserve the source
   behavior: epoch one optimizes in training mode; subsequent epochs optimize in
   evaluation mode with dropout disabled and BatchNorm statistics frozen.
   Validation loading is unshuffled; train/test loader settings remain unchanged.
5. Keep a copy of the complete model state at each strict improvement in
   validation AUC, including BatchNorm buffers. Select the earliest maximum.
   Test predictions and test metrics are not calculated during epoch selection.
6. After all ten epochs, restore the selected state and evaluate the outer test
   fold once. There is no early stopping and no refit on the full outer training
   fold. Final fold metrics and their aggregate describe these test predictions.
7. Record the outer training, fitting, validation and test assignments; class
   counts and fitting-only weights; validation history; selected validation
   predictions; and final test metrics/predictions separately.

For the current prepared dataset and seed 42, each fitting subset has 21 drivers,
each validation subset has 7, and each test fold has 14. Validation selection can
therefore be unstable; the fold standard deviation is not a confidence interval.
The reduction in fitting samples is part of this protocol change.

This removes use of the outer test labels for epoch selection. It does not
rebuild preprocessing within each fold, remove shared graph-feature context, or
establish a fully inductive setting. Original normalized features remain supplied
as input. Keep these limits when interpreting the resulting scores.

Use B as the evaluation reference for a future C experiment that restores
training mode each epoch. B and C should share the exact inputs, outer/inner
splits, seeds and other settings. The existing test-selected A score is not an
unbiased comparator for claiming performance gains. See the
[B execution guide](brca_validation_experiment.md).

## Verification and limits

Tests check score-based AUC, inclusive thresholding, weighted NLL against its
formula, parameter updates in eval mode with frozen BN statistics, class counts,
disjoint target splits, three-fold coverage, best-epoch selection, constant LR,
metric aggregation, strict JSON output, and failure status. Repeating the small
CPU experiment with the same seed gives identical splits, per-epoch losses,
selected epochs, probabilities, and summary metrics.

Architecture equivalence was separately tested with transferred upstream weights.
Clean deterministic graph numbering and removal of dead modules mean a common
seed is not proof of identical upstream initial weights or paper scores. The
authors' original dependency versions and complete publication experiment
provenance are not provided by their repository.
