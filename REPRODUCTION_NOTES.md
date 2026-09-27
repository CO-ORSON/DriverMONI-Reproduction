# DriverMONI BRCA reproduction audit

Audit date: 2026-09-27. Reference: authors' repository at commit
[`ddb0a830158ba41c90019bd5ca92054c2ba0869b`](https://github.com/BML-cbnu/DriverMONI/tree/ddb0a830158ba41c90019bd5ca92054c2ba0869b).
The original checkout was inspected without modification. Scope: BRCA, supplied
processed inputs, active `train_data()` path and `gcn_linc_embedder_transfer`.
Unused alternative models in the same file are not the reproduction target.

Subsequent [fresh-teammate review](docs/teammate_review.md) fixed entry-point,
backend, path, config-validation, and seed-range issues without changing the
scientific model. Its fresh-environment full suite passed 78 tests with one CUDA
skip; the original parity evidence below remains applicable.

**Conclusion:** the tested forward computation, loss, gradients, graph semantics,
and one Adam update agree under controlled inputs and transferred active weights.
This does **not** establish an identical original training trajectory or reproduce
the paper's reported performance. Exact claims below are limited to the stated
operations and fixtures, on the tested CPU environment.

## 1. Reproduced exactly

The following architecture and training definitions match the inspected source.
For graph data, “exact” means matching genes, directed edge multiplicities,
features, and labels after aligning gene IDs; it does not mean identical numeric
node numbering. Numerical testing covers the cases described below, not every
possible input or hardware platform.

| Requested comparison | Authors' active implementation | Clean implementation and evidence |
| --- | --- | --- |
| 1. DNN dimensions | `torch_models_linear.py:16`, `feat_L_BRCA`: 12→256→128→64 | `src/models/dnn.py::DNNEncoder`; source parity and intermediate traces. |
| 2. Activation ordering | First two blocks: Linear→BatchNorm→ReLU→Dropout. Third: Linear→BatchNorm→ReLU. | Same sequence; hooks compare execution order, inputs, and outputs of every active module. |
| 3. Dropout | First block 0.7, second 0.6; no third dropout. | Same; train-mode comparisons reset the dropout RNG before each forward. |
| 4. GAT dimensions | `CV_multimodal_compare.py:65,122`, `GATLayer`/`GAT_L`: bias-free 13→64, target selection, BN64→ReLU→Linear64→32→BN32→ReLU. | `src/models/gat.py::GATLayer/GATEncoder`; node output `[N,64]`, target output `[B,32]`. |
| 5. Attention calculation | `z=W x`; concatenate **source then destination**; bias-free Linear128→1; LeakyReLU default slope 0.01. Aggregate projected source messages at destinations. | Same shared projection for attention inputs and messages; intermediate projection, raw edge score, activated edge score, and aggregate comparisons. |
| 6. Sigmoid | `reduce_func`: `F.sigmoid(e)`, then sum weighted incoming messages. No neighbor normalization. | `torch.sigmoid(e)` and sum; exact coefficients on the fixture, plus a hand-computed example that distinguishes sigmoid from softmax. |
| 7. Target extraction | `GAT_L.forward`: after graph message passing, select `(ndata['id']==1)`; then BN/ReLU/FC/BN/ReLU. | Select `is_target==1` after attention. Tests place targets at different local indices and compare the selected BN input `[B,64]`. |
| 8. Graph construction | `classify_graph_data_encoder_humannet`, lines 389–513: induced graph on target plus cached feature-bearing neighbors; one synthetic self-loop per global node; append each CSV edge only in its supplied direction. | `src/graph_utils.py::PPIGraph`, `src/dataset.py::DriverMONIDataset`; retain neighbor-neighbor edges, duplicate edges, and additional CSV self-loops. No reverse insertion or LLS threshold in this path. |
| 9. Feature order | Explicit column lists in the original dataset class, not inferred from CSV order. | Constants below match AST-extracted source lists. Actual feature rows match after gene alignment. |
| 10. Class weighting | `train_data`, lines 1179–1185: `w_c=(1/n_c)*(N_train/2)`, ordered non-driver then driver, float64. | `src/train.py::class_weights`; executed the original weight expressions on identical labels and obtained exact values. No downsampling. |
| 11. Optimizer | `train_data:1198`: `optim.Adam(parameters, lr=0.001)`. | Adam; default betas `(0.9,0.999)`, eps `1e-8`, weight decay 0, AMSGrad false made explicit. Inspected source expression agrees with config in the tested runtime. Active parameters and Adam moments match after one controlled update within `rtol=1e-5, atol=1e-6`. |
| 12. Loss | `train_data:1224`: weighted mean NLL of `log_softmax(logits, dim=1, dtype=float)`; here `float` means torch.float64. Label is column 1 of original one-hot labels. | `weighted_nll`, float64 log-softmax and weights, integer binary labels. Original expression and clean function return exactly equal values on the fixture. |
| 13. CV split procedure | `train_data:1149`: `StratifiedKFold(n_splits=3, shuffle=True)`, **no random_state**. | Same split algorithm; explicit `random_state=seed` is an intentional difference, detailed in section 3. Original and clean fold indices match when the original NumPy RNG is deliberately initialized to the same seed. Historical folds are unknown. |

The active fusion class (`CV_multimodal_compare.py:290`) adds
**Linear32→32→BN32→ReLU** to the GAT output before concatenation. The fused vector
is `[DNN64, GAT32]`, followed by Linear96→2 returning raw logits. This extra block
is present in `src/models/drivermoni.py::DriverMONI`; the simpler unused
`gcn_linc_embedder` is not the model instantiated by training.

Initialization rules are preserved: Xavier-uniform DNN linear weights, GAT
target FC weight, and extra fusion FC weight; default biases and default
initialization for attention projections and the classifier. DNN standalone
same-seed initialization matches exactly. Full-model same-seed initialization
does not, for the reason in section 3.

Exact ordered feature contract (indices 0–11, DNN):

```text
synonymous_variant
stop_gained
missense_variant
frameshift_variant
splice
inframe
lost_stop and start
deg
related_pathway
dir_pathway
muta_count
miss_ratio
```

GAT uses those same 12 columns followed by `PPI` at index 12. The constants assert
lengths 12 and 13. `deg` denotes differential expression. Supplied processed
features are converted to float32 without additional standardization. Label 1
means membership in `using_Data/BRCA_label.csv`'s `gene_label`; 0 means absent from
that list, not experimental proof that the gene is a non-driver.

Other preserved active settings: 10 epochs, batch size 32, train and test loader
shuffle enabled, workers 0, no dropping of the final batch, constant learning
rate, no scheduler, and no early stopping. A fresh model/Adam is created per fold.
All folds retain the full filtered feature context for neighbors; test genes may
appear as neighbors of training targets. Labels are never node features.

The source's `test()` switches to eval mode and training never restores train
mode. The default reproduction preserves this: after epoch one, dropout is off
and BatchNorm statistics freeze while gradient updates continue. It also selects
the maximum **test-fold** AUC across epochs, with strict improvement and earliest
tie. This is not an untouched-test-set evaluation protocol. AUC uses driver
probabilities, not hard predictions; original balanced accuracy uses threshold
`>=0.5`.

Verification command from this project root:

```bash
DGLBACKEND=pytorch .venv/bin/python -m pytest -q tests --brca-data-root /path/to/DriverMONI
```

Latest sharing-preparation verification: **59 passed, 1 skipped in 23.86 seconds**. The skipped test requires
CUDA, which is unavailable on this host. This run included all five real BRCA
sample comparisons and all seven actual-source audit cases. Original-source
comparisons now require a separately obtained checkout: verbatim bundled excerpts
were removed after the license review. Without source, all source-dependent parity
cases skip explicitly. Standalone synthetic checks still run.

Evidence files:

- `tests/test_source_audit.py`: checks source hashes and that in-memory extraction of four model classes and
  one dataset class applies only the documented compatibility adapters;
  reads feature lists and executes original weight/loss/optimizer/CV expressions.
- `tests/test_intermediate_parity.py`: shared three-graph input, eight nodes,
  train/eval modes, transferred parameters/BN state, identical dropout seed.
  All captured intermediate inputs/outputs, target embeddings, edge logits,
  sigmoid coefficients, fused vectors, and logits match exactly on CPU.
  One Adam update and optimizer moments match within the stated tolerance.
- `tests/test_dnn.py`, `tests/test_gat.py`, `tests/test_model.py`: exact forward
  outputs and losses under matched conditions; input and parameter gradients
  compared with PyTorch's default tolerance; independent attention math checks.
- `tests/test_dataset.py`: synthetic duplicate/directed edges and loops, isolated
  targets, full dataset-to-logits parity under node reindexing (`rtol=atol=1e-5`),
  and actual BRCA samples. The entire filtered feature table matches an independent
  transcription of source filtering: 15,028 targets, 42 positives.
- `tests/test_training.py`: controlled repeated three-fold experiments, exact
  seeded replay, class counts, mode transitions, best-epoch selection, and metrics.

Actual BRCA graph comparisons (matched feature rows, directed edge multisets,
target masks, DNN features, and labels):

| Gene | Nodes | Edges | Label |
| --- | ---: | ---: | ---: |
| TSPAN6 / ENSG00000000003 | 46 | 109 | 0 |
| BRCA1 / ENSG00000012048 | 491 | 3,398 | 1 |
| TP53 / ENSG00000141510 | 473 | 3,620 | 1 |
| MYCBP2 / ENSG00000005810 | 1 | 1 | 0 |
| RC3H2 / ENSG00000056586 | 1 | 1 | 0 |

## 2. Refactored but behaviorally equivalent

These changes preserve the tested computation for valid processed inputs under
matched weights/RNG state; they do not promise identical random training runs.

- Split the monolithic code into model, dataset, graph, training, and evaluation
  modules with type hints, shape checks, and meaningful target-gene names.
- Renamed `id`→`is_target`, `idx`→`global_node_id`, and unused edge flag
  `direct_con`→`is_target_edge`; one target is validated per batched component.
  Typed samples carry a scalar label equivalent to the original one-hot column 1.
- `torch.sigmoid` replaces deprecated `F.sigmoid`; custom attention is retained.
- Graph-local scopes avoid the original forward's persistent feature mutation.
  This preserves the tested single-forward computation and permits graph reuse;
  repeated calls on one already-mutated original graph are not claimed equivalent.
- Sorted gene IDs and returned DGL parent IDs replace set order and positional
  assumptions. Comparison is by gene identity; summation order can cause roundoff.
  Modern DGL constructors and cached index lookups preserve tested graph semantics.
- Removed unused layers that never participate in forward/backward. Active-weight
  transfer proves conditional forward/update equivalence; RNG effects are separate.
- Device moves are explicit; CPU works without `.cuda()`. The CPU reference oracle
  adapts only the fusion's hard-coded float32 CUDA conversion to the input device.
  The dataset oracle uses modern graph constructors and a cached unique-list lookup.
- The loader reads the supplied archive directly and uses the shipped `PPI_graph/`
  cache path. JSON output replaces the original broken output filename dependency.
  These repair execution plumbing without changing supplied tensor/graph contents.

The AST audit checks the reference adapters against pinned source. Tests read
selected definitions from the user-supplied checkout into memory using
`tests/upstream_reference.py`; no authors’ class bodies are shipped. The pinned
original files themselves remain untouched. See `THIRD_PARTY_NOTICES.md`.

## 3. Could not determine / differs from paper

| Item | Verified difference or unresolved evidence |
| --- | --- |
| Paper versus active fusion | The paper's overview combines DNN64 and GAT32; active source adds a learned 32→32 BN/ReLU block before concatenation. We retain it. |
| Attention projections | Paper equations 8–10 describe trainable `W` and `W′`; active source uses one shared projection for attention inputs and messages, with no independent `W′`. It also applies LeakyReLU before sigmoid. |
| Network filtering | Paper describes filtering weak network edges. `data_preprocessor.py` thresholds at LLS 3.211 while making the scalar `PPI` feature. The active CV graph constructor uses every row of the supplied HumanNet v3 CSV without this threshold. Scalar-feature construction must not be confused with message-passing graph construction. |
| Edge direction | Although neighborhood discovery uses an undirected NetworkX graph, the actual DGL graph adds each CSV row in one direction only. We retain this asymmetry and make no claim that the paper specifies it. |
| Feature formulas | Paper describes missense/synonymous; preprocessing uses missense/(missense+synonymous). Paper describes log2 expression fold change; code subtracts group means of supplied expression values. Equivalence depends on input transformation and cannot be assumed. These raw preprocessing operations were inspected, not reproduced here. |
| Epoch/mode selection | Main paper reports weighted NLL, Adam 0.001, and three-fold stratified CV. It does not establish this source's persistent eval mode, ten-epoch/batch-32 settings, or selection on test AUC. We document these as source behavior. |
| Random seed and historical folds | No original seed or splitter random_state is specified. Clean seed 42, explicit splitter RNG, deterministic settings, and CPU thread count are new reproducibility choices. Controlled split parity does not recover the authors' historical splits. Explicit splitter RNG also avoids consuming the global NumPy stream as the original does. |
| Dead-layer removal | Original unused layers consume initialization randomness, including before active Xavier initialization. Same full-model seed does not yield identical active weights. Numerical parity tests deliberately transfer all active parameters and BN state; this is not evidence of identical training from independent initialization. |
| Cached neighborhoods | The shipped cache contains empty entries despite available PPI neighbors, including RC3H2. Defaults preserve the supplied cache. Regenerating ideal one-hop neighbors changes some inputs; the publication's exact cache provenance is unavailable. |
| Original runtime | Repository lacks a complete pinned original environment. Tests use Python 3.12, PyTorch 2.2.2, DGL 2.2.0 on macOS arm64 CPU, with compatibility adapters. Historical DGL ordering and CUDA numerics are not established; the CUDA test is skipped here. Adam defaults are verified in this runtime. |
| Complete experiment parity | The clean full BRCA experiment previously completed, but the unmodified original full training script was not run side by side with identical folds, initialization, batches, and hardware. Its missing `string/` path, hard-coded CUDA, and undefined `graph_name_` prevent treating it as a directly runnable oracle here. No exact paper-score or full-trajectory claim is made. |
| Added reporting | Accuracy, precision, recall, F1, zero-division=0, and mean/population std (`ddof=0`) are reporting additions; original active evaluation calculates AUC and balanced accuracy. |
| Deliberate edge-case differences | Clean first-epoch fallback handles all-zero AUC instead of leaving the original best-prediction variable undefined. Invalid inputs fail clearly; singleton train batches are rejected. Neither changes the tested default BRCA case. |
| Scope | BRCA only; other cancer architectures, raw-data regeneration, hyperparameter grid search, ablations, and publication-wide results are not validated by this audit. |

Source SHA-256 fingerprints for the inspected checkout:

```text
CV_multimodal_compare.py  9ece040b35d708e67eeddd74689e19300206cf10a2a85d07f0ef71024606911f
torch_models_linear.py   a8ed2d98ec78ea20ff48e788a725555031d9a8f213255f047b4da5414f6e7e20
data_preprocessor.py     9c5db69d9465efd40354034ba459ea8c05e015dccb4772e9f85bf0b9434d3afa
```
