# Configuration and path reference

`configs/brca.yaml` contains every supported setting. Unknown keys, missing keys,
unsupported model/graph settings, and invalid values fail before data loading.
YAML numbers must be numbers, not quoted strings; booleans are `true`/`false`.

## Paths and entry points

- `--config`: a YAML path relative to the **current shell directory**, or absolute.
  Omit it to use the project's `configs/brca.yaml`, regardless of shell directory.
- YAML `data.upstream_root` and `output_path`: relative to the **project root**,
  not the directory containing a custom YAML file. Absolute paths and `~` work.
- YAML per-file data paths: relative to `data.upstream_root`; an absolute or
  `~`-expanded path overrides that root for that file.
- CLI `--data-root` and `--output`: relative to the **current shell directory**,
  or absolute. `~` also expands, including when quoted. They override YAML paths.
- The full runner defaults to `results/brca_metrics.json`; each run replaces that
  output file. Use a different `--output` to preserve an earlier run.
- All three scripts support direct execution from another directory by supplying
  their paths. `python -m scripts.debug_brca` assumes you are at the project root.
  No editable install or `PYTHONPATH` configuration is needed.

## Top-level and reproducibility settings

| Key | Default / supported values | Meaning |
| --- | --- | --- |
| `cancer` | `BRCA` only | Selects the only implemented cancer architecture. |
| `upstream.repository`, `upstream.commit` | Pinned authors' URL and `ddb0a830158ba41c90019bd5ca92054c2ba0869b` | Fixed reference provenance, not a command to download data or switch model versions. |
| `seed` | `42`; integer `0 <= seed < 2**31` | New reproduction choice. DGL's signed C integer requires this range. Seeds Python/NumPy/PyTorch/CUDA/DGL once before CV. |
| `device` | `auto`, `cpu`, `cuda`, `cuda:N` | Auto chooses CUDA when PyTorch exposes it, otherwise CPU; MPS is unsupported. DGL must also support the selected CUDA device. |
| `reproducibility.deterministic_algorithms` | `true` | New setting; uses PyTorch determinism checks, deterministic cuDNN, and a supported cuBLAS workspace config. Unsupported operations fail rather than silently use nondeterminism. |
| `reproducibility.cpu_threads` | `1`; positive integer or `null` | New setting; null leaves the runtime thread count unchanged. |
| `output_path` | `results/brca_metrics.json` | Saved metrics, predictions, config, provenance and status. No checkpoints are written. |

DGL's PyTorch backend is selected before import; the package does not need to write
`~/.dgl/config.json`. An explicitly incompatible `DGLBACKEND` environment variable
is rejected with an explanation. Set it to `pytorch` or unset it. For deterministic
CUDA execution, an existing `CUBLAS_WORKSPACE_CONFIG` must be `:4096:8` or `:16:8`;
if absent the runner sets `:4096:8`. CPU is the validated baseline. Equal seeds do
not promise identical results across CPU/GPU, package versions, or hardware.

## Data settings

| Key below `data` | Meaning |
| --- | --- |
| `upstream_root` | Directory containing the original processed/reference file layout. |
| `features` | BRCA feature CSV with gene identifiers and ordered model columns. |
| `labels` | Driver Ensembl IDs in `gene_label`. |
| `comparison_genes` | Original comparison list (`gene` column). |
| `gene_annotations` | Symbol whitelist (`Symbol` column). |
| `ppi_edges` | CSV with `entrez_id_1`/`entrez_id_2` containing Ensembl IDs. If missing, look for `humannet_v2_and_v3.zip` in the same directory and read its `humannet_v2_and_v3/humannetv3_ens_ppi.csv` member. |
| `neighbors` | Supplied dictionary of target-gene IDs to neighbor lists. |
| `use_comparison_filter` | `true`; false skips the comparison/annotation files and changes the target population. |
| `use_supplied_neighbors` | `true`; false skips the cache and reconstructs undirected one-hop neighbor sets, changing anomalous cached inputs. |

Missing required files are reported together with resolved paths and acquisition
instructions. See [data preparation](../data/README.md) for download links and
feature order. Disabling an option does not require its unused input files to exist.

## Architecture and graph settings (fixed assertions)

The `model` and `graph` blocks document the implemented science. They are validated
against the following definitions, **not** used as an architecture search API.

| Key | Required value |
| --- | --- |
| `model.dnn.input_dim`, `hidden_dims`, `output_dim`, `dropout` | `12`, `[256,128]`, `64`, `[0.7,0.6]` |
| `model.gat.input_dim`, `attention_dim`, `output_dim`, `attention_activation` | `13`, `64`, `32`, `sigmoid` |
| `model.fusion.extra_gat_projection_dim`, `concatenated_dim`, `num_classes` | `32`, `96`, `2` |
| `graph.hops`, `neighbor_discovery` | `1`, `undirected` (for fallback neighborhood discovery) |
| `graph.message_edges`, `self_loops`, `edge_threshold` | `csv_direction`, `true`, `null` |

## Training and CV

| Key | Default / supported values | Meaning |
| --- | --- | --- |
| `training.optimizer` | `adam` only | Source optimizer. |
| `training.learning_rate` | `0.001`; positive finite number | Constant learning rate. |
| `training.weight_decay` | `0.0`; nonnegative finite number | Original Adam default made explicit. |
| `training.betas` | `[0.9,0.999]`; two numbers in `[0,1)` | Original Adam defaults. |
| `training.eps` | `1.0e-8`; positive finite number | Original Adam default. |
| `training.amsgrad` | `false` | Original Adam default. |
| `training.scheduler`, `early_stopping` | `null` only | Neither is implemented or used upstream. |
| `training.epochs` | `10`; integer >=1 | All epochs execute. |
| `training.batch_size` | `32`; integer >=2 | Reject folds with a final singleton train batch because of BatchNorm; do not silently drop it. |
| `training.loss`, `class_weights` | `weighted_nll`, `inverse_training_fold_frequency` only | Float64 log-softmax, weighted mean NLL; `N_train/(2*n_class)`. |
| `training.shuffle_train`, `shuffle_test` | `true`, `true` | Match upstream loader shuffle; configurable booleans. |
| `training.num_workers`, `drop_last` | `0`, `false` only | Preserve source batching and single-process loading. |
| `training.preserve_eval_mode_after_first_evaluation` | `true` | Preserve source dropout/BN mode behavior; false explicitly changes the protocol. |
| `cross_validation.n_splits`, `stratified`, `shuffle` | `3`, `true`, `true` only | Stratified shuffled target splits. Uses top-level seed as new explicit random_state. |
| `cross_validation.epoch_selection` | `best_test_roc_auc` only | Maximum test AUC, earliest epoch wins ties; not untouched-test evaluation. |

Changing optimizer values, epochs, batch size or shuffle is permitted for explicit
experiments but departs from the documented baseline. Effective settings are saved.

## Evaluation

| Key below `evaluation` | Default / supported values | Meaning |
| --- | --- | --- |
| `metrics` | All six names, exactly once: `auc`, `accuracy`, `precision`, `recall`, `f1`, `balanced_accuracy` | All are computed; list order does not control report order. |
| `positive_class` | `1` only | Driver label. |
| `probability_threshold` | `0.5`; finite number in `[0,1]` | Positive prediction when score >= threshold; does not affect AUC. |
| `zero_division` | `0`; integer 0 or 1 | Result for undefined precision/recall/F1. New reporting choice. |
| `std_ddof` | `0`; integer 0 or 1 | Population or sample fold standard deviation. New reporting choice, not a confidence interval. |

The smoke script is intentionally fixed to CPU, 12 targets, two epochs and batch
size 4 using the default file layout. It is a pipeline check; use the full runner
with `--config` for experiments with alternative settings or input filenames.
