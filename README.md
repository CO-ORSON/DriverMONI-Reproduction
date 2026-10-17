# DriverMONI Reproduction

## Objective

Reproduce the authors’ **BRCA** model architecture, processed-data graph inputs,
training, three-fold cross-validation, and evaluation in a readable Python project.
This is a group-project reimplementation, not an official release. The baseline
uses supplied processed inputs; separate raw-input source preparation is
documented below. Other cancers and algorithm improvements are outside this stage.

## Original paper and repository

Jeongho Park and Kyuri Jo, “DriverMONI: Cancer Driver Gene Prediction With
Multimodal Deep Learning Integrating Multiomics Data and Condition-Specific
Network Information,” *IEEE Transactions on Computational Biology and
Bioinformatics*, vol. 23, no. 1, 2026.
[Paper DOI](https://doi.org/10.1109/TCBBIO.2025.3643887).

Original repository: [BML-cbnu/DriverMONI](https://github.com/BML-cbnu/DriverMONI).
Reproduction reference commit: `ddb0a830158ba41c90019bd5ca92054c2ba0869b`.
The active `train_data()` / `gcn_linc_embedder_transfer` implementation takes
precedence when it differs from the paper. See
[REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md) for the component-by-component audit.

No license or redistribution grant was found in the pinned original repository.
Original source bodies, author-supplied processed data, and the paper PDF are
**not bundled in this Git project**. Optional parity tests read a separately
obtained local source checkout. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for provenance and
licensing status; this project does not grant rights to upstream materials.

## Model architecture

```text
Target gene features ── DNN: 12 → 256 → 128 → 64 ──────────────┐
                                                              ├→ concat 96 → FC 96→2
PPI ego graph ── GAT: 13 → 64 → target → FC 64→32 → FC 32→32 ──┘
```

- DNN: Linear → BatchNorm → ReLU → Dropout in the first two blocks, with dropout
  0.7 and 0.6. The final 64-dimensional block has no dropout.
- GAT: bias-free node projection 13→64; concatenate source/destination embeddings;
  bias-free attention projection → LeakyReLU(0.01) → **sigmoid**. Sum weighted
  incoming messages without softmax normalization. Select the target after message
  passing, then BN → ReLU → Linear64→32 → BN → ReLU.
- Fusion: the active source has an additional Linear32→32 → BN → ReLU on the GAT
  branch. Concatenate DNN first, then GAT; Linear96→2 returns raw logits.

Each sample is one target gene. Its graph contains the target and feature-bearing
neighbors from the supplied cache, retaining induced neighbor-neighbor edges.
The graph preserves CSV edge direction, duplicates, and self-loops; reverse edges
are not added. The DNN uses 12 ordered features; GAT appends the scalar `PPI`
feature as number 13. See [data/README.md](data/README.md) for exact column order.

## Repository structure

```text
DriverMONI-Reproduction/
├── README.md
├── REPRODUCTION_NOTES.md       # Evidence and limits of source parity
├── THIRD_PARTY_NOTICES.md      # Upstream attribution and license findings
├── requirements.txt
├── requirements-lock-macos-arm64.txt
├── .gitignore
├── pytest.ini
├── configs/
│   ├── brca.yaml                      # Baseline training settings
│   ├── brca_updated_data.yaml         # Separate experiment using data/brca CSV
│   ├── brca_preprocessing.json        # Combined canonical dataset workflow
│   ├── brca_omics_sources.json        # Mutation/expression sources and policies
│   ├── brca_sources.json              # KEGG retrieval and HGNC snapshot
│   ├── brca_driver_evidence.json      # Driver-evidence sources and output settings
│   └── brca_reviewed_driver_assertions.json # Reviewed literature assertions
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dnn.py
│   │   ├── gat.py
│   │   └── drivermoni.py
│   ├── dataset.py             # Features, labels, samples, batching
│   ├── feature_schema.py      # Shared feature order for preprocessing and loading
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── acquisition.py          # Download cache and source locks
│   │   ├── common.py               # Shared artifact/provenance utilities
│   │   ├── brca.py                 # Combined raw-to-feature workflow
│   │   ├── brca_omics.py           # Mutation, expression and scalar PPI features
│   │   ├── brca_sources.py         # KEGG/HGNC retrieval and label coverage audit
│   │   └── brca_driver_evidence.py # Linked source assertions and gene status
│   ├── graph_utils.py         # PPI loading and ego graphs
│   ├── train.py               # Training and three-fold CV
│   ├── evaluate.py            # Scores and metrics
│   └── utils.py               # Seeds, config, devices, provenance
├── scripts/
│   ├── run_brca.py                    # Full benchmark experiment
│   ├── debug_brca.py                  # Inspect graph inputs
│   ├── smoke_test.py                  # Small real-data pipeline check
│   ├── preprocess_brca.py             # Combined canonical BRCA export
│   ├── preprocess_brca_omics.py       # Optional omics diagnostics
│   ├── preprocess_brca_sources.py     # Optional pathway/label diagnostics
│   └── prepare_brca_driver_evidence.py # Build linked evidence/status tables
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── upstream_reference.py  # Loads external source in memory; no bundled excerpts
│   ├── test_dnn.py
│   ├── test_gat.py
│   ├── test_model.py
│   ├── test_dataset.py
│   ├── test_training.py
│   ├── test_portability.py
│   ├── test_intermediate_parity.py
│   ├── test_source_audit.py
│   ├── test_brca_omics.py
│   ├── test_brca_sources.py
│   ├── test_brca_pipeline.py
│   ├── test_brca_snapshot.py        # Optional cached-source replay
│   └── test_brca_driver_evidence.py
├── docs/
│   ├── graph_inputs.md
│   ├── training_protocol.md
│   ├── configuration.md
│   ├── teammate_review.md
│   ├── data_sources.md             # Source inventory and provenance limits
│   ├── brca_source_preparation.md  # Acquisition, formulas and validation
│   └── brca_driver_evidence.md     # Evidence rules, qualifications and replay
├── notebooks/
│   └── brca_driver_evidence_audit.ipynb # Source reconciliation and replay audit
├── data/
│   ├── README.md                  # Workflows, feature purpose and comparisons
│   ├── brca/
│   │   ├── BRCA_input_data.csv    # Canonical 13-feature gene dataset
│   │   ├── manifest.json          # Sources, policies and artifact checksums
│   │   └── summary.json           # Gene/sample coverage and exclusions
│   └── brca_driver_evidence/
│       ├── brca_driver_evidence.csv # Source assertions linked by evidence ID
│       ├── brca_gene_status.csv     # Per-gene evidence status and benchmark labels
│       ├── manifest.json           # Evidence-source provenance and checksums
│       └── summary.json            # Evidence coverage and review counts
└── results/
    └── README.md              # Output schema and interpretation
```

The canonical feature CSV, manifest and summary under `data/brca/`, plus the
driver-evidence tables, manifest and summary under `data/brca_driver_evidence/`,
are eligible for Git tracking.

## Environment setup

Validated: **Python 3.12, macOS arm64 CPU, PyTorch 2.2.2, DGL 2.2.0**. Direct
versions are pinned in `requirements.txt`; the macOS lock file records the full
original validated environment. The pinned macOS arm64 SciPy wheel requires
**macOS 14 or later**. Other OS/architecture combinations have not been tested.


```bash
git clone https://github.com/CO-ORSON/DriverMONI-Reproduction.git
cd DriverMONI-Reproduction
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

A fresh Python 3.12 environment was installed successfully from these requirements
during the teammate review; no preinstalled project packages were needed. For the
original baseline's full transitive dependency snapshot on macOS arm64, use
`python -m pip install -r requirements-lock-macos-arm64.txt` instead. Direct-only
pins allow transitive packages to evolve; use the lock when comparing environments.

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. Wheel
availability must be checked on that platform. DGL, PyTorch, and torchdata are
coupled dependencies; do not independently upgrade them. If a required DGL wheel
is unavailable, use a supported environment or the
[official DGL installation selector](https://www.dgl.ai/pages/start.html) and
rerun validation. The macOS lock file is not a CUDA installation recipe.
CUDA needs compatible CUDA-enabled PyTorch **and** DGL builds; GPU execution is
not validated here. `device: auto` uses CUDA when available, otherwise CPU;
`--device cpu` forces the validated execution path. MPS is not selected.

## Data preparation

Run `python3 scripts/preprocess_brca.py` to acquire pinned mutation/expression
inputs, KEGG/HGNC memberships and the authors' mapped HumanNet v3 network.
It writes one complete feature table, `data/brca/BRCA_input_data.csv`, plus
compact audits and a checksum manifest, using only Python's standard library.
All commands share `data/cache/`; replay tests use temporary directories.
See [BRCA source preparation](docs/brca_source_preparation.md) for source limits,
offline verification and optional modality diagnostics. The default training
configuration uses the baseline inputs below. The separate
[new-data experiment](docs/brca_updated_data_experiment.md) uses the uploaded
`data/brca/BRCA_input_data.csv` without regenerating it.

From the project root, obtain the pinned upstream files directly from the authors:

```bash
git -c core.autocrlf=false clone https://github.com/BML-cbnu/DriverMONI.git data/upstream
git -C data/upstream checkout --detach ddb0a830158ba41c90019bd5ca92054c2ba0869b
```

Skip this if the documented files are already present. Do not clone into an
existing nonempty `data/upstream` directory. If you already have the checkout
elsewhere, use `--data-root /absolute/path/to/DriverMONI` instead of copying it.
The entire `data/upstream/` directory is ignored by this project’s Git repository.

Only six processed/reference files are required, including the HumanNet ZIP;
no raw preprocessing or archive extraction is needed. Exact locations and
individual acquisition/placement instructions are in [data/README.md](data/README.md).
With the pinned inputs and default filters, expect **15,028 targets: 42 drivers
and 14,986 non-drivers**.

Inspect a sample before training:

```bash
python -m scripts.debug_brca --data-root data/upstream --gene ENSG00000012048
```

BRCA1 should have 491 nodes, 3,398 edges, DNN shape `(12,)`, node features
`(491, 13)`, and label 1. Preserve the supplied cache by default, including its
empty-neighborhood anomalies; rebuilding it changes the experiment.

## Run BRCA experiment

With the environment activated and data prepared:

```bash
python scripts/run_brca.py --config configs/brca.yaml
```

Using a separate upstream checkout and explicitly selecting CPU:

```bash
python scripts/run_brca.py --config configs/brca.yaml --data-root /absolute/path/to/DriverMONI --device cpu
```

The runner prints per-fold class counts, per-epoch metrics, selected-epoch metrics,
and mean ± std. It writes `results/brca_metrics.json`. A new run overwrites that
path; use `--output results/brca_seed43.json --seed 43` to keep a separate run.
No checkpoint is required or saved by this runner.

More precisely, YAML `upstream_root` and `output_path` are project-relative;
per-file data paths are relative to `upstream_root`. CLI `--config`, `--data-root`,
and `--output` paths are shell-directory-relative (absolute paths and `~` also
work). Omitting `--config` uses this project's default file from any directory.
See [every configuration value and path rule](docs/configuration.md). Unknown keys
and invalid settings fail explicitly instead of being silently ignored.

For a quick real-data end-to-end check:

```bash
python scripts/smoke_test.py --data-root data/upstream
```

This selects the first six genes per class, retains the full neighbor-feature
context, and runs **three folds, two epochs, batch size 4, CPU**. It verifies fold
coverage and source-compatible train/eval mode changes. Output goes to
`results/brca_smoke_metrics.json`, leaving the full experiment report untouched.
Its scores are a pipeline check, not a BRCA performance estimate.

### Run with the newly prepared dataset

To train on the newly prepared CSV instead, place the original labels and HumanNet
ZIP under `data/upstream` and run:

```bash
python scripts/run_brca.py --config configs/brca_updated_data.yaml --device cpu
```

This separate experiment uses all 16,786 genes, rebuilds PPI neighborhoods and
writes `results/brca_updated_data_metrics.json`. It keeps the model and training
settings below, but disables the original comparison-gene filter. See the
[input layout, PyCharm instructions and provenance limits](docs/brca_updated_data_experiment.md).
The original configuration and its results remain available.

## Training procedure

1. Read processed features and apply symbol deduplication. The baseline also
   applies the original comparison-gene/annotation filters; the new-data
   configuration explicitly disables them. Label membership in the supplied driver list.
2. Split targets using `StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)`.
   Graph features retain the full context selected by the active configuration
   across folds; labels are not features.
3. Create a fresh model and Adam optimizer per fold. Compute training-fold weights
   `w_c = (1 / n_c) * (N_train / 2)`, ordered `[non_driver, driver]`.
4. Train with weighted mean NLL after float64 log-softmax, Adam learning rate 0.001,
   default betas `(0.9, 0.999)`, eps `1e-8`, weight decay 0, and AMSGrad false.
   Run all 10 epochs with batch size 32, shuffled train/test loaders, workers 0,
   and no dropped final batch. No scheduler or early stopping is used.
5. Evaluate every epoch and retain the earliest epoch achieving maximum test AUC.
   Save all requested metrics at that same epoch.

**Two unusual source behaviors are deliberately preserved.** Epoch selection uses
**test-fold AUC**, so the reported scores are not an untouched-test-set estimate.
Also, evaluation calls `eval()` and the original never restores `train()`:
updates after epoch one use disabled dropout and frozen BatchNorm statistics.
The default config preserves this. Setting
`preserve_eval_mode_after_first_evaluation: false` is an explicit protocol change.
See [docs/training_protocol.md](docs/training_protocol.md).

## Evaluation metrics

Class 1 is the driver class. AUC uses `softmax(logits, dim=1)[:, 1]`; it never uses
hard predictions. Other metrics use driver probability **≥ 0.5**.

| Metric | Meaning |
| --- | --- |
| ROC-AUC | Area under the ROC curve using driver scores |
| Accuracy | Fraction of all predictions correct |
| Precision | Fraction of predicted drivers labeled as drivers |
| Recall | Fraction of labeled drivers recovered |
| F1 | Harmonic mean of driver precision and recall |
| Balanced accuracy | Mean of driver recall and non-driver recall |

Undefined precision/recall/F1 return zero. Fold summaries are unweighted mean and
population standard deviation (`ddof=0`), not confidence intervals. The source
reports AUC and balanced accuracy; other metrics and aggregation are additions.
High overall accuracy should be interpreted with the severe class imbalance.

## Results

A complete original-data baseline seed-42 CPU run finished all 30 epochs in approximately 16.6
minutes. These are results of this implementation and protocol, not a claim of
matching the paper’s scores.

| Metric | Mean ± std across three folds |
| --- | --- |
| ROC-AUC | 0.917385 ± 0.008582 |
| Accuracy | 0.977243 ± 0.012144 |
| Precision | 0.102249 ± 0.069126 |
| Recall | 0.523810 ± 0.121405 |
| F1 | 0.150854 ± 0.074647 |
| Balanced accuracy | 0.751162 ± 0.054469 |

Fold AUCs were 0.920022, 0.905806, and 0.926326; selected epochs were 7, 2, and 6.
The local full report is `results/brca_metrics.json`; generated reports are not
included in Git. A teammate regenerates it with the command above. It includes
fold IDs, predictions, effective configuration, runtime versions, input SHA-256
hashes, counts, weights, epoch histories, and completion status. Only
`status: complete` represents a finished run. See [results/README.md](results/README.md).

The separate **new-data experiment** completed all three folds and 30 epochs on
17 October 2026 in 19.7 minutes on CPU. With all 16,786 targets, seed 42 and rebuilt
neighborhoods, its mean ROC-AUC was **0.932188 ± 0.020985**; precision was
0.060602 ± 0.020310 and recall 0.523810 ± 0.089087 at threshold 0.5. These results
use a different population and graph context from the baseline, and preserve
test-fold epoch selection; they do not establish an improvement over the baseline.
All metrics, fold details, PyCharm instructions and unresolved preprocessing
provenance are documented in the [new-data experiment guide](docs/brca_updated_data_experiment.md).

## Differences from original implementation

- Clean modules and names replace the monolithic script. Target masks use
  `is_target`, graph temporaries do not mutate input features, and device moves
  support CPU/CUDA. State-dictionary names differ; tests explicitly map weights.
- Modern DGL calls, constant-time gene lookups, deterministic node ordering, and
  ZIP loading replace legacy plumbing. The shipped `PPI_graph/` path and explicit
  output path repair the original missing `string/` and `graph_name_` dependencies.
- Unused layers are removed. This preserves the active computation under matched
  weights but changes initialization RNG consumption. Equal seeds alone do not
  give identical full-model upstream weights.
- Seed 42, explicit CV random_state, determinism settings, CPU thread count,
  reporting conventions, and all-zero-AUC first-epoch fallback are new choices.
- The active source’s extra GAT 32→32 block, shared attention/message projection,
  and unfiltered directed graph differ from aspects of the paper’s description.
  Source behavior is retained. Raw feature-formula differences are documented,
  not silently corrected or regenerated.

See [REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md) for exact evidence and unresolved
points. There is no claim of an identical original full training trajectory.

## Reproducibility

The runner seeds Python, NumPy, PyTorch, CUDA when present, and DGL once before CV.
The config records seed, device, deterministic algorithms, and CPU thread count.
Keep input files and package versions fixed. Hardware changes, graph summation
order, and library versions may affect floating-point results.

Run standalone synthetic tests without upstream data:

```bash
python -m pytest -q tests
```

Tests that require original source/data skip explicitly when those files are not
provided. Raw preprocessing also has a standard-library test suite and an
optional cached-source replay test; see [BRCA source preparation](docs/brca_source_preparation.md).
For the complete parity suite, supply the separately obtained checkout:

```bash
python -m pytest -q tests --brca-data-root data/upstream
```

Original classes are extracted into memory from hash-verified source files;
module-level imports, CLI execution, and data-loading code are not run. Minimal
CPU/DGL compatibility adapters are audited. No verbatim reference class files are
shipped. A data-only directory cannot run original-source parity comparisons.

Latest full verification: **103 passed, 2 skipped** (CUDA unavailable; optional
raw-source replay runs separately). Intermediate
tensors/logits match exactly on controlled CPU fixtures with transferred active
weights; gradients and one Adam update agree within numerical tolerance. Five
real BRCA graphs match by gene identity, features, labels, and directed edge
multiplicity. The separate 12-target real-data smoke test completed all six epochs
and three folds. These checks do not establish historical folds, original
initialization, CUDA parity, or publication-wide performance reproduction.
The [teammate review](docs/teammate_review.md) records fresh installation, relocated
execution, configuration validation, and the fixes made without changing the model.

## References

1. Park, J. and Jo, K. DriverMONI paper. [DOI: 10.1109/TCBBIO.2025.3643887](https://doi.org/10.1109/TCBBIO.2025.3643887).
2. Authors’ [DriverMONI source and processed inputs, pinned revision](https://github.com/BML-cbnu/DriverMONI/tree/ddb0a830158ba41c90019bd5ca92054c2ba0869b).
3. [DGL installation documentation](https://www.dgl.ai/pages/start.html).
4. [GitHub documentation: licensing a repository](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).
