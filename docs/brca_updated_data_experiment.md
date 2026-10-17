# BRCA experiment with the newly prepared dataset

This experiment uses `data/brca/BRCA_input_data.csv` from teammate branch
`my-extension` at commit `eea3323d95c8ece5914b15546434933a1114f4aa`.
Its configuration is `configs/brca_updated_data.yaml`. The original-data baseline
remains `configs/brca.yaml`; the model and training implementation are unchanged.

## Inputs and population

| Setting | New-data experiment |
| --- | --- |
| Targets and shared graph-feature context | All 16,786 genes in the uploaded CSV |
| Positive class | Original 42 IDs in the authors' `using_Data/BRCA_label.csv` |
| Class 0 | Remaining 16,744 genes under the original benchmark convention; biologically unlabelled |
| Original comparison/annotation filter | Disabled: its comparison list requests 84 absent genes |
| Supplied neighborhood cache | Disabled: it has no entries for three newly present genes |
| Neighborhoods | Rebuilt one-hop neighborhoods using the existing graph implementation |
| Message edges | Original unfiltered HumanNet directions, duplicates and synthetic self-loops |
| Feature schema | Same 12 ordered DNN features and 13 GAT features |

The new driver-evidence/status tables are not used to redefine labels. The scalar
`PPI` values already present in the CSV are used as supplied; rebuilding graph
neighborhoods does not recompute that feature. The training graph is unfiltered
as in the source, even though the preprocessing manifest describes a threshold
for calculating the scalar PPI feature.

The original feature table has 16,875 genes, while its filtered baseline uses
15,028 targets. Changing to 16,786 targets changes class balance, folds and graph
context. Rebuilding neighborhoods also changes five known empty-cache cases.
Results must be identified as a separate new-data experiment; a score difference
does not isolate an improvement due to preprocessing or the model.

## Place the files

Use the Python 3.12 training environment described in the main README. The new
CSV is already tracked in this branch. Only two original auxiliary files are
required for this configuration:

```text
data/
├── brca/
│   ├── BRCA_input_data.csv
│   ├── manifest.json
│   └── summary.json
└── upstream/
    ├── using_Data/BRCA_label.csv
    └── for_ref/humannet_v2_and_v3.zip
```

Obtain those two files from the authors' pinned revision
`ddb0a830158ba41c90019bd5ca92054c2ba0869b`, or copy them from an existing checkout
into these locations. No ZIP extraction is needed. A full upstream checkout at
`data/upstream` also works. The original files remain ignored by Git.

- [Original labels](https://raw.githubusercontent.com/BML-cbnu/DriverMONI/ddb0a830158ba41c90019bd5ca92054c2ba0869b/using_Data/BRCA_label.csv)
- [Original HumanNet archive](https://raw.githubusercontent.com/BML-cbnu/DriverMONI/ddb0a830158ba41c90019bd5ca92054c2ba0869b/for_ref/humannet_v2_and_v3.zip)

Expected SHA-256 values for this reviewed snapshot:

| Input | SHA-256 |
| --- | --- |
| New feature CSV | `f5b6d89aa4f08af738185a025bf1fe78deed8a57b60f183066024c6777552f4c` |
| Original labels | `da6235d0433cfeca6f62ee15df4ca59a4fb358643af5a26e7bc974dbdfb6bff9` |
| HumanNet ZIP | `a4f2def6d7b5665f5ea177928e6f129bb4585f2a8073474586cbdffe25c69c51` |

Here `data.upstream_root: data` is the base for all configured input paths. For
this configuration, `--data-root` would override the combined data folder, which
must contain both `brca/` and `upstream/`; do not pass an authors-only checkout.
The comparison, annotation and cache paths remain in YAML because the existing
configuration schema requires them; their files are not read with both switches
disabled.

## Run all three folds

From this reproduction project's root:

```bash
python scripts/run_brca.py --config configs/brca_updated_data.yaml --device cpu
```

The configuration retains seed 42, deterministic algorithms, one CPU thread,
three shuffled stratified folds, 10 epochs per fold, batch size 32, weighted NLL,
Adam at 0.001, no weight decay, no scheduler and no early stopping. CPU was chosen
for the local run; the configuration's `auto` default also supports compatible
CUDA installations when the command-line override is omitted.

| Fold | Train driver / class 0 | Test driver / class 0 |
| --- | --- | --- |
| 1 | 28 / 11,162 | 14 / 5,582 |
| 2 | 28 / 11,163 | 14 / 5,581 |
| 3 | 28 / 11,163 | 14 / 5,581 |

The source's unusual behavior remains explicit: updates after epoch one use
evaluation mode, and each fold's reported epoch is selected by maximum test-fold
AUC. These are source-compatible scores, not an untouched-test-set estimate.

### PyCharm

Create a **Python** run configuration with:

- Interpreter: the existing Python 3.12 environment with the project requirements.
- Script: `scripts/run_brca.py` in this project.
- Parameters: `--config configs/brca_updated_data.yaml --device cpu`.
- Working directory: this reproduction project's root.

Run it normally; this is a training configuration, not a pytest configuration.
The runner sets the DGL PyTorch backend automatically.

## Results and provenance

The runner writes `results/brca_updated_data_metrics.json`, separate from the
original `results/brca_metrics.json`. It records effective settings, package
versions, consumed input hashes, all fold IDs, epoch metrics, selected predictions
and completion status. A later run overwrites this new-data filename; use
`--output` to choose another path when retaining multiple runs.

The local reviewed run also saves `brca_updated_data_run.log`,
`brca_updated_data_tests.log`, and `brca_updated_data_provenance.json` under
`results/`. The provenance sidecar records the source/configuration snapshots,
the preprocessing manifest hashes, and unresolved issues; the standard training
runner does not automatically generate that sidecar for future runs.

**Preprocessing provenance remains unresolved:** the uploaded manifest's code
hashes for `brca_omics.py` and `brca.py` differ from the current source. This run
uses the uploaded CSV as a fixed input and does not regenerate or certify it.
The CSV and summary file checksums do match the manifest. Source-to-CSV replay
was not performed, and nine manifest-listed audit/reference outputs are absent
from the checkout. The teammate should identify the generating code version or
verify a replay before claiming complete preprocessing reproducibility.

## Local completion record

Completed **17 October 2026 at 11:34 Singapore time**, in **19.7 minutes** on
macOS arm64 CPU: Python 3.12.14, PyTorch 2.2.2, DGL 2.2.0, NumPy 1.26.4,
pandas 3.0.6 and scikit-learn 1.9.1. The JSON status is `complete`; every fold
contains all ten epochs.

| Fold | Selected epoch | ROC-AUC | Accuracy | Precision | Recall | F1 | Balanced accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 0.909288 | 0.987491 | 0.088235 | 0.428571 | 0.146341 | 0.708732 |
| 2 | 2 | 0.959985 | 0.970688 | 0.053571 | 0.642857 | 0.098901 | 0.807184 |
| 3 | 9 | 0.927292 | 0.968722 | 0.040000 | 0.500000 | 0.074074 | 0.734949 |

| Metric | Mean ± population standard deviation across folds |
| --- | --- |
| ROC-AUC | 0.932188 ± 0.020985 |
| Accuracy | 0.975634 ± 0.008423 |
| Precision | 0.060602 ± 0.020310 |
| Recall | 0.523810 ± 0.089087 |
| F1 | 0.106439 ± 0.029981 |
| Balanced accuracy | 0.750288 ± 0.041631 |

These are metrics at the selected epoch, not necessarily the final epoch. The
standard deviation is not a confidence interval. High overall accuracy reflects
the severe class imbalance; precision and recall should be reported alongside it.

Verification completed:

- Existing suite: **114 passed, 2 skipped**, 23.74 seconds, with the original
  source/data checkout supplied for parity checks. CUDA and cached-source replay
  remain optional skipped checks.
- All 16,786 targets were tested exactly once; training/test targets are disjoint
  within each fold. Counts and labels match the original 42-positive list.
- All saved probabilities and losses are finite; learning rate, class weights,
  source train/eval behavior and best-epoch selection match the configuration.
- All six fold metrics were independently recalculated from saved predictions;
  the aggregate mean and population standard deviation also match.
- Consumed input hashes match the files; model/training code and baseline YAML
  remain unchanged. No source-to-CSV replay or full-run repeat was performed.

Run the same suite from the project root, replacing the external checkout path
as needed:

```bash
python -m pytest tests -q --brca-data-root=/path/to/original/DriverMONI
```

The two-file `data/upstream` layout above is sufficient for the new-data training
run; original-source parity tests require the full external authors' checkout.
