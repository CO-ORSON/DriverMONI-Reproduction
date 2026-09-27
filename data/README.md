# BRCA input data

Datasets are excluded from Git. This workspace has local copies of the required
BRCA files in `data/upstream/`. For another checkout, use the processed data from
revision `ddb0a830158ba41c90019bd5ca92054c2ba0869b` of
[DriverMONI](https://github.com/BML-cbnu/DriverMONI/tree/ddb0a830158ba41c90019bd5ca92054c2ba0869b).
Raw mutation/expression preprocessing is outside this stage.

## Obtain the inputs

No redistribution license was found in the pinned authors' repository. We do not
bundle these files or claim permission to redistribute them. Download them directly
from the authors; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). Do not
force-add the data directory to Git or send a ZIP containing these local files.

From a fresh reproduction checkout, after activating its environment:

```bash
git -c core.autocrlf=false clone https://github.com/BML-cbnu/DriverMONI.git data/upstream
git -C data/upstream checkout --detach ddb0a830158ba41c90019bd5ca92054c2ba0869b
python -m scripts.debug_brca --data-root data/upstream --gene ENSG00000012048
python scripts/run_brca.py --config configs/brca.yaml
```

If `data/upstream/` already contains the files, skip cloning. For an existing
upstream checkout elsewhere, supply `--data-root /absolute/path/to/DriverMONI`
to the debug, experiment, and smoke-test commands. You do not need to copy it.
The clone command disables line-ending conversion so optional source-hash parity
checks also see the original file bytes on platforms that default to CRLF.

Alternatively, obtain just the following files from the pinned GitHub links
(use **Raw / Download raw file** on each file page), creating the listed directories
under `data/upstream/`. The ZIP is used in place of an extracted edge CSV:

| Download from the authors | Place below `data/upstream/` |
| --- | --- |
| [BRCA features](https://github.com/BML-cbnu/DriverMONI/blob/ddb0a830158ba41c90019bd5ca92054c2ba0869b/sample_preprocessed_file/BRCA_input_data.csv) | `sample_preprocessed_file/BRCA_input_data.csv` |
| [BRCA labels](https://github.com/BML-cbnu/DriverMONI/blob/ddb0a830158ba41c90019bd5ca92054c2ba0869b/using_Data/BRCA_label.csv) | `using_Data/BRCA_label.csv` |
| [Comparison genes](https://github.com/BML-cbnu/DriverMONI/blob/ddb0a830158ba41c90019bd5ca92054c2ba0869b/for_concat_data/after_concat_BRCA_genes2.csv) | `for_concat_data/after_concat_BRCA_genes2.csv` |
| [Gene annotations](https://github.com/BML-cbnu/DriverMONI/blob/ddb0a830158ba41c90019bd5ca92054c2ba0869b/gene_info_for_GOSemSim.csv) | `gene_info_for_GOSemSim.csv` |
| [HumanNet archive](https://github.com/BML-cbnu/DriverMONI/blob/ddb0a830158ba41c90019bd5ca92054c2ba0869b/for_ref/humannet_v2_and_v3.zip) | `for_ref/humannet_v2_and_v3.zip` |
| [BRCA neighbor cache](https://github.com/BML-cbnu/DriverMONI/blob/ddb0a830158ba41c90019bd5ca92054c2ba0869b/PPI_graph/gene_n_humannet_BRCA.pickle) | `PPI_graph/gene_n_humannet_BRCA.pickle` |

The data-only option supports training and the smoke command. Original-source
parity tests additionally need `CV_multimodal_compare.py` and
`torch_models_linear.py` at the root of a separately acquired pinned checkout.

## Required schema

The loader expects these paths below the supplied `upstream_root`. The debug
command accepts this root via `--data-root` and defaults to `data/upstream/`:

| Relative path | Purpose |
| --- | --- |
| `sample_preprocessed_file/BRCA_input_data.csv` | Gene identifiers and 13 numeric features. |
| `using_Data/BRCA_label.csv` | Positive Ensembl IDs in the `gene_label` column. |
| `for_concat_data/after_concat_BRCA_genes2.csv` | Default comparison-gene list. |
| `gene_info_for_GOSemSim.csv` | Symbol whitelist used by the default filter. |
| `for_ref/humannetv3_ens_ppi.csv` | Unfiltered HumanNet v3 edge table. |
| `PPI_graph/gene_n_humannet_BRCA.pickle` | Supplied one-hop neighbor lists. |

The edge CSV is inside `for_ref/humannet_v2_and_v3.zip`, under the archive member
`humannet_v2_and_v3/humannetv3_ens_ppi.csv`. The loader can read this member directly
without extraction when the standalone CSV is absent. It does not modify the
authors' checkout or write regenerated neighbor caches to it.
The upstream endpoint columns are named `entrez_id_1` and `entrez_id_2`, but the
bundled values are Ensembl IDs. Preserve the CSV's edge orientation.

## Feature order

The DNN columns, in order, are:

1. `synonymous_variant`
2. `stop_gained`
3. `missense_variant`
4. `frameshift_variant`
5. `splice`
6. `inframe`
7. `lost_stop and start`
8. `deg`
9. `related_pathway`
10. `dir_pathway`
11. `muta_count`
12. `miss_ratio`

GAT node features append `PPI` as column 13. Gene identifiers, target masks, and
labels are not feature columns. Keep processed values unchanged.

## Validation expectations

The inspected BRCA table contains 16,875 genes. Applying the upstream comparison
filter leaves 15,028 target genes, including 42 positives. Treat these as checks
against the pinned inputs, not as universal constants for other datasets.

Preserve the supplied neighbor cache by default: inspection found five empty
BRCA entries despite network neighbors being available. Cache regeneration changes
those graph inputs and must be recorded explicitly. Cache deserialization should
accept only the expected dictionary/list/string content.

The loader fails clearly on missing files, incompatible gene IDs, nonfinite
features, or missing target cache entries. Record file checksums before training.
Large inputs are excluded by `.gitignore`.

`GeneSample` returns `gene_features`, `graph`, scalar integer `label`, `gene_id`,
and `node_gene_ids` in graph-node order. The graph stores `feat` [N,13],
`is_target` [N] (boolean), and `global_node_id` [N]. Labels are never node features.
`collate_gene_samples` returns `GeneBatch`; `.to(device)` moves features, graph,
and labels together. Restrict `target_gene_ids` when creating fold datasets while
retaining the full filtered table as the feature context.
