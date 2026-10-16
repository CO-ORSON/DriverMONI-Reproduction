# BRCA input data

Two independent workflows are available: **reproduce the original benchmark**
using the authors' processed inputs, or **generate a canonical dataset** from
retrieved raw/reference sources. The training loader currently uses the original
benchmark inputs; canonical generation does not switch its configuration.
The canonical BRCA feature dataset is available at `data/brca/BRCA_input_data.csv`. The linked driver-evidence and gene-status tables are available under `data/brca_driver_evidence/`.

## Generate the canonical BRCA dataset

The new source-preparation workflow replaces **hard-coded gene and pathway
membership lists** with retrieved source tables and explicit identifier mappings.
It records **feature-to-source mapping**, formulas, mapping versions and checksums
so acquisition and raw-to-feature processing can be reproduced and audited.

From the reproduction root, using Python 3.10+ and the standard library:

```bash
python3 scripts/preprocess_brca.py
```

[brca_preprocessing.json](../configs/brca_preprocessing.json) selects the source
configs, cache and output directory. The command acquires its own inputs;
cloning the benchmark repository is unnecessary for this workflow.

| Feature family | Current source |
| --- | --- |
| Nine mutation features | Authors' pinned VarScan MAF, Ensembl 102 annotation and author reference tables. |
| Expression (`deg`) | Authors' pinned BRCA expression matrix. |
| KEGG (`dir_pathway`, `related_pathway`) | Retrieved breast-cancer and related-pathway memberships, mapped through dated HGNC cross-references. |
| HumanNet (`PPI`) | Authors' pinned mapped HumanNet v3 network; its edges also supply the GAT graphs. |

Ensembl 102 maps mutation symbols; the dated HGNC table maps KEGG/NCBI IDs and
is not an Ensembl-102 reconstruction. Labels are audited separately and do not
select genes or enter feature construction. The default expression example has
94 tumors and six normals; its units and equivalence to the full study cohort
remain undocumented.

| Output | Purpose |
| --- | --- |
| `data/brca/BRCA_input_data.csv` | Combined mutation, expression, KEGG and PPI features in the existing schema. |
| `data/brca/audit/`, `data/brca/references/` | Raw counts, group means, samples, mapping/exclusion audits, label coverage and pathway evidence. |
| `data/brca/manifest.json`, `data/brca/summary.json` | Source URLs, checksums, configuration, formulas, mapping versions and coverage. |
| `data/cache/` | Shared download cache, including the mapped HumanNet ZIP. |

Replay the recorded source snapshots without network access:

```bash
python3 scripts/preprocess_brca.py --offline --lock-manifest data/brca/manifest.json
```

The lock verifies source URLs and checksums. Optional modality diagnostics write
under `data/diagnostics/`; historical datasets and receipts are in `data/history/`.
See [BRCA source preparation](../docs/brca_source_preparation.md) for source links,
formula details, overrides and synthetic/real-source replay tests.

## Cancer-genomics purpose of the features

These features combine evidence of altered coding sequence, transcription,
known cancer mechanisms and network context. They support driver prediction;
none alone establishes that a gene drives BRCA.

1. **Mutation consequences and recurrence (nine features).** Seven consequence
   proportions describe potential functional changes: stop-gained and frameshift
   mutations can impair protein function; missense and inframe mutations can alter
   activity; splice-region mutations may affect RNA processing; and start/stop-loss
   mutations can alter translation initiation or termination. Synonymous mutations
   provide a protein-sequence-preserving comparison, although they are not always
   neutral ([Ensembl definitions](https://static.ensembl.org/info/genome/variation/prediction/predicted_data.html)).
   `miss_ratio` contrasts missense with synonymous substitutions. `muta_count`
   represents standardized recurrence across **distinct tumor samples** with at
   least one `stop_gained`, `missense_variant`, `frameshift_variant` or
   `splice_region_variant` annotation; each sample counts once. Together, these
   features inform driver prediction through functional consequences and
   recurrence, without establishing that individual mutations are drivers.
2. **Expression (`deg`).** Transcript abundance captures tumor-associated
   dysregulation that mutations alone miss, including regulatory and
   copy-number effects.
   `deg` represents the direction and magnitude of the tumor–normal expression
   contrast. It adds transcriptional context, rather than establishing a causal
   driver mechanism or statistically significant differential expression.
3. **Pathway membership (`dir_pathway`, `related_pathway`).** KEGG supplies prior
   biological context for breast-cancer signaling, hormone response, cell-cycle
   control and DNA repair ([BRCA map](https://www.kegg.jp/entry/path%3Ahsa05224)).
   `dir_pathway` marks involvement in the breast-cancer map; `related_pathway`
   represents participation across related biological modules. These features
   help place alterations in known mechanisms. Membership does not measure
   pathway activation, and absent annotation does not exclude a novel driver.
4. **Network context (`PPI` and GAT).** Cancer mechanisms involve gene modules,
   so network neighbors add context beyond a gene's own alterations. HumanNet
   includes physical and functional association networks; its edges should not
   automatically be interpreted as direct protein binding
   ([HumanNet v3](https://pmc.ncbi.nlm.nih.gov/articles/PMC8728227/)). The scalar
   `PPI` feature emphasizes connections to recurrently mutated genes, providing
   mutation-conditioned network context. The GAT learns from neighboring genes'
   multiomics features. Neither network proximity nor mutation recurrence alone
   proves driver status.

The model combines a gene's own features through the DNN with neighborhood
information through the GAT. See [BRCA source preparation](../docs/brca_source_preparation.md)
for calculation methods and [data sources](../docs/data_sources.md) for provenance
and remaining limitations.

## Canonical versus original BRCA features

Both CSVs have the same two identifiers and 13 features. The original contains
**16,875 genes**, the canonical file **16,786**, with **16,783 shared genes**
and all **42 supplied positives** retained. There are 92 original-only IDs and
three canonical-only IDs. Population changes include 83 genes without expression
under their mapped IDs, six ambiguous symbol mappings, and additional ID changes.

Comparisons below align shared Ensembl IDs; differences below `1e-10` count as
numerical agreement.

| Feature | Comparison and explanation |
| --- | --- |
| Seven mutation proportions and `miss_ratio` | Agree for every shared gene. |
| `muta_count` | Raw affected-sample counts agree; z-scores differ slightly because the normalization population changes. |
| `deg` | 16,748 shared genes differ (correlation 0.611). Canonical values use 94 tumors and six normals; the original CSV's exact expression cohort, scale and processing history remain unresolved. |
| `dir_pathway` | Identical for shared genes. Total direct members fall from 137 to 136 because FGF23 is excluded. |
| `related_pathway` | Raw counts change for 449 shared genes through PI3K–Akt deduplication and refreshed KEGG/HGNC membership; all z-scores change. |
| `PPI` | Same network and threshold; 46 shared genes lose one qualifying neighbor. Population changes also alter all PPI z-scores. |

Five retained genes have different displayed symbols, affecting the original
symbol whitelist. The current loader also rejects direct substitution because
its comparison list requests 84 IDs absent from the canonical file. Treat the
canonical export as a separate dataset version; use the original processed CSV
for strict benchmark reproduction. Matching schemas do not establish identical
feature values or model results.

## Review BRCA driver evidence

`python3 scripts/prepare_brca_driver_evidence.py --offline` builds two linked
tables under `data/brca_driver_evidence/`: source assertions and per-gene review
status. It combines public OncoKB exports, IntOGen BRCA cohort calls and reviewed
literature. As the full CGC export is unavailable, the **42 DriverMONI labels**
serve as the **paper-attributed CGC v95 subset**, with extraction rules and tiers
marked unverified. Benchmark labels/features remain unchanged; unresolved genes
are not confirmed negatives. Sources, mappings, cohort overlap and mechanism
coverage are recorded. See [driver evidence](../docs/brca_driver_evidence.md)
for qualifications, status rules and replay/validation instructions.

## Shared feature schema

[src/feature_schema.py](../src/feature_schema.py) defines the DNN feature order:

```text
synonymous_variant, stop_gained, missense_variant, frameshift_variant,
splice, inframe, lost_stop and start, deg, related_pathway, dir_pathway,
muta_count, miss_ratio
```

GAT node features append `PPI` as feature 13. Gene identifiers and labels are
separate from numeric features. Sharing this schema does not establish benchmark
parity; input populations, values and graph context must also agree.
