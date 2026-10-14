# BRCA source preparation

`scripts/preprocess_brca.py` acquires the authors' mutation/expression examples,
retrieves KEGG memberships and combines them with scalar PPI from the mapped
HumanNet v3 snapshot into one 13-feature CSV. This guide describes execution and
calculations; [data sources](data_sources.md) records provenance and
[data/README.md](../data/README.md#cancer-genomics-purpose-of-the-features) explains
biological purpose. Training continues to use the original benchmark inputs.

## Run and replay

From the reproduction root, using Python 3.10+ and the standard library:

```bash
python3 scripts/preprocess_brca.py
python3 scripts/preprocess_brca.py --offline --lock-manifest data/brca/manifest.json
```

[brca_preprocessing.json](../configs/brca_preprocessing.json) selects the
[omics config](../configs/brca_omics_sources.json),
[pathway config](../configs/brca_sources.json), mapped network, labels, shared
`data/cache/` and output `data/brca/`. Author sources are pinned to commit
`ddb0a830158ba41c90019bd5ca92054c2ba0869b`; mapping references are versioned
separately. The manifest records consumed sources, checksums, settings, formulas,
Python/implementation versions and output hashes. A locked replay rejects changed
source URLs or bytes.

Use `--output-dir` for another dataset version, `--cache-dir` to relocate sources,
and `--refresh` to fetch new bytes. CLI paths are relative to the shell directory;
config paths are relative to the project. Custom configs support raw URLs and
text/gzip/ZIP formats. Local MAF, expression and GTF overrides use `local_path`;
new expression units/transforms must be declared. Inputs must satisfy the MAF
and Ensembl gene-by-sample matrix contracts described below.

## Raw-to-feature processing

Mutation/expression parsing and scalar PPI are implemented in
[src/preprocessing/brca_omics.py](../src/preprocessing/brca_omics.py); KEGG retrieval
and mapping are in [brca_sources.py](../src/preprocessing/brca_sources.py).

### Mutation features

Resolve the primary `Hugo_Symbol` using Ensembl 102, then the authors' fallback
mapping. Exclude/report ambiguous symbols and apply the pinned protein-coding
filter. Expand semicolon-separated `all_effects` transcript annotations under
that primary gene, matching consequence terms exactly. Let `T` be the gene's
retained annotation count, including consequences outside the seven reported
proportions.

| Feature | Numerator, divided by `T` |
| --- | --- |
| `synonymous_variant` | `synonymous_variant` |
| `stop_gained` | `stop_gained` |
| `missense_variant` | `missense_variant` |
| `frameshift_variant` | `frameshift_variant` |
| `splice` | `splice_region_variant` only |
| `inframe` | `inframe_insertion` + `inframe_deletion` + `conservative_inframe_deletion` |
| `lost_stop and start` | `stop_lost` + `start_lost` |

`miss_ratio = missense / (missense + synonymous)`, or zero when the denominator
is zero. Raw `muta_count` counts distinct tumor sample–gene groups with at least
one stop-gained, missense, frameshift or splice-region annotation. Multiple
annotations in the same sample contribute once to this count. Sample IDs retain
the TCGA vial component; they are not patient counts.

Transcript multiplicity remains in consequence proportions, so these are
annotation-weighted rather than distinct-variant proportions. The supplied MAF
receives no additional `PASS` filter. These compatibility choices do not establish
optimal variant QC. Neither `miss_ratio` nor recurrence is a background-corrected
selection test.

### Expression

Normalize Ensembl version suffixes and parse the fourth TCGA barcode component:
primary tumors `01`, solid-tissue normals `11`. Require both groups; reject
duplicate normalized sample/gene IDs and missing/nonfinite values.
Calculate `deg = mean(tumor values) - mean(normal values)` on the declared scale.
Default `identity` preserves the authors' values without another log transform.

Numerical inversion (`x = 2**y - 1`) recovers 12-significant-digit quantities
within floating-point precision for all 1,720,480 positive expression values.
This is strong evidence of upstream `log₂(x + 1)`, consistent with
[Xena's documented convention](https://github.com/ucscXena/xena-GDC-ETL), and
supports retaining the mean difference. On that scale, `deg` is the log₂ ratio
of geometric means of `x + 1`; it differs from the paper's log₂ ratio of arithmetic
means on the linear scale. The historical dataset ID, release and unit remain
unverified. This feature is a group contrast, not a statistical DE test.

### Pathway retrieval and membership

1. Retrieve the [human pathway catalogue](https://rest.kegg.jp/list/pathway/hsa),
   require a unique exact disease-name match after removing the organism suffix,
   and record the selected ID/rationale in `pathway_selection.json`.
2. Fetch the selected entry (`hsa05224` for BRCA). Parse fixed-width `REL_PATHWAY`
   fields and continuation lines; deduplicate related IDs and exclude the direct ID.
3. Retrieve `https://rest.kegg.jp/link/hsa/<resolved-pathway-id>` for the direct
   and each related pathway. Validate direction and human NCBI Gene IDs against
   the [KEGG API contract](https://www.genome.jp/kegg/rest/keggapi.html).
4. Map through approved records in the configured dated HGNC snapshot, using
   `entrez_id`, `ensembl_gene_id` and `symbol`. Remove Ensembl version suffixes
   and report missing/ambiguous mappings rather than selecting the first match.
5. Assign `dir_pathway = 1` for direct membership, otherwise zero. Raw
   `related_pathway` counts distinct retrieved related pathways containing the
   gene; labels and gene overlap do not determine the related-pathway list.

HGNC is a dated cross-reference, not an Ensembl-102 reconstruction. The
[KEGG API policy](https://www.genome.jp/kegg/rest/) permits academic use with at
most three calls per second; cache responses for reproducible replay.

### Gene population, PPI and scaling

Join eligible mutation genes to expression by normalized Ensembl ID. Report
missing-expression exclusions and keep genes without KEGG membership. Mutation
and expression cohorts are aggregated independently. Labels audit coverage but
do not select genes or enter feature construction.

For scalar `PPI`, retain HumanNet edges with `LLS > 3.211`, preserving duplicate
adjacency entries. Across the final gene population, calculate the linearly
interpolated 75th percentile of **raw** `muta_count` and take its integer floor.
For targets with raw `muta_count > 0`, count feature-bearing neighbors whose raw
count strictly exceeds that threshold; otherwise assign zero. Report non-Ensembl
endpoints; only neighbors in the final feature table contribute. These thresholds
are upstream heuristics, not validated driver cutoffs.

Standardize raw `muta_count`, related-pathway counts and PPI counts as
`(value - mean) / sample_std`, using `ddof=1` over the final exported genes.
Fail on zero variance. Proportions, `miss_ratio`, `deg` and binary `dir_pathway`
retain their values. Scaling aligns count-feature ranges for training; it does
not correct biological background rates.

The GAT separately uses the **unfiltered** mapped network and supplied one-hop
neighborhoods, preserving CSV edge direction/multiplicity. This entry point does
not rebuild graphs or neighbor caches. See [graph contracts](graph_inputs.md)
for changes to network inputs.

## Outputs and diagnostics

| Artifact under `data/brca/` | Contents |
| --- | --- |
| `BRCA_input_data.csv` | Two identifiers and 13 numeric features in [shared schema](../src/feature_schema.py) order; no labels |
| `audit/gene_counts.csv` | Annotation denominators and raw mutation, related-pathway and PPI counts |
| `audit/expression_means.csv`, `audit/samples.csv` | Group means, sample inclusion and cohort overlap |
| `audit/mutation_mapping.csv`, `audit/pathway_mapping.csv` | Mapping choices and exclusions |
| `audit/excluded_genes.csv`, `audit/driver_label_comparison.csv` | Missing-expression exclusions and supplied-positive coverage |
| `references/pathway_selection.json`, `references/pathway_gene_membership.csv` | Selected pathway/rationale and mapped memberships |
| `manifest.json`, `summary.json` | Provenance, configuration, formulas, hashes and coverage |

HGNC and the mapped network remain in the shared cache; the network is read
directly from its ZIP. Optional modality diagnostics are available:

```bash
python3 scripts/preprocess_brca_omics.py
python3 scripts/preprocess_brca_sources.py --labels /path/to/BRCA_label.csv
```

Defaults write to `data/diagnostics/omics/` and `data/diagnostics/pathways/`.
The pathway command accepts `--base-features` to refresh only pathway columns;
the omics command accepts `--pathway-membership` and `--network-edges` for the
full schema. Neither is a prerequisite for the combined command. Earlier
receipts and the distinct pathway-only export remain under `data/history/`;
replay uses temporary directories rather than permanent duplicate datasets.

## Snapshot and benchmark comparison

The 13 October 2026 canonical run produced **16,786 genes**, retaining all
**42 supplied positives** and **136 direct pathway members**. It excluded six
ambiguous symbols (28 variant rows) and 83 mutation genes without expression.
Mutation covers 985 tumor samples; expression covers 94 tumors and six normals,
with 85 tumor samples shared. The MAF header's 986 analyzed samples does not match
the observed roster. These independently aggregated examples do not establish
the full paper cohort or equality with the supplied 16,875-gene feature table.
See [the feature comparison](../data/README.md#canonical-versus-original-brca-features)
for modality-level differences.

The earlier pathway-only audit used all **16,875 base genes**, with **137 direct
members**, eight unique related pathways and HGNC **2026-10-06**. It mapped
1,120 unique KEGG NCBI IDs to 1,117 Ensembl genes; IDs `125467750`, `102723407`
and `100653049` were unmapped. Of the 42 positives, 10 were direct members and
21 appeared anywhere in the retrieved pathway union. The union included 88
mapped genes absent from the base CSV, without creating other features for them.

The original BRCA list repeats `hsa04151`; the original READ list contains
`has04010`. Retrieved, deduplicated memberships avoid these defects. In the
pathway-only BRCA refresh, 453 genes' raw related counts changed and all related
z-scores changed; direct membership stayed unchanged. These figures use a
different population from the canonical comparison. Their effect on model
performance has not been evaluated.

## Label audit

The paper attributes the 42 supplied labels to CGC v95, but the full export and
BRCA extraction policy are unavailable. Somatic/germline fields, tiers, historical
mapping and availability filters therefore remain unverified. Training assigns
one for list membership and zero otherwise; biologically, the remaining genes
are **unlabelled**, not established negatives. Pathway membership does not supply
replacement labels.

The separate [driver-evidence workflow](brca_driver_evidence.md) uses the 42 labels
as a qualified CGC-v95 proxy and links OncoKB, IntOGen and reviewed literature
without changing features or benchmark labels. If a CGC export becomes available,
an optional exact-symbol diagnostic reports tiers and breast-associated
somatic/germline fields:

```bash
python3 scripts/preprocess_brca_sources.py \
  --labels /path/to/BRCA_label.csv \
  --base-features /path/to/BRCA_input_data.csv --offline \
  --cgc-csv /path/to/cancer_gene_census_v95.csv --cgc-version v95 \
  --output-dir data/diagnostics/pathways_cgc
```

This diagnostic never changes labels and does not reproduce an unknown historical
selection rule; symbol changes can also cause unmatched records.

## Validation

```bash
python3 -m unittest tests.test_brca_sources tests.test_brca_omics tests.test_brca_pipeline -v
BRCA_PREPROCESSING_DATA_ROOT=data python3 -m unittest tests.test_brca_snapshot -v
```

Synthetic tests check all 13 features against hand-calculated evidence, including
annotation/sample counting, expression groups/scale, exclusion before scaling,
pathway deduplication, ambiguous mappings, PPI thresholds/edge multiplicity,
label independence and source locks. The optional snapshot test rebuilds all
canonical artifacts offline and compares output hashes, including the pinned
16,786 rows, 42 retained positives and eight related pathways. These counts
apply to this snapshot, not arbitrary future inputs.

The recorded cleanup validation passed **26 preprocessing tests**, including
replay of all 11 exported artifacts. The pinned-upstream regression suite passed
**103 tests**, with two optional checks skipped. Full loader/model checks remain
in `pytest`; feature validation alone does not establish predictive performance.
