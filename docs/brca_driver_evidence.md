# BRCA driver evidence

This catalogue links acquired oncology evidence to gene review status. It keeps
the original 42 DriverMONI positives and feature tables unchanged; it does not
export replacement training labels or confirmed negative genes.

```bash
python3 scripts/prepare_brca_driver_evidence.py --offline
python3 -m unittest tests.test_brca_driver_evidence -v
python3 scripts/prepare_brca_driver_evidence.py --offline \
  --lock-manifest data/brca_driver_evidence/manifest.json --output-dir /tmp/brca-evidence-replay
```

The source config pins download hashes. Retain `data/cache/` with the manifest:
live API exports may change, so a new download must match the recorded snapshot.
Canonical BRCA features must already exist under `data/brca/`; their IDs provide
coverage context and their values never enter evidence decisions.

## Two linked tables

| File under `data/brca_driver_evidence/` | Grain and purpose |
| --- | --- |
| `brca_driver_evidence.csv` | One source assertion per gene/entity, biological setting and cohort/study. Includes original source payload, version, checksum, mapping, citations, filtering and overlap flags. |
| `brca_gene_status.csv` | One stable Ensembl gene key, plus explicitly unresolved source entities. Summarizes evidence links, benchmark label, feature coverage, BRCA scope and review status. |

`gene_key` joins the tables; `evidence_ids` identifies the supporting rows.
Unmapped entities have blank `gene_ens`, an `UNRESOLVED_SYMBOL:` key and
`entity_type=unresolved_source_entity`. They remain visible without being
assigned a biological gene or driver status. Use `brca_evidence_scope` to
distinguish breast-specific evidence from broad pan-cancer/panel membership.

## Sources and qualifications

- **CGC v95 surrogate:** at the user's request, the 42 supplied DriverMONI IDs
  represent the paper-attributed BRCA subset because the full CGC export is
  unavailable. Their exact extraction, evidence tiers and somatic/germline
  scope remain unverified. These are not a downloaded CGC v95 export.
- **OncoKB:** public cancer-gene and curated-gene exports, with server metadata
  reporting **v7.6, 2026-09-30**. Somatic and germline records are separate.
  Broad list membership, embedded CGC v99 membership, summary keyword matches
  and therapeutic levels do not establish somatic BRCA driver status. Citations
  in gene backgrounds are not individually adjudicated as breast evidence.
  The alteration/tumour-level export requires authorization and was not acquired.
- **IntOGen:** exact `CANCER_TYPE=BRCA` selects 1,338 assertions across 14
  cohorts. Accepting `DRIVER=True` **and** `FILTER=PASS`, checked against the
  compendium, retains 287 assertions for 118 genes. The other 1,051 assertions
  remain review evidence rather than negative labels. Primary/metastatic,
  treatment, significance and method fields are preserved. Cohort counts do
  not establish patient-level independence.
- **Version discrepancy:** the IntOGen download is advertised as 2024-09-20,
  archive directories say 2024-06-18, and the driver README says 20230531.
  The manifest records all three labels and the exact bytes; it does not resolve
  the analysis date by assumption.
- **Literature:** the separate reviewed-assertion JSON records the earlier
  spot-check of seven somatic-evidence genes, three germline-risk genes and one
  context-limited negative association. Its checksum covers those recorded
  assertions, not downloaded journal webpages. This is not exhaustive curation.

Mappings use approved HGNC/NCBI cross-references dated **2026-10-06**, with
transcript checks against **Ensembl 102**. Ambiguities/conflicts remain unresolved;
a transcript missing from release 102 is recorded when HGNC independently
provides a unique gene mapping.

## Review decisions

`supported_somatic_BRCA_driver` currently requires explicitly reviewed published
somatic BRCA evidence. This covers **PTEN, RUNX1, CBFB, NF1, SF3B1, PIK3R1 and
KMT2C**, while preserving their original benchmark zeros. Database evidence
alone remains `unresolved_candidate`; genes with no acquired evidence are
`insufficient_evidence`. All statuses require review before use as new labels.
Germline breast-risk evidence occupies a separate field and never promotes a
somatic status. Every gene outside the benchmark-positive list remains
unlabelled for training purposes.

OncoKB/CGC literature overlap and possible TCGA/ICGC/PCAWG sample overlap are
explicit. Source/cohort counts are not independent-evidence votes. The existing
features cover mutation consequences and expression, with no dedicated
copy-number, fusion or germline features; strong evidence for those mechanisms
can coexist with weak feature representation.

`summary.json` reports coverage and exclusions; `manifest.json` records source
and local-input hashes and the decision policy. The companion
[audit notebook](../notebooks/brca_driver_evidence_audit.ipynb) checks source
integrity, exact source-row reconciliation, table links, label preservation and
locked offline replay.
