# DriverMONI preprocessing sources

This document identifies the feature sources, mapping references and unresolved
provenance. See [data/README.md](../data/README.md) for benchmark acquisition and
biological rationale, and [BRCA source preparation](brca_source_preparation.md)
for the implemented pipeline, formulas and validation.

## Source inventory

The [DriverMONI paper, Experimental Data section](https://www.researchgate.net/publication/398716583_DriverMONI_Cancer_Driver_Gene_Prediction_with_Multimodal_Deep_Learning_Integrating_Multiomics_Data_and_Condition-specific_Network_Information)
names UCSC Xena/TCGA for mutation and expression, KEGG for pathways, HumanNet
v2/v3 and STRING for network experiments, and CGC v95 for evaluation labels.
The active BRCA workflow uses the authors' raw examples and mapped HumanNet v3,
pinned to commit `ddb0a830158ba41c90019bd5ca92054c2ba0869b`.

| Input | Current source | Provenance and alternatives |
| --- | --- | --- |
| Mutation | Authors' [VarScan MAF archive](https://github.com/BML-cbnu/DriverMONI/blob/ddb0a830158ba41c90019bd5ca92054c2ba0869b/sample_data/sample_data_mutation.zip), member `BRCA.varscan.maf` | Annotated TCGA BRCA calls with a 2017 GDC snapshot in the retained header; not sequencing reads. [GDC MAF documentation](https://docs.gdc.cancer.gov/Data/File_Formats/MAF_Format/) defines the format. |
| Expression | Authors' [expression archive](https://github.com/BML-cbnu/DriverMONI/blob/ddb0a830158ba41c90019bd5ca92054c2ba0869b/sample_data/sample_data_expression.zip), member `BRCA-gene-exp.tsv` | A gene-by-sample example; historical Xena dataset ID, release and unit are undocumented. Numerical evidence of a log transform is recorded in the preparation guide. |
| Pathways | [KEGG human catalogue](https://rest.kegg.jp/list/pathway/hsa), [BRCA entry](https://rest.kegg.jp/get/hsa05224) and [gene-membership endpoint](https://rest.kegg.jp/link/hsa/hsa05224) | The entry supplies related pathways. Retrieved memberships replace the original embedded lists; snapshots and mappings are recorded. |
| Network | Authors' [mapped HumanNet archive](https://github.com/BML-cbnu/DriverMONI/blob/ddb0a830158ba41c90019bd5ca92054c2ba0869b/for_ref/humannet_v2_and_v3.zip), member `humannet_v2_and_v3/humannetv3_ens_ppi.csv` | [Official HumanNet v3 downloads](https://www.inetbio.org/humannetv3/download.php) provide a reference for fresh acquisition. The bundled counts match the PI component, but byte identity and the historical conversion remain unverified. |
| Driver labels | Authors' `using_Data/BRCA_label.csv` | The 42 positives are the paper-attributed CGC v95 subset; the full export and extraction rules are unavailable. See [driver evidence](brca_driver_evidence.md) for the qualified proxy and linked evidence tables. |

## Identifier references

Mutation symbols use the official [Ensembl 102 GRCh38 GTF](https://ftp.ensembl.org/pub/release-102/gtf/homo_sapiens/Homo_sapiens.GRCh38.102.gtf.gz),
then the authors' pinned fallback mapping. Their `hsa_id_conv.txt` supplies the
protein-coding filter, with an undocumented annotation release; `soterm_list.txt`
supplies the consequence vocabulary. [Ensembl VEP documentation](https://static.ensembl.org/info/genome/variation/prediction/predicted_data.html)
explains the underlying Sequence Ontology terms.

KEGG human gene IDs correspond to NCBI Gene IDs. The workflow maps them through
approved records in the dated [HGNC archive](https://www.genenames.org/download/archive/)
(snapshot **2026-10-06**). This cross-reference table is versioned but is not an
Ensembl-102 reconstruction. Ambiguous mappings are reported rather than silently
selecting the first ID. Sources and mapping decisions are saved in the manifest
and audits.

The network's `entrez_id_*` columns already contain mostly Ensembl IDs, plus
some unmapped symbols. The workflow preserves this mapped snapshot and reports
unresolved endpoints; a freshly downloaded network requires mapping both
endpoints and regenerating its neighbor cache.

## Remaining limits 

- **Cohort lineage:** the supplied mutation and expression examples cover
  different cohorts. Their reproducible reconstruction does not establish the
  full lineage of the shipped feature table. Snapshot counts and expression-scale
  evidence are in the [preparation guide](brca_source_preparation.md#snapshot-and-benchmark-comparison).
- **Candidate population:** the original comparison and GOSemSim annotation
  lists reduce the evaluated population to 15,028 genes, but their construction
  is undocumented. Ensembl mapping does not resolve this ascertainment issue.
- **Labels:** absence from the 42-gene list means unlabelled, not a confirmed
  negative; CGC somatic/germline and tier-selection rules remain unknown.

## Relationship to training

`scripts/preprocess_brca.py` is the implemented acquisition/preprocessing entry
point. It exports a separate canonical table; `src/dataset.py::load_brca_dataset()`
continues to read original processed benchmark inputs. Feature column names in
`src/feature_schema.py` define the model contract, not embedded gene lists.

A future switch must explicitly configure `data.features`, `data.ppi_edges`,
`data.neighbors` and the comparison-filter policy in `configs/brca.yaml`.
Changed edges require a matching neighbor cache or
`data.use_supplied_neighbors: false`. Changed gene populations require compatible
comparison/annotation filters or a documented decision to disable them. Preserve
the preprocessing manifest alongside the training input hashes. Matching feature
schemas alone does not establish benchmark parity; see the
[canonical comparison](../data/README.md#canonical-versus-original-brca-features)
and [graph contracts](graph_inputs.md).
