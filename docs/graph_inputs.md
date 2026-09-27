# Graph-input reproduction and verification

Source: `BML-cbnu/DriverMONI` commit
`ddb0a830158ba41c90019bd5ca92054c2ba0869b`,
`CV_multimodal_compare.py`, lines 389-513 and 1093-1177.

## Input contract

Each classification example is one target Ensembl gene ID. `GeneSample` contains:

| Attribute | Meaning |
| --- | --- |
| `gene_features` | Float32 target features `[12]`. |
| `graph` | DGL ego graph, including the target and feature-bearing neighbors. |
| `label` | Integer 1 if in the supplied cancer driver list, otherwise 0. |
| `gene_id` | Target Ensembl ID. |
| `node_gene_ids` | Gene IDs in actual local graph-node order. |

The exact DNN column order is:

```text
synonymous_variant, stop_gained, missense_variant, frameshift_variant,
splice, inframe, lost_stop and start, deg, related_pathway,
dir_pathway, muta_count, miss_ratio
```

The GAT order is those same columns followed by `PPI`. Both constants and length
assertions are in `src/dataset.py`. Features are read as supplied, converted to
float32 as upstream does, and never standardized again. `deg` means differential
expression, not network degree.

## Graph semantics

1. Construct global nodes from all available feature genes plus both CSV endpoints.
2. Add one synthetic self-loop for every global node.
3. Append every HumanNet CSV row as its first-to-second endpoint directed edge.
   Preserve duplicate CSV rows and any CSV self-loops in addition to synthetic
   loops. Do not add reverse edges or simplify the graph.
4. Load the supplied target-neighbor dictionary by default. With explicit
   regeneration, use radius-one neighborhoods in the undirected CSV graph. This
   matches the node sets from the authors' `nx.ego_graph(..., radius=1)` fallback.
5. Intersect neighbors with genes having features. Always include the target,
   even if its cache entry is empty. Extract the induced graph, keeping edges
   between neighbors as well as target-neighbor edges.
6. Attach features by the actual returned parent-node IDs, so reindexing cannot
   silently misalign feature rows. Mark exactly one boolean `is_target` node.

The unfiltered v3 edge table is used: LLS scores are not edge attributes consumed
by the model and no threshold is applied. The precomputed scalar `PPI` column is
a separate input; its preprocessing threshold does not filter this graph.

The source's five anomalously empty BRCA cache entries are preserved by default.
Rebuilding neighborhoods can change their graphs, so it requires the explicit
`use_supplied_neighbors=False` option or debug flag `--rebuild-neighbors`.
There is no silent fallback for missing cache entries.

## Filtering and fold context

The loader reproduces upstream ordering: deduplicate by symbol, resolve the
comparison list using an Ensembl lookup from the original feature table, intersect
the resulting symbols with `gene_info_for_GOSemSim.csv`, retain original row order,
and assign labels from `BRCA_label.csv`. Missing comparison IDs raise an error.

The pinned inputs produce 15,028 targets, including 42 positives. Fold-specific
datasets should restrict target IDs while sharing this entire filtered context
table, PPI graph, and neighbor cache. A held-out target may appear as a neighbor
in a training graph, as upstream permits, but labels are never node features.

## Engineering differences

- `id` becomes `is_target`; GATEncoder was updated accordingly. No legacy-key alias
  is added to new data. The original oracle uses a test-only adapter.
- `idx` becomes `global_node_id`; `direct_con` becomes `is_target_edge`. The last
  field is the same edge-touching-target flag and remains unused by attention.
- Stable sorted global/neighbor IDs replace nondeterministic set order. A target
  is requested first, but masking and feature attachment use returned IDs rather
  than assuming DGL preserves that position. Graph comparisons use biological
  IDs and edge multiplicity, not raw numeric node/edge order. Floating-point sums
  may differ at roundoff level under a changed neighbor order.
- Modern DGL graph constructors and constant-time gene lookups replace deprecated
  constructors and repeated linear list searches. No graph semantics change.
- The clean sample returns an integer label, equivalent to upstream `one_hot[1]`
  consumed by its loss, and typed records instead of positional tuples.
- Batching stays on CPU until an explicit `GeneBatch.to(device)` call. No CUDA
  device is hard-coded in data loading.
- The loader reads the HumanNet ZIP directly when necessary, and reads the cache
  from its shipped `PPI_graph/` location instead of the broken `string/` path.
- Raw preprocessing is excluded. Training consumes these processed graph inputs.

## Verification on the authors' inputs

Command from the project root (replace the data root for another machine):

```bash
DGLBACKEND=pytorch .venv/bin/python -m pytest -q tests --brca-data-root /path/to/DriverMONI
```

Result on Python 3.12 / DGL 2.2.0 / PyTorch 2.2.2, macOS CPU:
At the graph-input implementation stage: **41 passed, 1 skipped** (CUDA unavailable).
The full suite now also includes training tests; see the main README.

The independent oracle loads the authors' dataset class from a separately
obtained, hash-verified local checkout through `tests/upstream_reference.py`.
No verbatim class bodies are redistributed. Its documented compatibility changes are modern DGL
constructor equivalents and a cached `list.index` lookup over unique nodes;
neighborhood, edge, feature, mask, and label logic are unchanged. An independent
transcription of the upstream BRCA filter also matches the entire context table.

Each of the following examples matched the oracle exactly for target identity,
label, retained gene set, directed edge multiset (including multiplicity), DNN
features, GAT feature rows by gene ID, and target mask:

| Target | Ensembl ID | Nodes | Edges | DNN shape | Node-feature shape | Label |
| --- | --- | ---: | ---: | --- | --- | ---: |
| TSPAN6 | ENSG00000000003 | 46 | 109 | `[12]` | `[46,13]` | 0 |
| BRCA1 | ENSG00000012048 | 491 | 3,398 | `[12]` | `[491,13]` | 1 |
| TP53 | ENSG00000141510 | 473 | 3,620 | `[12]` | `[473,13]` | 1 |
| MYCBP2 | ENSG00000005810 | 1 | 1 | `[12]` | `[1,13]` | 0 |
| RC3H2 | ENSG00000056586 | 1 | 1 | `[12]` | `[1,13]` | 0 |

RC3H2 is an example of a preserved empty cache despite having network neighbors.
Toy tests additionally exercise duplicate directed edges, an existing CSV loop,
edges between neighbors, unavailable neighbor features, shared fold context,
batch/model integration, independent samples, and invalid data.

## Debug command

```bash
DGLBACKEND=pytorch .venv/bin/python -m scripts.debug_brca --data-root /path/to/DriverMONI
DGLBACKEND=pytorch .venv/bin/python -m scripts.debug_brca --data-root /path/to/DriverMONI --gene ENSG00000012048
```

The first command selects an ordinary target, a positive target, and an empty-cache
target when available. Use repeated `--gene` arguments for specific Ensembl IDs.
It prints all six requested fields. No files in the upstream checkout are changed.
