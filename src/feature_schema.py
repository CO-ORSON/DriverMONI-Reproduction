"""Ordered feature contract shared by source preparation and model loading."""

DNN_FEATURES = (
    "synonymous_variant", "stop_gained", "missense_variant", "frameshift_variant",
    "splice", "inframe", "lost_stop and start", "deg", "related_pathway",
    "dir_pathway", "muta_count", "miss_ratio",
)
GAT_FEATURES = DNN_FEATURES + ("PPI",)
FEATURE_SCHEMA = ("gene_ens", "gene_symbol") + GAT_FEATURES
