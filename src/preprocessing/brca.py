"""Single BRCA raw-to-feature pipeline with source and calculation provenance."""

from collections import Counter
import json
from pathlib import Path

from ..feature_schema import FEATURE_SCHEMA
from .brca_omics import run_omics
from .brca_sources import read_base_features, read_labels, retrieve_pathways
from .common import sha256_file, write_csv, write_json


def run_brca(config, omics_config, pathway_config, *, cache, output_dir):
    """Join modalities by version-normalized Ensembl ID, never by label status."""
    output = Path(output_dir)
    evidence = retrieve_pathways(cache, pathway_config["pathway_name"], pathway_config["hgnc_snapshot"])
    membership = evidence["membership"]
    summary = run_omics(omics_config, cache=cache, output_dir=output,
                        pathway_membership=membership, network_edges=config["network"], compact=True)
    # Labels are acquired only after the features have been calculated.
    labels_path = cache.get("supplied_positive_labels", config["labels"])
    labels = read_labels(labels_path)
    rows = read_base_features(output / "BRCA_input_data.csv")
    genes = {r["gene_ens"] for r in rows}
    symbols = {r["gene_ens"]: r["gene_symbol"] for r in rows}
    direct = {r["gene_ens"] for r in membership if r["role"] == "direct"}
    related = {}
    for row in membership:
        if row["role"] == "related":
            related.setdefault(row["gene_ens"], set()).add(row["pathway"])
    comparison = [{"gene_ens": gene, "gene_symbol": symbols.get(gene, ""),
                   "label_status": "supplied_positive", "in_feature_table": int(gene in genes),
                   "in_direct_pathway": int(gene in direct),
                   "related_pathway_ids": "|".join(sorted(related.get(gene, set())))} for gene in labels]
    extra_outputs = {
        "references/pathway_selection.json": evidence["selection"],
        "references/pathway_gene_membership.csv": (
            ["pathway", "role", "kegg_gene_id", "ncbi_gene_id", "gene_ens", "gene_symbol"], membership),
        "audit/pathway_mapping.csv": (
            ["pathway", "role", "ncbi_gene_id", "mapping_status", "ensembl_matches"], evidence["mapping_audit"]),
        "audit/driver_label_comparison.csv": (list(comparison[0]), comparison),
    }
    for name, content in extra_outputs.items():
        if name.endswith(".json"):
            write_json(output / name, content)
        else:
            write_csv(output / name, *content)
    summary["pathways"] = {
        "direct_pathway": evidence["selection"]["pathway_id"],
        "related_pathway_count": len(evidence["selection"]["related_pathways"]),
        "hgnc_snapshot": pathway_config["hgnc_snapshot"],
        "mapped_candidate_genes": len({r["gene_ens"] for r in membership}),
        "direct_genes_in_feature_table": len(direct & genes),
        "mapping_status_memberships": dict(Counter(r["mapping_status"] for r in evidence["mapping_audit"])),
    }
    summary["labels"] = {
        "supplied_positive_genes": len(labels), "positives_in_feature_table": len(set(labels) & genes),
        "positives_missing_from_features": sorted(set(labels) - genes),
        "unlabelled_feature_genes": len(genes - set(labels)),
        "interpretation": "Absence from the supplied positive list means unlabelled, not a confirmed non-driver",
        "used_in_feature_construction": False,
    }
    write_json(output / "summary.json", summary)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({
        "schema_version": 2, "cancer": "BRCA", "feature_schema": list(FEATURE_SCHEMA),
        "configuration": {"pipeline": config, "omics": omics_config,
                          "pathways": pathway_config},
        "sources": list(cache.sources.values()),
        "mapping_versions": {
            "mutation_symbols": {"provider": "Ensembl GTF", "release": omics_config["sources"]["annotation"].get("release")},
            "kegg_ncbi_to_ensembl": {"provider": "HGNC dated archive", "snapshot": pathway_config["hgnc_snapshot"],
                                    "url": evidence["mapping_url"], "ensembl_release": "Not specified by HGNC"},
            "network": "Authors' pre-mapped HumanNet; original conversion provenance is not fully supplied",
        },
        "pathway_selection": evidence["selection"],
        "policies": {
            "join": "Mutation genes with expression; keep zero-membership genes; no label-based filtering",
            "related_pathways": "Distinct retrieved REL_PATHWAY identifiers, excluding the direct pathway",
            "mapping": "Require one approved HGNC Ensembl ID and symbol per NCBI Gene ID; report exclusions",
            "standardization": "ddof=1 over the final mutation-expression gene universe",
            "network": "LLS > 3.211; retain edge multiplicity; only feature-bearing neighbors contribute to PPI",
            "labels": "Supplied positives audited separately; remaining genes unlabelled",
        },
    })
    manifest["implementation"]["pipeline_code_sha256"] = sha256_file(__file__)
    names = [r["file"] for r in manifest["outputs"]] + list(extra_outputs)
    manifest["outputs"] = [{"file": name, "sha256": sha256_file(output / name)} for name in names]
    write_json(manifest_path, manifest)
    return summary
