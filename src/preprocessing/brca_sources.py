"""Retrieve BRCA pathway evidence without using labels to select features.

Only Python's standard library is needed. Non-pathway features can be retained
from an explicitly supplied processed CSV; this is not raw TCGA regeneration.
"""

from collections import defaultdict
import csv
from datetime import date
import io
import json
import math
from pathlib import Path
import re
import statistics

from ..feature_schema import FEATURE_SCHEMA
from .acquisition import FileCache
from .common import sha256_file, utc_now, write_csv, write_json


KEGG_ROOT = "https://rest.kegg.jp"
HGNC_ROOT = "https://storage.googleapis.com/public-download-files/hgnc/archive/archive/monthly/tsv"
PATHWAY_COLUMNS = {"related_pathway", "dir_pathway"}
ENSEMBL_PATTERN = re.compile(r"ENSG\d{11}(?:\.\d+)?")


# Compatibility name for the standalone pathway review command.
SourceCache = FileCache


def normalize_ensembl(value):
    value = value.strip()
    if not ENSEMBL_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid human Ensembl gene ID: {value!r}")
    return value.split(".")[0]


def resolve_pathway(catalogue, name):
    matches = []
    for line in catalogue.splitlines():
        if not line.strip():
            continue
        pathway, title = line.split("\t", 1)
        pathway = pathway.removeprefix("path:")
        title = title.removesuffix(" - Homo sapiens (human)")
        if title.casefold() == name.casefold():
            if not re.fullmatch(r"hsa\d{5}", pathway):
                raise ValueError("Catalogue contains a malformed human pathway ID")
            matches.append((pathway, title))
    if len(matches) != 1:
        raise ValueError(f"Expected one exact human pathway named {name!r}; found {matches}")
    pathway, title = matches[0]
    return {
        "query": name, "pathway_id": pathway, "name": title,
        "rationale": "Unique exact disease-name match in the retrieved Homo sapiens pathway catalogue; labels were not used.",
    }


def parse_related_pathways(entry, expected_id):
    section = ""
    related = {}
    entry_id = None
    for line in entry.splitlines():
        field, value = line[:12].strip(), line[12:].strip()
        if field:
            section = field
        if section == "ENTRY" and field == "ENTRY":
            entry_id = value.split()[0]
        if section == "REL_PATHWAY" and value:
            identifier, name = value.split(maxsplit=1)
            if not re.fullmatch(r"hsa\d{5}", identifier):
                raise ValueError(f"Malformed human related pathway: {identifier}")
            if identifier != expected_id:
                related[identifier] = name
    if entry_id != expected_id:
        raise ValueError(f"Retrieved entry {entry_id!r} differs from selected {expected_id}")
    if not related:
        raise ValueError("Selected BRCA entry has no REL_PATHWAY records")
    return dict(sorted(related.items()))


def parse_membership(body, pathway):
    genes = set()
    for line in body.splitlines():
        if not line.strip():
            continue
        source, target = line.split("\t")
        if source.removeprefix("path:") != pathway or not re.fullmatch(r"hsa:\d+", target):
            raise ValueError(f"Unexpected KEGG membership row: {line!r}")
        genes.add(target.split(":", 1)[1])
    if not genes:
        raise ValueError(f"No genes returned for selected pathway {pathway}")
    return sorted(genes, key=int)


def parse_hgnc(body):
    reader = csv.DictReader(io.StringIO(body), delimiter="\t")
    required = {"hgnc_id", "symbol", "status", "entrez_id", "ensembl_gene_id"}
    if not required <= set(reader.fieldnames or []):
        raise ValueError(f"HGNC snapshot missing columns: {sorted(required - set(reader.fieldnames or []))}")
    by_ncbi = defaultdict(set)
    symbols = defaultdict(set)
    rows = []
    for row in reader:
        if row["status"] != "Approved":
            continue
        ncbi = row["entrez_id"].strip()
        ensembl = {normalize_ensembl(g) for g in ENSEMBL_PATTERN.findall(row["ensembl_gene_id"])}
        if ncbi and not ncbi.isdigit():
            raise ValueError(f"Invalid HGNC NCBI identifier: {ncbi!r}")
        for gene in sorted(ensembl):
            symbols[gene].add(row["symbol"])
            if ncbi:
                by_ncbi[ncbi].add(gene)
            rows.append({
                "ncbi_gene_id": ncbi, "gene_ens": gene, "gene_symbol": row["symbol"],
                "hgnc_id": row["hgnc_id"], "locus_type": row.get("locus_type", ""),
            })
    if not by_ncbi:
        raise ValueError("HGNC snapshot contains no approved NCBI/Ensembl mappings")
    return dict(by_ncbi), dict(symbols), rows


def convert_membership(pathways, by_ncbi, symbols):
    membership, audit = [], []
    for pathway in pathways:
        for ncbi in pathway["ncbi_genes"]:
            choices = sorted(by_ncbi.get(ncbi, set()))
            status = "mapped" if len(choices) == 1 else "unmapped" if not choices else "ambiguous"
            if status == "mapped" and len(symbols.get(choices[0], set())) != 1:
                status = "ambiguous_symbol"
            audit.append({
                "pathway": pathway["pathway_id"], "role": pathway["role"],
                "ncbi_gene_id": ncbi, "mapping_status": status,
                "ensembl_matches": "|".join(choices),
            })
            if status == "mapped":
                membership.append({
                    "pathway": pathway["pathway_id"], "role": pathway["role"],
                    "kegg_gene_id": "hsa:" + ncbi, "ncbi_gene_id": ncbi,
                    "gene_ens": choices[0], "gene_symbol": next(iter(symbols[choices[0]])),
                })
    return membership, audit


def read_labels(path):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "gene_label" not in (reader.fieldnames or []):
            raise ValueError("Label CSV requires gene_label")
        genes = [normalize_ensembl(row["gene_label"]) for row in reader]
    if not genes or len(genes) != len(set(genes)):
        raise ValueError("Label list must be nonempty and unique after ID normalization")
    return genes


def read_base_features(path):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(FEATURE_SCHEMA) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Base feature CSV missing columns: {sorted(missing)}")
        rows = list(reader)
    seen = set()
    for row in rows:
        gene = normalize_ensembl(row["gene_ens"])
        if gene in seen:
            raise ValueError(f"Duplicate Ensembl gene in base features: {gene}")
        seen.add(gene)
        if not row["gene_symbol"].strip():
            raise ValueError(f"Missing gene symbol for {gene}")
        for column in FEATURE_SCHEMA[2:]:
            if not math.isfinite(float(row[column])):
                raise ValueError(f"Nonfinite base feature: {gene}, {column}")
        row["gene_ens"] = gene
    if not rows:
        raise ValueError("Base features contain no genes")
    return rows


def pathway_features(universe, membership):
    direct, related = set(), defaultdict(set)
    for row in membership:
        if row["role"] == "direct":
            direct.add(row["gene_ens"])
        else:
            related[row["gene_ens"]].add(row["pathway"])
    counts = [len(related[gene]) for gene in universe]
    mean = statistics.mean(counts)
    std = statistics.stdev(counts) if len(counts) > 1 else 0.0
    if std == 0:
        raise ValueError("Related-pathway counts have zero variance; cannot standardize this gene universe")
    return [{
        "gene_ens": gene, "dir_pathway": int(gene in direct),
        "related_pathway_count": count, "related_pathway": (count - mean) / std,
        "related_pathway_ids": "|".join(sorted(related[gene])),
    } for gene, count in zip(universe, counts)], {"mean": mean, "std": std, "ddof": 1, "gene_count": len(universe)}


def input_record(role, path):
    return {"role": role, "path": str(Path(path).resolve()), "sha256": sha256_file(path)}


def audit_cgc(path, version, labels, symbol_by_gene):
    """Report evidence under an explicit breast/somatic-or-germline policy.

    This diagnoses an externally supplied CGC export, never changes labels or
    claims the authors used the same policy.
    """
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Gene Symbol", "Tier", "Tumour Types(Somatic)", "Tumour Types(Germline)"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("CGC export needs Gene Symbol, Tier, Tumour Types(Somatic), Tumour Types(Germline)")
        records = defaultdict(list)
        for row in reader:
            records[row["Gene Symbol"]].append(row)
    result = []
    for gene in labels:
        symbol = symbol_by_gene.get(gene, "")
        hits = records.get(symbol, [])
        somatic = "|".join(sorted({row["Tumour Types(Somatic)"] for row in hits}))
        germline = "|".join(sorted({row["Tumour Types(Germline)"] for row in hits}))
        result.append({
            "gene_ens": gene, "gene_symbol": symbol, "cgc_version": version,
            "in_cgc": int(bool(hits)), "cgc_tier": "|".join(sorted({row["Tier"] for row in hits})),
            "somatic_tumour_types": somatic, "germline_tumour_types": germline,
            "breast_association_under_audit_policy": int(bool(re.search("breast", somatic + " " + germline, re.I))),
        })
    return result


def retrieve_pathways(cache, pathway_name, hgnc_snapshot):
    """Resolve the disease and retrieve its evidence, independently of labels."""
    date.fromisoformat(hgnc_snapshot)
    selection = resolve_pathway(cache.get_text(KEGG_ROOT + "/list/pathway/hsa"), pathway_name)
    direct_id = selection["pathway_id"]
    related = parse_related_pathways(cache.get_text(KEGG_ROOT + "/get/" + direct_id), direct_id)
    selection["related_pathways"] = [{"pathway_id": key, "name": name} for key, name in related.items()]
    pathways = [{"pathway_id": direct_id, "name": selection["name"], "role": "direct"}]
    pathways += [{"pathway_id": key, "name": name, "role": "related"} for key, name in related.items()]
    for pathway in pathways:
        pathway["ncbi_genes"] = parse_membership(
            cache.get_text(KEGG_ROOT + "/link/hsa/" + pathway["pathway_id"]), pathway["pathway_id"])
    mapping_url = HGNC_ROOT + "/hgnc_complete_set_" + hgnc_snapshot + ".txt"
    by_ncbi, symbols, cross_references = parse_hgnc(cache.get_text(mapping_url))
    membership, audit = convert_membership(pathways, by_ncbi, symbols)
    if not membership:
        raise ValueError("No pathway genes could be mapped to Ensembl")
    return {"selection": selection, "pathways": pathways, "membership": membership,
            "mapping_audit": audit, "cross_references": cross_references,
            "symbols": symbols, "mapping_url": mapping_url}


def run_sources(*, cache, output_dir, pathway_name, hgnc_snapshot, labels_path,
                base_features_path=None, cgc_path=None, cgc_version=None):
    date.fromisoformat(hgnc_snapshot)
    # Read and validate user inputs before starting downloads.
    labels = read_labels(labels_path)
    base = read_base_features(base_features_path) if base_features_path else None
    inputs = [input_record("supplied_positive_labels", labels_path)]
    if base_features_path:
        inputs.append(input_record("unchanged_nonpathway_base_features", base_features_path))
    if cgc_path:
        if not cgc_version:
            raise ValueError("A CGC audit requires its version")
        inputs.append(input_record("cgc_audit_export", cgc_path))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_manifest = output_dir / "source_manifest.json"
    if previous_manifest.exists():
        old_outputs = {row["file"] for row in json.loads(previous_manifest.read_text())["outputs"]}
        for optional_file, requested in (("BRCA_input_data.sources.csv", base is not None),
                                          ("cgc_label_audit.csv", cgc_path is not None)):
            if (optional_file in old_outputs) != requested:
                raise ValueError("Use a new output directory when changing full-schema or CGC audit mode; stale artifacts must not be confused with this run")
    evidence = retrieve_pathways(cache, pathway_name, hgnc_snapshot)
    selection = evidence["selection"]
    direct_id = selection["pathway_id"]
    related = selection["related_pathways"]
    pathways, mapping_url = evidence["pathways"], evidence["mapping_url"]
    membership, mapping_audit = evidence["membership"], evidence["mapping_audit"]
    symbols, cross_references = evidence["symbols"], evidence["cross_references"]
    pathway_genes = sorted({row["gene_ens"] for row in membership})
    label_set = set(labels)
    symbol_by_gene = {gene: next(iter(values)) for gene, values in symbols.items() if len(values) == 1}
    base_genes = set() if base is None else {row["gene_ens"] for row in base}
    if base is not None:
        # The supplied historical symbols remain available even for retired IDs.
        for row in base:
            symbol_by_gene.setdefault(row["gene_ens"], row["gene_symbol"])
    universe = pathway_genes if base is None else [row["gene_ens"] for row in base]
    feature_rows, normalization = pathway_features(universe, membership)
    # Pathway candidates also include genes absent from the supplied base table.
    candidate_rows, candidate_normalization = pathway_features(pathway_genes, membership)
    for row in candidate_rows:
        row.update({
            "gene_symbol": symbol_by_gene[row["gene_ens"]],
            "benchmark_label": int(row["gene_ens"] in label_set),
            "label_status": "supplied_positive" if row["gene_ens"] in label_set else "unlabelled",
            "in_base_features": "" if base is None else int(row["gene_ens"] in base_genes),
        })
    direct_genes = {row["gene_ens"] for row in membership if row["role"] == "direct"}
    related_by_gene = defaultdict(set)
    for row in membership:
        if row["role"] == "related":
            related_by_gene[row["gene_ens"]].add(row["pathway"])
    comparison = [{
        "gene_ens": gene, "gene_symbol": symbol_by_gene.get(gene, ""),
        "benchmark_label": 1, "label_status": "supplied_positive",
        "in_direct_pathway": int(gene in direct_genes),
        "related_pathway_ids": "|".join(sorted(related_by_gene[gene])),
        "in_retrieved_pathways": int(gene in set(pathway_genes)),
        "in_base_features": "" if base is None else int(gene in base_genes),
    } for gene in labels]
    mapping_issues = [row for row in mapping_audit if row["mapping_status"] != "mapped"]
    summary = {
        "direct_pathway": direct_id, "related_pathway_count": len(related),
        "hgnc_snapshot": hgnc_snapshot,
        "unique_kegg_ncbi_genes": len({g for p in pathways for g in p["ncbi_genes"]}),
        "mapped_pathway_candidate_genes": len(pathway_genes),
        "mapped_direct_pathway_genes": len(direct_genes),
        "supplied_positive_genes": len(labels),
        "positives_in_direct_pathway": len(label_set & direct_genes),
        "positives_in_any_retrieved_pathway": len(label_set & set(pathway_genes)),
        "positives_outside_retrieved_pathways": len(label_set - set(pathway_genes)),
        "unlabelled_pathway_candidates": len(set(pathway_genes) - label_set),
        "mapping_issue_membership_rows": len(mapping_issues),
        "mapping_issue_unique_ncbi_genes": len({r["ncbi_gene_id"] for r in mapping_issues}),
        "feature_gene_universe": "supplied_base_feature_rows" if base else "mapped_pathway_union_only",
        "pathway_standardization": normalization,
        "candidate_table_standardization": candidate_normalization,
        "base_feature_rows": None if base is None else len(base),
        "base_positive_genes": None if base is None else len(base_genes & label_set),
        "pathway_genes_absent_from_base": None if base is None else len(set(pathway_genes) - base_genes),
        "pathway_genes_in_base": None if base is None else len(set(pathway_genes) & base_genes),
        "full_schema_exported": base is not None,
        "label_provenance": {
            "paper_attribution": "Cancer Gene Census v95",
            "exact_original_selection_rule": "Not supplied in the inspected repository; not independently verified",
            "negative_class_interpretation": "Absent from supplied positive list; biologically unlabelled, not established non-driver",
            "cgc_audit_version": cgc_version if cgc_path else None,
        },
    }
    cgc_rows = None if not cgc_path else audit_cgc(cgc_path, cgc_version, labels, symbol_by_gene)
    if base is not None:
        by_gene = {row["gene_ens"]: row for row in feature_rows}
        for row in base:
            for column in PATHWAY_COLUMNS:
                row[column] = by_gene[row["gene_ens"]][column]
        write_csv(output_dir / "BRCA_input_data.sources.csv", FEATURE_SCHEMA, base)
    write_json(output_dir / "pathway_selection.json", selection)
    write_csv(output_dir / "gene_cross_references.csv",
              ["ncbi_gene_id", "gene_ens", "gene_symbol", "hgnc_id", "locus_type"], cross_references)
    write_csv(output_dir / "pathway_gene_membership.csv",
              ["pathway", "role", "kegg_gene_id", "ncbi_gene_id", "gene_ens", "gene_symbol"], membership)
    write_csv(output_dir / "mapping_audit.csv",
              ["pathway", "role", "ncbi_gene_id", "mapping_status", "ensembl_matches"], mapping_audit)
    write_csv(output_dir / "pathway_features.csv",
              ["gene_ens", "dir_pathway", "related_pathway_count", "related_pathway", "related_pathway_ids"], feature_rows)
    write_csv(output_dir / "pathway_candidates.csv",
              ["gene_ens", "gene_symbol", "dir_pathway", "related_pathway_count", "related_pathway",
               "related_pathway_ids", "benchmark_label", "label_status", "in_base_features"], candidate_rows)
    write_csv(output_dir / "driver_label_comparison.csv",
              ["gene_ens", "gene_symbol", "benchmark_label", "label_status", "in_direct_pathway",
               "related_pathway_ids", "in_retrieved_pathways", "in_base_features"], comparison)
    if cgc_path:
        write_csv(output_dir / "cgc_label_audit.csv", list(cgc_rows[0]), cgc_rows)
        summary["cgc_audit_breast_associated_positive_genes"] = sum(r["breast_association_under_audit_policy"] for r in cgc_rows)
    write_json(output_dir / "summary.json", summary)
    manifest = {
        "schema_version": 1, "created_at": utc_now(), "cancer": "BRCA",
        "sources": list(cache.sources.values()), "inputs": inputs,
        "mapping_version": {"provider": "HGNC dated archive", "snapshot": hgnc_snapshot,
                            "url": mapping_url, "ensembl_release": "Not specified by HGNC; not an Ensembl-102 reconstruction"},
        "policies": {
            "pathway_selection": selection["rationale"],
            "related_pathways": "Unique REL_PATHWAY entries; direct pathway excluded; no repeated PI3K-Akt count",
            "mapping": "Approved HGNC records; require one unambiguous Ensembl ID and symbol per NCBI Gene ID; report exclusions",
            "standardization": normalization,
            "labels": "Used only for comparison; not included in feature CSV or source selection",
            "nonpathway_features": "Unchanged supplied CSV values; no new mutation/expression/PPI derivation" if base else "Not generated; full schema requires --base-features",
            "loader_integration": "None; training configuration is unchanged",
        },
        "outputs": [],
    }
    # Record only this run's outputs, not arbitrary pre-existing directory files.
    names = ["pathway_selection.json", "gene_cross_references.csv", "pathway_gene_membership.csv",
             "mapping_audit.csv", "pathway_features.csv", "pathway_candidates.csv",
             "driver_label_comparison.csv", "summary.json"]
    if base:
        names.append("BRCA_input_data.sources.csv")
    if cgc_path:
        names.append("cgc_label_audit.csv")
    manifest["outputs"] = [{"file": name, "sha256": sha256_file(output_dir / name)} for name in names]
    write_json(output_dir / "source_manifest.json", manifest)
    return summary
