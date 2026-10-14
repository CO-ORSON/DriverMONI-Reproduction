"""BRCA MAF and gene-by-sample expression to auditable gene-level features."""

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import pickle
import platform
import re
import statistics

from .acquisition import FileCache, open_text_source
from ..feature_schema import FEATURE_SCHEMA
from .brca_sources import normalize_ensembl, pathway_features
from .common import sha256_file, utc_now, write_csv, write_json


MUTATION_FEATURES = (
    "synonymous_variant", "stop_gained", "missense_variant", "frameshift_variant",
    "splice", "inframe", "lost_stop and start", "muta_count", "miss_ratio",
)
DELETERIOUS_TERMS = {"stop_gained", "missense_variant", "frameshift_variant", "splice_region_variant"}


def tcga_sample(barcode):
    parts = barcode.strip().split("-")
    if len(parts) < 4 or parts[0] != "TCGA" or not re.fullmatch(r"\d{2}[A-Z]?", parts[3]):
        raise ValueError(f"Invalid TCGA sample/aliquot barcode: {barcode!r}")
    return "-".join(parts[:4]), parts[3][:2]


def parse_gtf_genes(handle):
    by_symbol, genes = defaultdict(set), {}
    for line in handle:
        if line.startswith("#"):
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) != 9:
            raise ValueError("Malformed Ensembl GTF record")
        if fields[2] != "gene":
            continue
        attrs = dict(re.findall(r'(\w+) "([^"]*)"', fields[8]))
        gene = normalize_ensembl(attrs["gene_id"])
        symbol = attrs.get("gene_name", "")
        genes[gene] = {"gene_ens": gene, "gene_symbol": symbol, "gene_biotype": attrs.get("gene_biotype", "")}
        if symbol:
            by_symbol[symbol].add(gene)
    if not genes:
        raise ValueError("No gene records in annotation GTF")
    return dict(by_symbol), genes


class PrimitiveMappingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("Fallback mapping must contain only primitive data")


def read_fallback(path):
    with Path(path).open("rb") as handle:
        value = PrimitiveMappingUnpickler(handle).load()
    if not isinstance(value, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in value.items()):
        raise ValueError("Fallback mapping must be symbol -> Ensembl-ID strings")
    return {k: normalize_ensembl(v) for k, v in value.items()}


def parse_protein_filter(handle):
    reader = csv.DictReader(handle, delimiter="\t")
    if not {"ensembl_gene_id", "gene_biotype"} <= set(reader.fieldnames or []):
        raise ValueError("Protein filter requires ensembl_gene_id and gene_biotype")
    genes = {normalize_ensembl(r["ensembl_gene_id"]) for r in reader if r["gene_biotype"] == "protein_coding"}
    if not genes:
        raise ValueError("Protein filter has no protein-coding genes")
    return genes


def parse_consequence_terms(handle):
    reader = csv.DictReader(handle, delimiter="\t")
    if "SO term" not in (reader.fieldnames or []):
        raise ValueError("Consequence vocabulary requires SO term")
    terms = {r["SO term"].replace(" ", "_") for r in reader}
    # The original preprocessing explicitly adds this Sequence Ontology category.
    terms.add("non_coding_transcript_variant")
    if not DELETERIOUS_TERMS <= terms or "synonymous_variant" not in terms:
        raise ValueError("Incomplete consequence vocabulary")
    return terms


def resolve_mutation_symbol(symbol, annotation, fallback):
    choices = sorted(annotation.get(symbol, set()))
    if len(choices) == 1:
        return choices[0], "ensembl_release_unique", choices
    if choices:
        return None, "ambiguous_excluded", choices
    if symbol in fallback:
        return fallback[symbol], "authors_fallback", [fallback[symbol]]
    return None, "unmapped_excluded", []


def mutation_features(handle, annotation, fallback, protein_genes, terms):
    """Count all transcript effects under the MAF's primary Hugo_Symbol.

    This deliberately retains repeated transcript annotations as the upstream
    code does. Consequence matching is exact and deterministic.
    """
    comments = []
    def lines():
        for line in handle:
            if line.startswith("#"):
                # Do not retain the huge full aliquot list as header metadata.
                if not line.startswith("#tumor.aliquots"):
                    comments.append(line.rstrip())
            elif line.strip():
                yield line
    reader = csv.reader(lines(), delimiter="\t")
    fields = next(reader)
    required = {"Hugo_Symbol", "Tumor_Sample_Barcode", "all_effects"}
    if not required <= set(fields):
        raise ValueError("MAF requires Hugo_Symbol, Tumor_Sample_Barcode and all_effects; a compact Xena mutation-event table is insufficient")
    counters, affected_samples, gene_symbols = defaultdict(Counter), defaultdict(set), defaultdict(set)
    resolutions, samples, builds = {}, set(), set()
    report = Counter()
    for number, values in enumerate(reader, 1):
        if len(values) == len(fields) + 1 and values[0].isdigit():
            values = values[1:]
            report["indexed_rows"] += 1
        if len(values) != len(fields):
            raise ValueError(f"MAF record {number} has incompatible column count")
        row = dict(zip(fields, values))
        if row.get("NCBI_Build"):
            builds.add(row["NCBI_Build"])
        report["variant_rows"] += 1
        sample, sample_type = tcga_sample(row["Tumor_Sample_Barcode"])
        if not 1 <= int(sample_type) <= 9:
            raise ValueError(f"Tumor MAF contains non-tumor sample: {sample}")
        samples.add(sample)
        symbol = row["Hugo_Symbol"]
        if symbol not in resolutions:
            gene, status, choices = resolve_mutation_symbol(symbol, annotation, fallback)
            resolutions[symbol] = {"gene_symbol": symbol, "gene_ens": gene or "", "mapping_status": status,
                                   "ensembl_matches": "|".join(choices), "variant_rows": 0}
        resolution = resolutions[symbol]
        resolution["variant_rows"] += 1
        gene = resolution["gene_ens"]
        if not gene:
            report["unresolved_variant_rows"] += 1
            continue
        if gene not in protein_genes:
            report["nonprotein_variant_rows"] += 1
            continue
        if not row["all_effects"]:
            report["missing_effect_variant_rows"] += 1
            continue
        deleterious = False
        for effect in row["all_effects"].split(";"):
            pieces = effect.split(",")
            if len(pieces) < 2 or not pieces[1]:
                raise ValueError(f"Malformed all_effects annotation at MAF record {number}")
            term = pieces[1]
            if term not in terms:
                raise ValueError(f"Unrecognized consequence {term!r} at record {number}; review the source vocabulary rather than silently changing its denominator")
            counters[gene][term] += 1
            report["retained_effect_annotations"] += 1
            deleterious |= term in DELETERIOUS_TERMS
        gene_symbols[gene].add(symbol)
        if deleterious:
            affected_samples[gene].add(sample)
        report["retained_variant_rows"] += 1
    output = []
    for gene in sorted(counters):
        count = counters[gene]
        total = sum(count.values())
        denominator = count["missense_variant"] + count["synonymous_variant"]
        output.append({
            "gene_ens": gene, "gene_symbol": sorted(gene_symbols[gene])[0],
            "synonymous_variant": count["synonymous_variant"] / total,
            "stop_gained": count["stop_gained"] / total,
            "missense_variant": count["missense_variant"] / total,
            "frameshift_variant": count["frameshift_variant"] / total,
            "splice": count["splice_region_variant"] / total,
            "inframe": sum(count[t] for t in ("inframe_insertion", "inframe_deletion", "conservative_inframe_deletion")) / total,
            "lost_stop and start": (count["start_lost"] + count["stop_lost"]) / total,
            "muta_count_raw": len(affected_samples[gene]),
            "miss_ratio": count["missense_variant"] / denominator if denominator else 0.0,
            "effect_annotation_count": total,
        })
    if not output:
        raise ValueError("No eligible mutation genes remain")
    report.update({"mutation_samples": len(samples), "mutation_genes_before_expression_join": len(output)})
    return output, list(resolutions.values()), samples, dict(report, maf_header=comments, genome_builds=sorted(builds))


def expression_features(handle, tumor_codes, normal_codes, transform):
    """Aggregate an Ensembl gene-by-sample matrix by TCGA sample-type codes."""
    if set(tumor_codes) & set(normal_codes):
        raise ValueError("Expression tumor and normal sample-code groups overlap")
    if transform not in ("identity", "log2_plus_one"):
        raise ValueError("Expression transform must be identity or log2_plus_one")
    reader = csv.reader((line for line in handle if line.strip()), delimiter="\t")
    fields = next(reader)
    if len(fields) < 3:
        raise ValueError("Expression input must be an Ensembl gene-by-TCGA-sample matrix")
    tumor, normal, audit, seen_samples = [], [], [], set()
    for index, barcode in enumerate(fields[1:], 1):
        sample, code = tcga_sample(barcode)
        if sample in seen_samples:
            raise ValueError(f"Duplicate expression sample after aliquot normalization: {sample}")
        seen_samples.add(sample)
        group = "excluded"
        if code in tumor_codes:
            group = "tumor"
        elif code in normal_codes:
            group = "normal"
        if group == "tumor":
            tumor.append(index)
        elif group == "normal":
            normal.append(index)
        audit.append({"input_barcode": barcode, "sample_id": sample, "sample_type": code, "group": group})
    if not tumor or not normal:
        raise ValueError("Expression comparison requires at least one tumor and one normal sample")
    output, seen_genes = {}, set()
    for number, row in enumerate(reader, 1):
        if len(row) != len(fields):
            raise ValueError(f"Expression row {number} has incompatible column count")
        gene = normalize_ensembl(row[0])
        if gene in seen_genes:
            raise ValueError(f"Duplicate expression gene after version normalization: {gene}")
        seen_genes.add(gene)
        means = []
        for indices in (tumor, normal):
            values = [float(row[i]) for i in indices]
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f"Missing or nonfinite expression for {gene}")
            if transform == "log2_plus_one":
                if min(values) < 0:
                    raise ValueError("log2_plus_one requires nonnegative input values")
                values = [math.log2(v + 1) for v in values]
            means.append(statistics.mean(values))
        output[gene] = {"gene_ens": gene, "tumor_mean": means[0], "normal_mean": means[1],
                        "deg": means[0] - means[1]}
    if not output:
        raise ValueError("Expression matrix contains no genes")
    return output, audit


def standardize(rows, raw_column, output_column):
    values = [r[raw_column] for r in rows]
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    if std == 0:
        raise ValueError(f"Zero variance in {raw_column}; cannot standardize")
    for row, value in zip(rows, values):
        row[output_column] = (value - mean) / std
    return {"mean": mean, "std": std, "ddof": 1, "gene_count": len(rows)}


def ppi_features(rows, handle):
    """Original scalar PPI formula, distinct from the unfiltered GAT graph."""
    reader = csv.DictReader(handle)
    if not {"entrez_id_1", "entrez_id_2", "LLS"} <= set(reader.fieldnames or []):
        raise ValueError("Network CSV requires mapped endpoints and LLS")
    context = {r["gene_ens"]: r for r in rows}
    values = sorted(r["muta_count_raw"] for r in rows)
    position = (len(values) - 1) * .75
    lower, upper = math.floor(position), math.ceil(position)
    threshold = int(values[lower] + (values[upper] - values[lower]) * (position - lower))
    adjacency = defaultdict(list)
    non_ensembl_endpoints = set()
    retained = 0
    for row in reader:
        score = float(row["LLS"])
        if not math.isfinite(score):
            raise ValueError("Network contains nonfinite LLS")
        a, b = row["entrez_id_1"].strip(), row["entrez_id_2"].strip()
        if not a or not b:
            raise ValueError("Network contains an empty endpoint")
        if score > 3.211:
            retained += 1
            for endpoint in (a, b):
                if not re.fullmatch(r"ENSG\d{11}", endpoint):
                    non_ensembl_endpoints.add(endpoint)
            adjacency[a].append(b)
            adjacency[b].append(a)
    for row in rows:
        active = sum(row[t] for t in ("stop_gained", "splice", "frameshift_variant", "missense_variant")) > 0
        row["PPI_raw"] = sum(context[n]["muta_count_raw"] > threshold for n in adjacency[row["gene_ens"]] if n in context) if active else 0
    return {"lls_threshold_exclusive": 3.211, "mutation_quartile_floor": threshold, "retained_edge_rows": retained,
            "non_ensembl_endpoint_values": sorted(non_ensembl_endpoints),
            "neighbor_policy": "Only neighbors present in the feature context contribute, following upstream",
            "standardization": standardize(rows, "PPI_raw", "PPI")}


def validate_config(config):
    required = {"profile", "upstream_commit", "cache_dir", "output_dir", "sources",
                "expression_tumor_codes", "expression_normal_codes", "gene_mapping_policy",
                "mutation_consequence_policy", "mutation_filter_policy", "cohort_policy", "missing_expression_policy"}
    if set(config) != required:
        raise ValueError(f"Omics configuration must contain exactly {sorted(required)}")
    required_sources = {"mutation", "expression", "annotation", "protein_filter", "fallback_mapping", "consequence_terms"}
    if set(config["sources"]) != required_sources:
        raise ValueError(f"Omics sources must contain exactly {sorted(required_sources)}")
    expected = {
        "gene_mapping_policy": "unique_ensembl_release_symbol_then_authors_fallback",
        "mutation_consequence_policy": "all_effects_exact_terms_keep_transcript_multiplicity",
        "mutation_filter_policy": "as_supplied_no_additional_filter",
        "cohort_policy": "independent_mutation_and_expression_cohorts",
        "missing_expression_policy": "exclude_and_report",
    }
    for key, value in expected.items():
        if config[key] != value:
            raise ValueError(f"Unsupported {key}: {config[key]}")
    if not config["sources"]["expression"].get("unit"):
        raise ValueError("Expression source requires an explicit unit/provenance declaration")


def run_omics(config, *, cache, output_dir, pathway_membership=None, network_edges=None, compact=False):
    """Derive features; full mode accepts membership rows and a network source spec.

    Paths remain supported by the standalone diagnostic command. Compact mode
    emits one full feature table and raw-count audits for the combined pipeline.
    """
    validate_config(config)
    if bool(pathway_membership) != bool(network_edges):
        raise ValueError("Full schema requires both pathway membership and mapped network edges")
    if compact and pathway_membership is None:
        raise ValueError("Compact export requires the complete feature schema")
    paths = {}
    for role, spec in config["sources"].items():
        print(f"Acquiring/verifying {role}...", flush=True)
        paths[role] = cache.get(role, spec)
    specs = config["sources"]
    with open_text_source(paths["annotation"], specs["annotation"]) as handle:
        annotation, annotation_genes = parse_gtf_genes(handle)
    fallback = read_fallback(paths["fallback_mapping"])
    with open_text_source(paths["protein_filter"], specs["protein_filter"]) as handle:
        protein = parse_protein_filter(handle)
    with open_text_source(paths["consequence_terms"], specs["consequence_terms"]) as handle:
        terms = parse_consequence_terms(handle)
    print("Deriving mutation features...", flush=True)
    with open_text_source(paths["mutation"], specs["mutation"]) as handle:
        mutations, mapping_audit, mutation_samples, mutation_report = mutation_features(handle, annotation, fallback, protein, terms)
    if mutation_report["genome_builds"] and mutation_report["genome_builds"] != [specs["annotation"]["assembly"]]:
        raise ValueError("MAF genome build differs from the declared annotation assembly")
    print("Deriving expression features...", flush=True)
    with open_text_source(paths["expression"], specs["expression"]) as handle:
        expression, expression_audit = expression_features(handle, config["expression_tumor_codes"],
                                                         config["expression_normal_codes"], specs["expression"]["transform"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    full = pathway_membership is not None
    previous = output / ("manifest.json" if compact else "omics_manifest.json")
    if previous.exists() and json.loads(previous.read_text())["full_schema_exported"] != full:
        raise ValueError("Choose a new output directory when changing full-schema mode")
    inputs, membership, network_spec = [], None, None
    if full:
        if isinstance(pathway_membership, list):
            membership = [dict(row) for row in pathway_membership]
        else:
            spec = {"local_path": str(Path(pathway_membership).resolve()), "format": "text"}
            paths["pathway_membership"] = cache.get("pathway_membership", spec)
            inputs.append(dict(cache.sources["pathway_membership"]))
            with paths["pathway_membership"].open(newline="") as handle:
                membership = list(csv.DictReader(handle))
        network_spec = network_edges if isinstance(network_edges, dict) else {
            "local_path": str(Path(network_edges).resolve()), "format": "text"}
        paths["network_edges"] = cache.get("network_edges", network_spec)
        inputs.append(dict(cache.sources["network_edges"]))
    joined = [dict(row, deg=expression[row["gene_ens"]]["deg"])
              for row in mutations if row["gene_ens"] in expression]
    missing_expression = [{"gene_ens": row["gene_ens"], "gene_symbol": row["gene_symbol"], "reason": "no_expression_gene"}
                          for row in mutations if row["gene_ens"] not in expression]
    if not joined:
        raise ValueError("No mutation genes have expression data")
    normalization = standardize(joined, "muta_count_raw", "muta_count")
    ppi_report, path_norm = None, None
    if full:
        if not membership or not {"gene_ens", "role", "pathway"} <= set(membership[0]):
            raise ValueError("Pathway membership requires gene_ens, role, pathway")
        membership = [dict(row, gene_ens=normalize_ensembl(row["gene_ens"])) for row in membership]
        if any(row["role"] not in ("direct", "related") for row in membership):
            raise ValueError("Invalid pathway role")
        path_rows, path_norm = pathway_features([row["gene_ens"] for row in joined], membership)
        for row, features in zip(joined, path_rows):
            row.update({key: features[key] for key in ("dir_pathway", "related_pathway", "related_pathway_count")})
        with open_text_source(paths["network_edges"], network_spec) as handle:
            ppi_report = ppi_features(joined, handle)
    tumor_samples = {r["sample_id"] for r in expression_audit if r["group"] == "tumor"}
    sample_counts = Counter(r["group"] for r in expression_audit)
    summary = {
        "profile": config["profile"], "mutation": mutation_report,
        "expression_gene_rows": len(expression), "expression_sample_groups": dict(sample_counts),
        "expression_unit": specs["expression"]["unit"], "expression_transform": specs["expression"]["transform"],
        "mutation_expression_shared_tumor_samples": len(mutation_samples & tumor_samples),
        "cohorts_matched_for_aggregation": False, "genes_missing_expression": len(missing_expression),
        "exported_gene_rows": len(joined), "full_schema_exported": full,
        "mutation_count_standardization": normalization, "ppi": ppi_report,
        "pathway_standardization": path_norm,
        "mapping_status_symbols": dict(Counter(r["mapping_status"] for r in mapping_audit)),
        "provenance_limits": [
            "Authors' expression matrix contains a subset of samples; it does not establish the full paper cohort.",
            "The default author expression unit is undocumented; values are processed without another transformation.",
            "Ambiguous symbol mappings are excluded; exact SO matching replaces upstream substring/set-order matching.",
            "No claim of raw-to-shipped-CSV identity or calibrated driver probabilities.",
        ],
    }
    names = []
    def export(name, columns, rows):
        write_csv(output / name, columns, rows)
        names.append(name)

    if compact:
        export("BRCA_input_data.csv", FEATURE_SCHEMA, joined)
        export("audit/gene_counts.csv", ["gene_ens", "effect_annotation_count", "muta_count_raw",
               "related_pathway_count", "PPI_raw"], joined)
        export("audit/expression_means.csv", ["gene_ens", "tumor_mean", "normal_mean"], expression.values())
        sample_rows = [{"source": "expression", "input_barcode": r["input_barcode"],
                        "sample_id": r["sample_id"], "sample_type": r["sample_type"],
                        "group": r["group"], "in_other_tumor_cohort": int(r["sample_id"] in mutation_samples)}
                       for r in expression_audit]
        sample_rows += [{"source": "mutation", "input_barcode": "", "sample_id": s,
                         "sample_type": tcga_sample(s)[1], "group": "tumor",
                         "in_other_tumor_cohort": int(s in tumor_samples)} for s in sorted(mutation_samples)]
        export("audit/samples.csv", list(sample_rows[0]), sample_rows)
        export("audit/mutation_mapping.csv", list(mapping_audit[0]), mapping_audit)
        export("audit/excluded_genes.csv", ["gene_ens", "gene_symbol", "reason"], missing_expression)
    else:
        export("mutation_features.raw.csv", list(mutations[0]), mutations)
        export("expression_features.csv", ["gene_ens", "tumor_mean", "normal_mean", "deg"], expression.values())
        export("mutation_mapping_audit.csv", list(mapping_audit[0]), mapping_audit)
        export("expression_sample_audit.csv", list(expression_audit[0]), expression_audit)
        export("mutation_sample_audit.csv", ["sample_id", "sample_type", "in_expression_tumor_group"],
               [{"sample_id": s, "sample_type": tcga_sample(s)[1], "in_expression_tumor_group": int(s in tumor_samples)}
                for s in sorted(mutation_samples)])
        export("excluded_genes.csv", ["gene_ens", "gene_symbol", "reason"], missing_expression)
        if not full:
            schema = ("gene_ens", "gene_symbol") + MUTATION_FEATURES + ("deg",)
            export("BRCA_mutation_expression.features.csv", schema, joined)
    if full and not compact:
        write_csv(output / "BRCA_input_data.raw_sources.csv", FEATURE_SCHEMA, joined)
        names.append("BRCA_input_data.raw_sources.csv")
    write_json(output / "summary.json", summary)
    names.append("summary.json")
    manifest = {"schema_version": 1, "created_at": utc_now(), "configuration": config,
                "sources": list(cache.sources.values()), "inputs": inputs, "full_schema_exported": full,
                "annotation_gene_records": len(annotation_genes),
                "implementation": {"python": platform.python_version(), "feature_code_sha256": sha256_file(__file__),
                                   "acquisition_code_sha256": sha256_file(Path(__file__).with_name("acquisition.py")),
                                   "pathway_code_sha256": sha256_file(Path(__file__).with_name("brca_sources.py")),
                                   "shared_schema_code_sha256": sha256_file(Path(__file__).parents[1] / "feature_schema.py"),
                                   "common_code_sha256": sha256_file(Path(__file__).with_name("common.py"))},
                "formulas": {
                    "mutation_proportions": "all_effects category annotation counts / all retained consequence annotations for that gene",
                    "splice": "splice_region_variant only, following upstream",
                    "muta_count": "distinct tumor sample-gene groups with any selected deleterious annotation; z-score after expression join, ddof=1",
                    "miss_ratio": "missense / (missense + synonymous); zero for zero denominator",
                    "deg": "mean(tumor input values after configured transform) - mean(normal input values after configured transform)",
                    "no_labels_in_features": True,
                },
                "outputs": [{"file": n, "sha256": sha256_file(output / n)} for n in names]}
    write_json(previous, manifest)
    return summary
