"""Standard-library tests for raw acquisition and scientifically relevant rules."""

import csv
import gzip
import io
import json
import pickle
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from src.preprocessing.acquisition import FileCache, open_text_source
from src.preprocessing.brca_omics import (
    DELETERIOUS_TERMS, expression_features, mutation_features, parse_gtf_genes,
    ppi_features, read_fallback, resolve_mutation_symbol, run_omics,
    standardize, tcga_sample,
)


GENES = [f"ENSG{i:011d}" for i in range(1, 6)]
TERMS = DELETERIOUS_TERMS | {"synonymous_variant", "upstream_gene_variant", "inframe_deletion"}
MAF = (
    "#version gdc-example\n"
    "Hugo_Symbol\tTumor_Sample_Barcode\tall_effects\n"
    "0\tG1\tTCGA-AA-0001-01A-11D-X-Y\tG1,missense_variant,t1;G1,missense_variant,t2;OTHER,upstream_gene_variant,t3\n"
    "1\tG1\tTCGA-AA-0001-01A\tG1,synonymous_variant,t4\n"
    "2\tG1\tTCGA-AA-0002-01A\tG1,frameshift_variant,t5\n"
    "3\tG2\tTCGA-AA-0001-01A\tG2,stop_gained,t6\n"
    "4\tG3\tTCGA-AA-0002-01A\tG3,synonymous_variant,t7\n"
)
EXPRESSION = (
    "Ensembl_ID\tTCGA-AA-0001-01A-11R-X-Y\tTCGA-AA-0002-01A\tTCGA-AA-0003-11A\n"
    + f"{GENES[0]}.2\t2\t4\t1\n"
    + f"{GENES[1]}.1\t6\t8\t3\n"
    + f"{GENES[2]}\t4\t4\t4\n"
)


class OmicsTests(unittest.TestCase):
    def mutation(self, text=MAF):
        return mutation_features(io.StringIO(text), {f"G{i+1}": {g} for i, g in enumerate(GENES)}, {}, set(GENES), TERMS)

    def test_transcript_annotations_and_sample_prevalence_have_different_denominators(self):
        rows, audit, samples, report = self.mutation()
        first = rows[0]
        self.assertEqual(first["effect_annotation_count"], 5)
        self.assertAlmostEqual(first["missense_variant"], .4)
        self.assertAlmostEqual(first["synonymous_variant"], .2)
        self.assertAlmostEqual(first["miss_ratio"], 2/3)
        self.assertEqual(first["muta_count_raw"], 2)
        self.assertEqual(rows[1]["muta_count_raw"], 1)
        self.assertEqual(rows[2]["muta_count_raw"], 0)
        self.assertEqual(report["indexed_rows"], 5)
        self.assertEqual(len(samples), 2)
        self.assertEqual(len(audit), 3)

    def test_exact_consequence_matching_and_compact_maf_rejection(self):
        with self.assertRaisesRegex(ValueError, "Unrecognized consequence"):
            self.mutation(MAF.replace("missense_variant", "invented_consequence"))
        with self.assertRaisesRegex(ValueError, "all_effects"):
            self.mutation("Hugo_Symbol\tTumor_Sample_Barcode\nG1\tTCGA-AA-0001-01A\n")

    def test_all_seven_mutation_proportions_include_noncoding_denominator(self):
        categories = ["synonymous_variant", "stop_gained", "missense_variant", "frameshift_variant",
                      "splice_region_variant", "inframe_insertion", "inframe_deletion",
                      "conservative_inframe_deletion", "start_lost", "stop_lost", "intron_variant"]
        text = "Hugo_Symbol\tTumor_Sample_Barcode\tall_effects\nG1\tTCGA-AA-0001-01A\t"
        text += ";".join(f"G1,{term},t{i}" for i, term in enumerate(categories)) + "\n"
        rows, _, _, _ = mutation_features(io.StringIO(text), {"G1": {GENES[0]}}, {}, {GENES[0]}, set(categories))
        row = rows[0]
        for column, count in [("synonymous_variant", 1), ("stop_gained", 1), ("missense_variant", 1),
                              ("frameshift_variant", 1), ("splice", 1), ("inframe", 3), ("lost_stop and start", 2)]:
            self.assertAlmostEqual(row[column], count/11, msg=column)
        self.assertEqual(row["muta_count_raw"], 1)
        self.assertEqual(row["miss_ratio"], .5)

    def test_ambiguous_symbols_excluded_and_fallback_explicit(self):
        annotation = {"MULTI": {GENES[0], GENES[1]}}
        self.assertEqual(resolve_mutation_symbol("MULTI", annotation, {"MULTI": GENES[0]})[1], "ambiguous_excluded")
        self.assertEqual(resolve_mutation_symbol("OLD", annotation, {"OLD": GENES[2]})[:2], (GENES[2], "authors_fallback"))
        self.assertEqual(resolve_mutation_symbol("UNKNOWN", annotation, {})[1], "unmapped_excluded")

    def test_expression_means_preserve_supplied_scale(self):
        rows, audit = expression_features(io.StringIO(EXPRESSION), ["01"], ["11"], "identity")
        self.assertEqual(rows[GENES[0]]["deg"], 2)
        self.assertEqual(rows[GENES[1]]["deg"], 4)
        self.assertEqual(rows[GENES[2]]["deg"], 0)
        self.assertEqual(audit[0]["sample_id"], "TCGA-AA-0001-01A")
        transformed, _ = expression_features(io.StringIO(EXPRESSION), ["01"], ["11"], "log2_plus_one")
        self.assertNotEqual(transformed[GENES[0]]["deg"], rows[GENES[0]]["deg"])

    def test_expression_missing_groups_duplicates_nonfinite_fail(self):
        for text in (EXPRESSION.replace("11A", "01B"),
                     EXPRESSION.replace("0002-01A", "0001-01A"),
                     EXPRESSION.replace("\t6\t8\t3", "\tNaN\t8\t3"),
                     EXPRESSION + f"{GENES[0]}.3\t1\t1\t1\n"):
            with self.assertRaises(ValueError):
                expression_features(io.StringIO(text), ["01"], ["11"], "identity")

    def test_gtf_gene_records_and_id_versions(self):
        gtf = f'1\tENSEMBL\tgene\t1\t9\t.\t+\t.\tgene_id "{GENES[0]}.2"; gene_name "G1"; gene_biotype "protein_coding";\n'
        symbols, genes = parse_gtf_genes(io.StringIO(gtf))
        self.assertEqual(symbols["G1"], {GENES[0]})
        self.assertEqual(genes[GENES[0]]["gene_biotype"], "protein_coding")
        with self.assertRaises(ValueError):
            tcga_sample("not-a-TCGA-barcode")

    def test_ppi_counts_before_mutation_standardization_and_retains_duplicate_edges(self):
        rows, _, _, _ = self.mutation()
        network = "entrez_id_1,entrez_id_2,LLS\n" + f"{GENES[0]},{GENES[1]},4\n" * 2 + f"{GENES[0]},{GENES[2]},3.211\n" + f"{GENES[0]},UNCONVERTED_SYMBOL,4\n"
        report = ppi_features(rows, io.StringIO(network))
        self.assertEqual(report["mutation_quartile_floor"], 1)
        self.assertEqual([r["PPI_raw"] for r in rows], [0, 2, 0])
        self.assertEqual(report["retained_edge_rows"], 3)
        self.assertEqual(report["non_ensembl_endpoint_values"], ["UNCONVERTED_SYMBOL"])
        standardize(rows, "muta_count_raw", "muta_count")
        self.assertEqual([r["muta_count_raw"] for r in rows], [2, 1, 0])

    def test_fallback_unpickling_blocks_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.pickle"
            path.write_bytes(pickle.dumps({"OLD": GENES[0]}))
            self.assertEqual(read_fallback(path), {"OLD": GENES[0]})
            path.write_bytes(pickle.dumps(Path("example")))
            with self.assertRaises(pickle.UnpicklingError):
                read_fallback(path)

    def test_compressed_inputs_and_cache_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            zipped = root / "raw.zip"
            with ZipFile(zipped, "w") as archive:
                archive.writestr("data.tsv", EXPRESSION)
            with open_text_source(zipped, {"format": "zip", "member": "data.tsv"}) as handle:
                self.assertEqual(handle.read(), EXPRESSION)
            compressed = root / "raw.gz"
            compressed.write_bytes(gzip.compress(EXPRESSION.encode()))
            with open_text_source(compressed, {"format": "gzip"}) as handle:
                self.assertEqual(handle.read(), EXPRESSION)
            cache = FileCache(root / "cache", offline=True)
            spec = {"local_path": str(compressed), "format": "gzip"}
            cache.get("expression", spec)
            lock = {"sources": list(cache.sources.values())}
            locked = FileCache(root / "cache", offline=True, lock=lock)
            locked.get("expression", spec)
            compressed.write_bytes(gzip.compress(b"changed"))
            with self.assertRaisesRegex(ValueError, "differs from lock"):
                locked.get("expression", spec)
            with self.assertRaises(FileNotFoundError):
                FileCache(root / "empty", offline=True).get("mutation", {"url": "https://example.org/maf", "format": "text"})

    def test_end_to_end_offline_has_no_labels_and_preserves_output_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = Path(__file__).resolve().parents[1] / "configs/brca_omics_sources.json"
            config = json.loads(config_path.read_text())
            contents = {
                "mutation": MAF, "expression": EXPRESSION,
                "annotation": "\n".join(f'1\tE\tgene\t1\t9\t.\t+\t.\tgene_id "{g}"; gene_name "G{i+1}"; gene_biotype "protein_coding";' for i,g in enumerate(GENES)),
                "protein_filter": "ensembl_gene_id\tgene_biotype\n" + "".join(f"{g}\tprotein_coding\n" for g in GENES),
                "consequence_terms": "SO term\n" + "\n".join(t.replace("_", " ") for t in TERMS),
            }
            for role, body in contents.items():
                path = root / (role + ".txt")
                path.write_text(body)
                config["sources"][role].update(local_path=str(path), format="text")
                config["sources"][role].pop("url")
            fallback = root / "fallback.pickle"
            fallback.write_bytes(pickle.dumps({}))
            config["sources"]["fallback_mapping"] = {"local_path": str(fallback), "format": "pickle"}
            cache = FileCache(root / "cache", offline=True)
            summary = run_omics(config, cache=cache, output_dir=root / "out")
            self.assertEqual(summary["exported_gene_rows"], 3)
            self.assertEqual(summary["mutation_expression_shared_tumor_samples"], 2)
            with (root / "out/BRCA_mutation_expression.features.csv").open() as handle:
                reader = csv.DictReader(handle)
                self.assertNotIn("label", reader.fieldnames)
                self.assertNotIn("PPI", reader.fieldnames)
            original = json.loads((root / "out/omics_manifest.json").read_text())
            run_omics(config, cache=FileCache(root / "cache", offline=True, lock=original), output_dir=root / "replay")
            replay = json.loads((root / "replay/omics_manifest.json").read_text())
            self.assertEqual(original["outputs"], replay["outputs"])
            self.assertEqual(len(original["sources"]), 6)


if __name__ == "__main__":
    unittest.main()
