"""Run without ML dependencies: python3 -m unittest tests.test_brca_sources."""

import csv
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch

from src.preprocessing.brca_sources import (
    FEATURE_SCHEMA, HGNC_ROOT, KEGG_ROOT, SourceCache, audit_cgc,
    convert_membership, parse_hgnc, parse_membership, parse_related_pathways,
    pathway_features, read_labels, resolve_pathway, run_sources,
)


GENES = [f"ENSG{index:011d}" for index in range(1, 5)]
CATALOGUE = "hsa05224\tBreast cancer - Homo sapiens (human)\nhsa05200\tPathways in cancer - Homo sapiens (human)\n"
ENTRY = (
    "ENTRY       hsa05224                    Pathway\n"
    "NAME        Breast cancer\n"
    "REL_PATHWAY hsa04151  PI3K-Akt signaling pathway\n"
    "            hsa04151  PI3K-Akt signaling pathway\n"
    "            hsa04115  p53 signaling pathway\n"
    "KO_PATHWAY  ko05224\n///\n"
)
HGNC = (
    "hgnc_id\tsymbol\tstatus\tentrez_id\tensembl_gene_id\n"
    + "\n".join(f"HGNC:{i}\tG{i}\tApproved\t{i}\t{GENES[i-1]}" for i in range(1, 5)) + "\n"
)


class FixtureCache:
    def __init__(self):
        self.sources = {}
        self.responses = {
            KEGG_ROOT + "/list/pathway/hsa": CATALOGUE,
            KEGG_ROOT + "/get/hsa05224": ENTRY,
            KEGG_ROOT + "/link/hsa/hsa05224": "path:hsa05224\thsa:1\npath:hsa05224\thsa:2\n",
            KEGG_ROOT + "/link/hsa/hsa04151": "path:hsa04151\thsa:1\npath:hsa04151\thsa:3\n",
            KEGG_ROOT + "/link/hsa/hsa04115": "path:hsa04115\thsa:1\n",
            HGNC_ROOT + "/hgnc_complete_set_2026-10-06.txt": HGNC,
        }

    def get_text(self, url):
        text = self.responses[url]
        self.sources[url] = {"url": url, "sha256": hashlib.sha256(text.encode()).hexdigest()}
        return text


class SourceTests(unittest.TestCase):
    def test_disease_name_resolution_and_ambiguity(self):
        self.assertEqual(resolve_pathway(CATALOGUE, "breast cancer")["pathway_id"], "hsa05224")
        with self.assertRaises(ValueError):
            resolve_pathway(CATALOGUE + CATALOGUE, "Breast cancer")
        with self.assertRaises(ValueError):
            resolve_pathway(CATALOGUE, "BRCA")

    def test_related_section_continuations_and_deduplication(self):
        related = parse_related_pathways(ENTRY, "hsa05224")
        self.assertEqual(set(related), {"hsa04115", "hsa04151"})
        with self.assertRaises(ValueError):
            parse_related_pathways(ENTRY, "hsa05215")
        with self.assertRaises(ValueError):
            parse_related_pathways(ENTRY.replace("hsa04151", "has04151"), "hsa05224")

    def test_membership_orientation_validation(self):
        self.assertEqual(parse_membership("path:hsa05224\thsa:1\npath:hsa05224\thsa:1\n", "hsa05224"), ["1"])
        for invalid in ("hsa:1\tpath:hsa05224", "path:hsa05215\thsa:1", ""):
            with self.assertRaises(ValueError):
                parse_membership(invalid, "hsa05224")

    def test_ambiguous_and_unmapped_identifiers_are_reported_not_chosen(self):
        mapping, symbols, _ = parse_hgnc(HGNC + f"HGNC:99\tALT\tApproved\t1\t{GENES[3]}\n")
        membership, audit = convert_membership([
            {"pathway_id": "hsa05224", "role": "direct", "ncbi_genes": ["1", "2", "99"]}
        ], mapping, symbols)
        self.assertEqual([row["ncbi_gene_id"] for row in membership], ["2"])
        self.assertEqual([row["mapping_status"] for row in audit], ["ambiguous", "mapped", "unmapped"])

    def test_pathway_counts_use_sets_and_normalize_over_all_candidate_genes(self):
        membership = [
            {"role": "direct", "gene_ens": GENES[0], "pathway": "hsa05224"},
            {"role": "related", "gene_ens": GENES[0], "pathway": "hsa04151"},
            {"role": "related", "gene_ens": GENES[0], "pathway": "hsa04151"},
            {"role": "related", "gene_ens": GENES[1], "pathway": "hsa04151"},
            {"role": "related", "gene_ens": GENES[1], "pathway": "hsa04115"},
        ]
        rows, normalization = pathway_features(GENES, membership)
        self.assertEqual([row["related_pathway_count"] for row in rows], [1, 2, 0, 0])
        self.assertEqual([row["dir_pathway"] for row in rows], [1, 0, 0, 0])
        self.assertEqual(normalization["ddof"], 1)
        self.assertAlmostEqual(statistics.mean(row["related_pathway"] for row in rows), 0)
        self.assertAlmostEqual(statistics.stdev(row["related_pathway"] for row in rows), 1)
        with self.assertRaises(ValueError):
            pathway_features([GENES[2], GENES[3]], membership)

    def test_cache_offline_integrity_and_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = SourceCache(directory)
            url = KEGG_ROOT + "/list/pathway/hsa"
            def download(source_url, destination, *, expected_sha=None):
                destination.write_bytes(CATALOGUE.encode())
                return {"url": source_url, "sha256": hashlib.sha256(CATALOGUE.encode()).hexdigest(),
                        "bytes": len(CATALOGUE.encode())}
            with patch.object(cache, "_download", side_effect=download):
                self.assertEqual(cache.get_text(url), CATALOGUE)
            lock = {"sources": list(cache.sources.values())}
            replay = SourceCache(directory, offline=True, lock=lock)
            with patch.object(replay, "_download", side_effect=AssertionError("Offline attempted network")):
                self.assertEqual(replay.get_text(url), CATALOGUE)
            with self.assertRaises(ValueError):
                replay.get_text(KEGG_ROOT + "/get/hsa05224")
            for payload in Path(directory).glob("*.bin"):
                payload.write_text("corruption")
            with self.assertRaises(ValueError):
                SourceCache(directory, offline=True).get_text(url)

    def test_missing_offline_source_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                SourceCache(directory, offline=True).get_text(KEGG_ROOT + "/get/hsa05224")

    def test_duplicate_normalized_labels_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "labels.csv"
            labels.write_text(f"gene_label\n{GENES[0]}\n{GENES[0]}.2\n")
            with self.assertRaises(ValueError):
                read_labels(labels)

    def test_full_export_preserves_other_modalities_and_labels_do_not_affect_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "labels.csv"
            labels.write_text(f"gene_label\n{GENES[0]}\n{GENES[3]}\n")
            base = root / "base.csv"
            original = []
            with base.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FEATURE_SCHEMA)
                writer.writeheader()
                for index, gene in enumerate(GENES):
                    row = dict.fromkeys(FEATURE_SCHEMA, "0.123456789")
                    row.update(gene_ens=gene, gene_symbol=f"G{index+1}")
                    original.append(row)
                    writer.writerow(row)
            summary = run_sources(cache=FixtureCache(), output_dir=root / "out", pathway_name="Breast cancer",
                                  hgnc_snapshot="2026-10-06", labels_path=labels, base_features_path=base)
            self.assertEqual(summary["supplied_positive_genes"], 2)
            self.assertEqual(summary["positives_in_any_retrieved_pathway"], 1)
            self.assertEqual(summary["unlabelled_pathway_candidates"], 2)
            with (root / "out/BRCA_input_data.sources.csv").open() as handle:
                exported = list(csv.DictReader(handle))
            for before, after in zip(original, exported):
                for key in FEATURE_SCHEMA:
                    if key not in {"dir_pathway", "related_pathway"}:
                        self.assertEqual(before[key], after[key])
            self.assertEqual(set(exported[0]), set(FEATURE_SCHEMA))
            labels.write_text(f"gene_label\n{GENES[1]}\n")
            run_sources(cache=FixtureCache(), output_dir=root / "changed_labels", pathway_name="Breast cancer",
                        hgnc_snapshot="2026-10-06", labels_path=labels, base_features_path=base)
            self.assertEqual((root / "out/BRCA_input_data.sources.csv").read_bytes(),
                             (root / "changed_labels/BRCA_input_data.sources.csv").read_bytes())
            with (root / "out/pathway_candidates.csv").open() as handle:
                candidates = list(csv.DictReader(handle))
            self.assertEqual({r["label_status"] for r in candidates}, {"supplied_positive", "unlabelled"})
            manifest = json.loads((root / "out/source_manifest.json").read_text())
            self.assertEqual(len(manifest["sources"]), 6)
            with self.assertRaisesRegex(ValueError, "new output directory"):
                run_sources(cache=FixtureCache(), output_dir=root / "out", pathway_name="Breast cancer",
                            hgnc_snapshot="2026-10-06", labels_path=labels)

    def test_source_only_mode_does_not_fabricate_full_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "labels.csv"
            labels.write_text(f"gene_label\n{GENES[0]}\n")
            summary = run_sources(cache=FixtureCache(), output_dir=root / "out", pathway_name="Breast cancer",
                                  hgnc_snapshot="2026-10-06", labels_path=labels)
            self.assertFalse(summary["full_schema_exported"])
            self.assertFalse((root / "out/BRCA_input_data.sources.csv").exists())
            self.assertEqual(summary["feature_gene_universe"], "mapped_pathway_union_only")

    def test_optional_cgc_audit_reports_evidence_without_relabelling(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cgc.csv"
            path.write_text("Gene Symbol,Tier,Tumour Types(Somatic),Tumour Types(Germline)\nG1,1,breast,\nG2,2,lung,\n")
            rows = audit_cgc(path, "v95", GENES[:3], {g: f"G{i+1}" for i, g in enumerate(GENES)})
            self.assertEqual([r["in_cgc"] for r in rows], [1, 1, 0])
            self.assertEqual([r["breast_association_under_audit_policy"] for r in rows], [1, 0, 0])


if __name__ == "__main__":
    unittest.main()
