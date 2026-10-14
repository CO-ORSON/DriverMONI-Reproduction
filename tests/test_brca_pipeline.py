"""Analytical raw-to-feature checks for all modalities and temporary replay."""

import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from src.feature_schema import FEATURE_SCHEMA
from src.preprocessing.acquisition import FileCache
from src.preprocessing.brca import run_brca
from tests.test_brca_omics import EXPRESSION, GENES, MAF, TERMS
from tests.test_brca_sources import FixtureCache


ROOT = Path(__file__).resolve().parents[1]


def zip_bytes(member, body):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(member, body)
    return buffer.getvalue()


class PipelineTests(unittest.TestCase):
    def fixture(self):
        config = json.loads((ROOT / "configs/brca_preprocessing.json").read_text())
        omics = json.loads((ROOT / config["omics_config"]).read_text())
        pathways = json.loads((ROOT / config["pathway_config"]).read_text())
        bodies = {
            "mutation": MAF, "expression": EXPRESSION,
            "annotation": "\n".join(f'1\tE\tgene\t1\t9\t.\t+\t.\tgene_id "{g}.2"; gene_name "G{i+1}"; gene_biotype "protein_coding";' for i,g in enumerate(GENES)),
            "protein_filter": "ensembl_gene_id\tgene_biotype\n" + "".join(f"{g}\tprotein_coding\n" for g in GENES),
            "consequence_terms": "SO term\n" + "\n".join(sorted(TERMS)),
        }
        payloads = {url: body.encode() for url, body in FixtureCache().responses.items()}
        for role, spec in omics["sources"].items():
            if role == "fallback_mapping":
                payload = pickle.dumps({})
            elif spec["format"] == "zip":
                payload = zip_bytes(spec["member"], bodies[role])
            elif spec["format"] == "gzip":
                payload = gzip.compress(bodies[role].encode())
            else:
                payload = bodies[role].encode()
            payloads[spec["url"]] = payload
        network = "entrez_id_1,entrez_id_2,LLS\n" + f"{GENES[0]},{GENES[1]},4\n" * 2
        network += f"{GENES[0]},{GENES[2]},3.211\n{GENES[0]},UNCONVERTED_SYMBOL,4\n"
        payloads[config["network"]["url"]] = zip_bytes(config["network"]["member"], network)
        payloads[config["labels"]["url"]] = f"gene_label\n{GENES[0]}\n{GENES[3]}\n".encode()
        return config, omics, pathways, payloads

    def run_fixture(self, root, configs, payloads, **options):
        def download(cache, url, destination, *, expected_sha=None):
            body = payloads[url]
            digest = hashlib.sha256(body).hexdigest()
            if expected_sha is not None and digest != expected_sha:
                raise ValueError("Downloaded source differs from lock")
            destination.write_bytes(body)
            return {"url": url, "sha256": digest, "bytes": len(body)}
        with patch.object(FileCache, "_download", download):
            return run_brca(*configs, cache=FileCache(root / "cache", **options), output_dir=root / "out")

    def test_all_thirteen_features_follow_hand_calculated_raw_evidence(self):
        config, omics, pathways, payloads = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = self.run_fixture(root, (config, omics, pathways), payloads)
            with (root / "out/BRCA_input_data.csv").open() as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, list(FEATURE_SCHEMA))
                rows = list(reader)
            self.assertEqual([r["gene_ens"] for r in rows], GENES[:3])
            # G1: five transcript effects (two missense), two affected samples.
            # G2: one stop gain / one sample; G3: synonymous / zero samples.
            expected = [
                [.2, 0, .4, .2, 0, 0, 0, 2, 1, 1, 1, 2/3, -1/math.sqrt(3)],
                [0, 1, 0, 0, 0, 0, 0, 4, -1, 1, 0, 0, 2/math.sqrt(3)],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1/math.sqrt(3)],
            ]
            for row, values in zip(rows, expected):
                for column, value in zip(FEATURE_SCHEMA[2:], values):
                    self.assertAlmostEqual(float(row[column]), value, msg=f"{row['gene_ens']} {column}")
            with (root / "out/audit/gene_counts.csv").open() as handle:
                counts = list(csv.DictReader(handle))
            self.assertEqual([int(r["related_pathway_count"]) for r in counts], [2, 0, 1])
            self.assertEqual([int(r["PPI_raw"]) for r in counts], [0, 2, 0])
            self.assertEqual(summary["ppi"]["mutation_quartile_floor"], 1)
            self.assertEqual(summary["pathways"]["related_pathway_count"], 2)
            self.assertEqual(summary["labels"]["positives_missing_from_features"], [GENES[3]])
            manifest = json.loads((root / "out/manifest.json").read_text())
            self.assertEqual(len(manifest["sources"]), 14)
            self.assertFalse((root / "out/BRCA_mutation_expression.features.csv").exists())
            self.assertFalse((root / "out/gene_cross_references.csv").exists())
            selection = json.loads((root / "out/references/pathway_selection.json").read_text())
            self.assertIn("exact disease-name", selection["rationale"])
            self.assertEqual(selection["pathway_id"], "hsa05224")

    def test_locked_offline_replay_is_identical_and_labels_cannot_change_features(self):
        config, omics, pathways, payloads = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_fixture(root, (config, omics, pathways), payloads)
            original = json.loads((root / "out/manifest.json").read_text())
            with patch.object(FileCache, "_download", side_effect=AssertionError("Offline attempted network")):
                run_brca(config, omics, pathways, cache=FileCache(root / "cache", offline=True, lock=original),
                         output_dir=root / "replay")
            replay = json.loads((root / "replay/manifest.json").read_text())
            self.assertEqual(original["outputs"], replay["outputs"])
            payloads[config["labels"]["url"]] = f"gene_label\n{GENES[2]}\n".encode()
            with self.assertRaisesRegex(ValueError, "differs from lock"):
                self.run_fixture(root, (config, omics, pathways), payloads, refresh=True, lock=original)
            self.run_fixture(root, (config, omics, pathways), payloads, refresh=True)
            changed = json.loads((root / "out/manifest.json").read_text())
            self.assertEqual(original["outputs"][0], changed["outputs"][0])
            self.assertNotEqual(original["outputs"][-1], changed["outputs"][-1])

    def test_missing_expression_gene_is_excluded_before_normalization_and_ppi(self):
        config, omics, pathways, payloads = self.fixture()
        extra = "".join(f"G4\tTCGA-AA-{i:04d}-01A\tG4,missense_variant,t1\n" for i in (1, 2, 3))
        payloads[omics["sources"]["mutation"]["url"]] = zip_bytes(omics["sources"]["mutation"]["member"], MAF + extra)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = self.run_fixture(root, (config, omics, pathways), payloads)
            self.assertEqual(summary["genes_missing_expression"], 1)
            self.assertEqual(summary["mutation_count_standardization"]["mean"], 1)
            self.assertEqual(summary["mutation_count_standardization"]["std"], 1)
            self.assertEqual(summary["ppi"]["mutation_quartile_floor"], 1)
            with (root / "out/audit/excluded_genes.csv").open() as handle:
                self.assertEqual(next(csv.DictReader(handle))["gene_ens"], GENES[3])


if __name__ == "__main__":
    unittest.main()
