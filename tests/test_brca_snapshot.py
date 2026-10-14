"""Optional real-source replay; uses a temporary directory, never a replay folder.

Run: BRCA_PREPROCESSING_DATA_ROOT=data python3 -m unittest tests.test_brca_snapshot -v
"""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.preprocessing.acquisition import FileCache
from src.preprocessing.brca import run_brca
from src.preprocessing.common import sha256_file


@unittest.skipUnless(os.environ.get("BRCA_PREPROCESSING_DATA_ROOT"), "Set BRCA_PREPROCESSING_DATA_ROOT for real-source replay")
class CachedSnapshotTests(unittest.TestCase):
    def test_raw_sources_recreate_every_canonical_output(self):
        root = Path(os.environ["BRCA_PREPROCESSING_DATA_ROOT"]).expanduser().resolve()
        original = json.loads((root / "brca/manifest.json").read_text())
        for artifact in original["outputs"]:
            self.assertEqual(sha256_file(root / "brca" / artifact["file"]), artifact["sha256"])
        config = original["configuration"]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(FileCache, "_download", side_effect=AssertionError("Offline attempted network")), contextlib.redirect_stdout(io.StringIO()):
                summary = run_brca(config["pipeline"], config["omics"], config["pathways"],
                                   cache=FileCache(root / "cache", offline=True, lock=original), output_dir=output)
            replay = json.loads((output / "manifest.json").read_text())
            self.assertEqual(original["outputs"], replay["outputs"])
            self.assertEqual(summary["exported_gene_rows"], 16786)
            self.assertEqual(summary["labels"]["positives_in_feature_table"], 42)
            self.assertEqual(summary["pathways"]["related_pathway_count"], 8)


if __name__ == "__main__":
    unittest.main()
