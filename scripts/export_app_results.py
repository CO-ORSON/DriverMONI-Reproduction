"""Export an optional app snapshot from committed published results or full reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.app_results import DEFAULT_REPORT_FILENAMES, export_summary, load_published_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published-dir", type=Path, default=PROJECT_ROOT / "results/published",
                        help="Primary source: committed compact Original/A/B/C/D results")
    for experiment in DEFAULT_REPORT_FILENAMES:
        parser.add_argument(f"--{experiment.lower()}", type=Path,
                            help=f"Optional full experiment {experiment} report; explicit report flags use legacy raw export")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "app_assets/experiment_summary.json")
    parser.add_argument("--feature-sha256", help="Require this updated-data feature SHA-256; Original stays separate")
    arguments = parser.parse_args()
    sources = {name: getattr(arguments, name.lower()) for name in DEFAULT_REPORT_FILENAMES
               if getattr(arguments, name.lower()) is not None}
    try:
        if sources:
            result = export_summary(sources, arguments.output, expected_feature_sha256=arguments.feature_sha256)
        else:
            result = load_published_experiments(arguments.published_dir)
            if arguments.feature_sha256 is not None and any(
                entry["feature_sha256"] != arguments.feature_sha256
                for entry in result["experiments"] if entry["experiment"] != "original"
            ):
                raise ValueError("Published updated-data feature SHA-256 does not match")
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Exported {len(result['experiments'])} validated experiment summaries to {arguments.output}")


if __name__ == "__main__":
    main()
