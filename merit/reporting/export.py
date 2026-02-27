"""Export experiment results to various formats."""
import json
import csv
from pathlib import Path
from typing import Dict, Any, List


def export_json(results: Dict[str, Any], path: str) -> None:
    """Export results as formatted JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)


def export_csv(
    results: Dict[str, Dict[str, Dict[str, float]]],
    path: str,
) -> None:
    """Export results as CSV for supplementary materials.

    Args:
        results: model_name -> dataset_name -> metric_name -> score
        path: Output file path
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        return

    # Flatten results
    rows = []
    for model, datasets in results.items():
        for dataset, metrics in datasets.items():
            row = {"model": model, "dataset": dataset}
            row.update(metrics)
            rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
