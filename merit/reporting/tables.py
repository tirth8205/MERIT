"""Generate LaTeX tables for paper inclusion."""
from typing import Dict, List
import numpy as np
from scipy.stats import spearmanr


def generate_results_table(
    results: Dict[str, Dict[str, Dict[str, float]]],
    caption: str = "MERIT scores across models and datasets",
    label: str = "tab:main_results",
) -> str:
    """Generate Table 1: MERIT scores across models x datasets.

    Args:
        results: Nested dict: model_name -> dataset_name -> metric_name -> score
        caption: LaTeX table caption
        label: LaTeX table label

    Returns:
        LaTeX table string
    """
    if not results:
        return ""

    # Extract all datasets and metrics
    models = list(results.keys())
    datasets = list(next(iter(results.values())).keys())
    metrics = list(next(iter(next(iter(results.values())).values())).keys())

    # Build LaTeX
    n_cols = 1 + len(datasets) * len(metrics)
    col_spec = "l" + "c" * (len(datasets) * len(metrics))

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row 1: dataset names (spanning metrics)
    header1 = "Model"
    for ds in datasets:
        header1 += f" & \\multicolumn{{{len(metrics)}}}{{c}}{{{ds}}}"
    header1 += r" \\"
    lines.append(header1)

    # Header row 2: metric names
    header2 = ""
    for ds in datasets:
        for m in metrics:
            short_name = m[:4].title()  # e.g., "Cons", "Fact", "Reas", "Alig"
            header2 += f" & {short_name}"
    header2 += r" \\"
    lines.append(r"\cmidrule(lr){2-" + str(n_cols) + "}")
    lines.append(header2)
    lines.append(r"\midrule")

    # Data rows
    for model in models:
        row = model.replace("_", r"\_")
        for ds in datasets:
            for m in metrics:
                score = results[model].get(ds, {}).get(m, 0.0)
                row += f" & {score:.2f}"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_correlation_table(
    metric_scores: Dict[str, List[float]],
    annotations: List[float],
    caption: str = "Spearman correlation with annotations",
    label: str = "tab:correlation",
) -> str:
    """Generate Table 2: Spearman correlation between metrics and annotations.

    Args:
        metric_scores: metric_name -> list of scores
        annotations: list of annotation scores (same length as each metric's scores)

    Returns:
        LaTeX table string
    """
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Method & $\rho$ & $p$-value \\")
    lines.append(r"\midrule")

    for name, scores in metric_scores.items():
        rho, p = spearmanr(scores, annotations)
        p_str = f"{p:.4f}" if p >= 0.0001 else "$<$0.0001"
        display_name = name.replace("_", r"\_")
        lines.append(f"{display_name} & {rho:.3f} & {p_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)
