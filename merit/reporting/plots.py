"""Publication-quality matplotlib figures for MERIT papers."""
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

# Configure matplotlib for publication
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Publication style
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def radar_chart(
    model_scores: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "Multi-dimensional Model Profiles",
) -> None:
    """Generate radar chart showing per-model multi-dimensional profiles.

    Args:
        model_scores: model_name -> {dimension: score}
        output_path: Where to save the figure
    """
    dimensions = list(next(iter(model_scores.values())).keys())
    n_dims = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = plt.cm.Set2(np.linspace(0, 1, len(model_scores)))

    for (model, scores), color in zip(model_scores.items(), colors):
        values = [scores[d] for d in dimensions]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions)
    ax.set_ylim(0, 1)
    ax.set_title(title, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def scaling_plot(
    model_sizes: List[float],
    metric_scores: Dict[str, List[float]],
    output_path: str,
    title: str = "Metric Scores vs. Model Size",
) -> None:
    """Generate scatter plot of metrics vs model size with trend lines.

    Args:
        model_sizes: List of model sizes in billions of parameters
        metric_scores: metric_name -> list of scores (one per model)
        output_path: Where to save
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.Set1(np.linspace(0, 1, len(metric_scores)))

    for (name, scores), color in zip(metric_scores.items(), colors):
        ax.scatter(model_sizes, scores, color=color, s=60, zorder=5)
        # Trend line
        z = np.polyfit(np.log(model_sizes), scores, 1)
        x_smooth = np.linspace(min(model_sizes), max(model_sizes), 100)
        y_smooth = np.polyval(z, np.log(x_smooth))
        ax.plot(x_smooth, y_smooth, '--', color=color, alpha=0.5, label=name)

    ax.set_xlabel("Model Size (B parameters)")
    ax.set_ylabel("Score")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}B"))
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def correlation_heatmap(
    correlations: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "Metric Correlation Heatmap",
) -> None:
    """Generate heatmap of metric-to-metric correlations.

    Args:
        correlations: metric_name -> {other_metric: correlation_value}
        output_path: Where to save
    """
    try:
        import seaborn as sns
    except ImportError:
        # Fallback to plain matplotlib
        sns = None

    names = list(correlations.keys())
    n = len(names)
    matrix = np.zeros((n, n))
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            matrix[i, j] = correlations[n1].get(n2, 0.0)

    fig, ax = plt.subplots(figsize=(8, 6))

    if sns:
        sns.heatmap(matrix, annot=True, fmt=".2f", xticklabels=names,
                     yticklabels=names, cmap="RdYlGn", center=0, ax=ax,
                     vmin=-1, vmax=1)
    else:
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1, vmax=1)
        ax.set_xticks(range(n))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(names)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")
        fig.colorbar(im)

    ax.set_title(title)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def cost_comparison_bar(
    methods: List[str],
    times: List[float],
    costs: List[float],
    output_path: str,
    title: str = "Evaluation Cost Comparison",
) -> None:
    """Generate bar chart comparing heuristic vs LLM-judge cost/speed.

    Args:
        methods: Method names
        times: Time per sample in seconds
        costs: Cost per sample in USD
        output_path: Where to save
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(methods))

    ax1.bar(x, times, color=plt.cm.Set2(np.linspace(0, 1, len(methods))))
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=30, ha="right")
    ax1.set_ylabel("Time per sample (seconds)")
    ax1.set_title("Speed")

    ax2.bar(x, costs, color=plt.cm.Set2(np.linspace(0, 1, len(methods))))
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=30, ha="right")
    ax2.set_ylabel("Cost per sample (USD)")
    ax2.set_title("Cost")

    fig.suptitle(title)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
