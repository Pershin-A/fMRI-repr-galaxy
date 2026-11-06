# file: tvalue_plots.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from config import AnalysisConfig

logger = logging.getLogger(__name__)


def _prepare_valid_vectors(T_values: np.ndarray, R_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T_values, dtype=float)
    R = np.asarray(R_counts, dtype=float)

    valid = np.isfinite(T) & np.isfinite(R)

    T = T[valid]
    R = R[valid].astype(int)

    return T, R


def _build_class_masks(
    R_counts: np.ndarray,
    n_runs: int,
    config: AnalysisConfig,
) -> Dict[str, np.ndarray]:
    R = np.asarray(R_counts, dtype=int)
    rate = R / max(1, int(n_runs))

    none = R == 0
    strong = rate >= config.strong_reproducibility
    moderate = (rate >= config.moderate_reproducibility) & (rate < config.strong_reproducibility)
    weak = (rate >= config.weak_reproducibility) & (rate < config.moderate_reproducibility)

    other_positive = (R > 0) & ~(weak | moderate | strong)

    return {
        "none": none,
        "weak": weak,
        "moderate": moderate,
        "strong": strong,
        "other_positive": other_positive,
    }


def save_tvalue_distribution_plot(
    T_values: np.ndarray,
    R_counts: np.ndarray,
    n_runs: int,
    config: AnalysisConfig,
    out_path: Path,
) -> Dict[str, int]:
    T, R = _prepare_valid_vectors(T_values, R_counts)
    masks = _build_class_masks(R, n_runs, config)

    bins = int(getattr(config, "tvalue_hist_bins", 60))
    density = bool(getattr(config, "tvalue_hist_density", True))

    fig, ax = plt.subplots(figsize=(9, 6))

    plotting_order = [
        ("none", "Not reproducible (R=0)"),
        ("weak", "Weak reproducibility"),
        ("moderate", "Moderate reproducibility"),
        ("strong", "Strong reproducibility"),
    ]

    class_counts: Dict[str, int] = {}

    for key, label in plotting_order:
        values = T[masks[key]]
        class_counts[key] = int(len(values))

        if len(values) == 0:
            continue

        ax.hist(
            values,
            bins=bins,
            density=density,
            histtype="step",
            linewidth=2,
            label=f"{label}, n={len(values)}",
        )

    other_count = int(np.sum(masks["other_positive"]))
    class_counts["other_positive_below_weak"] = other_count

    ax.axvline(0.0, linestyle="--", linewidth=1, alpha=0.7)

    y_label = "Density" if density else "Count"
    ax.set_xlabel("Second-level T value")
    ax.set_ylabel(y_label)
    ax.set_title("T-value distributions by reproducibility class")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    text_lines = [
        f"Finite T values: {len(T)}",
        f"R>0 but below weak threshold: {other_count}",
    ]
    ax.text(
        0.98,
        0.98,
        "\n".join(text_lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.10),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    logger.info("Saved T-value distribution plot: %s", out_path)

    return class_counts


def save_tvalue_heatmap(
    T_values: np.ndarray,
    R_counts: np.ndarray,
    n_runs: int,
    config: AnalysisConfig,
    out_path: Path,
) -> Dict[str, float]:
    T, R = _prepare_valid_vectors(T_values, R_counts)

    if len(T) == 0:
        raise RuntimeError("No finite T values available for the heatmap")

    y_bins = int(getattr(config, "tvalue_2d_bins_y", 80))

    y_min_cfg = getattr(config, "tvalue_2d_y_min", None)
    y_max_cfg = getattr(config, "tvalue_2d_y_max", None)

    y_min = float(np.min(T)) if y_min_cfg is None else float(y_min_cfg)
    y_max = float(np.max(T)) if y_max_cfg is None else float(y_max_cfg)

    if not np.isfinite(y_min) or not np.isfinite(y_max):
        raise RuntimeError("Invalid T-value range for the heatmap")

    if y_max <= y_min:
        delta = 0.5 if y_min == 0 else abs(y_min) * 0.1 + 0.5
        y_min -= delta
        y_max += delta

    # X bins: R = 0, 1, ..., n_runs.
    x_edges = np.arange(-0.5, n_runs + 1.5, 1.0)
    y_edges = np.linspace(y_min, y_max, y_bins + 1)

    H, _, _ = np.histogram2d(R, T, bins=[x_edges, y_edges])

    fig, ax = plt.subplots(figsize=(9, 6))

    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        H.T,
        shading="auto",
        cmap="viridis",
    )

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Voxel count")

    ax.axhline(0.0, linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xlabel("Number of active runs (R)")
    ax.set_ylabel("Second-level T value")
    ax.set_title("2D histogram: T values vs active-run count")
    ax.set_xticks(np.arange(0, n_runs + 1, 1))
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    logger.info("Saved T-value heatmap: %s", out_path)

    return {
        "y_min": y_min,
        "y_max": y_max,
        "y_bins": y_bins,
        "n_total": int(len(T)),
    }


def save_tvalue_plots(
    T_values: np.ndarray,
    R_counts: np.ndarray,
    n_runs: int,
    config: AnalysisConfig,
    out_dir: Path,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    distribution_path = out_dir / "tvalue_distribution.png"
    heatmap_path = out_dir / "tvalue_heatmap.png"

    class_counts = save_tvalue_distribution_plot(
        T_values=T_values,
        R_counts=R_counts,
        n_runs=n_runs,
        config=config,
        out_path=distribution_path,
    )

    heatmap_info = save_tvalue_heatmap(
        T_values=T_values,
        R_counts=R_counts,
        n_runs=n_runs,
        config=config,
        out_path=heatmap_path,
    )

    info = {
        "distribution_plot": distribution_path.name,
        "heatmap_plot": heatmap_path.name,
        "class_counts": class_counts,
        "heatmap_info": heatmap_info,
    }

    return info