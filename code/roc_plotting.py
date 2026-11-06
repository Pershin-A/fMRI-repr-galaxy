from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from roc import EnglandEROC


# ---------------------------------------------------------------------
# Summary schema normalization
# ---------------------------------------------------------------------


def _normalize_summary_schema(contrast_summary: Dict[str, object]) -> Dict[str, object]:
    """Return an in-memory legacy-shaped view for plotting.

    Summary schema 2.0 stores each ROC exactly once under
    ``roc_models[method][direction][lambda_mode]``.  The plotting code predates
    that compact schema, so this adapter builds a transient view without
    duplicating data in ``summary.json``.
    """
    if not isinstance(contrast_summary, dict):
        return {}

    roc_models = contrast_summary.get("roc_models")
    if not isinstance(roc_models, dict):
        return contrast_summary

    direction = str(contrast_summary.get("primary_direction", "positive"))
    selected = contrast_summary.get("selected_model", {})
    if not isinstance(selected, dict):
        selected = {}

    methods: Dict[str, object] = {}
    lambda_modes_seen: List[str] = []

    for method_name, method_data in roc_models.items():
        if not isinstance(method_data, dict):
            continue
        direction_data = method_data.get(direction, {})
        if not isinstance(direction_data, dict):
            continue

        lambda_items: Dict[str, object] = {}
        for raw_mode, item in direction_data.items():
            if not isinstance(item, dict):
                continue
            mode = normalize_lambda_mode(raw_mode)
            curve = item.get("curve", {})
            if not isinstance(curve, dict):
                curve = {}
            operating_point = item.get("operating_point", {})
            if not isinstance(operating_point, dict):
                operating_point = {}

            # Keep invalid curves visible for diagnosis, but do not invent a
            # valid working point.  The plot marks an operating point only when
            # finite coordinates are present.
            threshold = dict(operating_point)
            threshold.setdefault("valid", bool(item.get("valid", False)))
            threshold.setdefault("invalid_reason", item.get("invalid_reason"))

            lambda_items[mode] = {
                "threshold": threshold,
                "roc": {
                    "lambda": item.get("lambda"),
                    "lambda_optimization_mode": mode,
                    "converged": item.get("em_converged"),
                    "n_iter": item.get("em_iterations"),
                    "optimization": item.get("optimization"),
                    "thresholds": curve.get("thresholds", []),
                    "sensitivity": curve.get("sensitivity", []),
                    "false_alarm": curve.get("false_alarm", []),
                    "kappa": curve.get("kappa", []),
                    "england_eroc": curve.get("england_eroc"),
                    "approximated": curve.get("approximated", {}),
                    "auc": item.get("auc"),
                    "stochastic_dominance_valid": item.get(
                        "stochastic_dominance_valid"
                    ),
                    "dominance_min_margin": item.get("dominance_min_margin"),
                    "valid": item.get("valid"),
                    "invalid_reason": item.get("invalid_reason"),
                },
            }
            if mode not in lambda_modes_seen:
                lambda_modes_seen.append(mode)

        if lambda_items:
            methods[str(method_name)] = {
                "lambda_modes": lambda_items,
                "primary_lambda_optimization_mode": selected.get(
                    "lambda_mode", "free"
                ),
            }

    if not methods:
        return contrast_summary

    configured = contrast_summary.get("configured_glm_method", "bayes")
    selected_method = selected.get("method", configured)
    selected_lambda = selected.get("lambda_mode", "free")

    normalized = dict(contrast_summary)
    normalized.update(
        {
            "main_glm_method": selected_method,
            "selected_glm_method": selected_method,
            "selected_lambda_mode": selected_lambda,
            "roc_lambda_optimization_modes": lambda_modes_seen,
            "methods": methods,
        }
    )
    return normalized


# ---------------------------------------------------------------------
# GLM method helpers
# ---------------------------------------------------------------------


def normalize_method_name(method: str) -> str:
    """
    Normalize only real GLM methods.

    Implementation detail.
    for the best curve/model by max Kappa, therefore it must not be passed through
    normalize_method_name().
    """
    m = str(method).lower().strip()

    aliases = {
        "bayes": "bayes",
        "bayesian": "bayes",
        "eb": "bayes",
        "empirical_bayes": "bayes",
        "ols": "ols",
        "ordinary_least_squares": "ols",
        "least_squares": "ols",
        "lsm": "ols",
        "mnk": "ols",
        
    }

    if m not in aliases:
        raise ValueError(f"Unknown GLM method: {method}")

    return aliases[m]


def is_choose_best_mode(value: object) -> bool:
    text = str(value).lower().strip()
    return text in {
        "choose_best_kappa",
        "best_kappa",
        "choose_best",
        "best",
        "auto",
    }


def _available_methods_from_summary(method_summaries: Dict[str, object]) -> List[str]:
    result: List[str] = []

    for method in ("bayes", "ols"):
        if method in method_summaries:
            result.append(method)

    for method in method_summaries.keys():
        try:
            norm = normalize_method_name(str(method))
        except Exception:
            continue

        if norm not in result:
            result.append(norm)

    return result


def methods_from_plot_mode(
    plot_mode: str,
    main_method: str,
    available_methods: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Return the list of GLM methods to plot.

    Supports legacy modes:
        glm_method / bayes / ols / both

    And the new mode:
        choose_best_kappa

    For choose_best_kappa, show both ROC curves if both are available,
    because the best model is selected between Bayes and OLS.
    """
    mode = str(plot_mode).lower().strip()
    available = [
        normalize_method_name(m)
        for m in (available_methods or [])
        if str(m).strip()
    ]

    def available_or_default(default: List[str]) -> List[str]:
        if not available:
            return default
        return [m for m in default if m in available] or available

    if mode == "glm_method":
        if is_choose_best_mode(main_method):
            return available_or_default(["bayes", "ols"])
        return available_or_default([normalize_method_name(main_method)])

    if mode == "both":
        return available_or_default(["bayes", "ols"])

    if is_choose_best_mode(mode):
        return available_or_default(["bayes", "ols"])

    return available_or_default([normalize_method_name(mode)])


def method_label(method: str) -> str:
    method = normalize_method_name(method)

    if method == "bayes":
        return "Bayes / EB"

    if method == "ols":
        return "OLS"

    return method


# ---------------------------------------------------------------------
# Lambda mode helpers
# ---------------------------------------------------------------------


def normalize_lambda_mode(mode: object) -> str:
    text = str(mode).lower().strip()

    aliases = {
        "free": "free",
        "variable": "free",
        "varying": "free",
        "estimated": "free",
        "estimate": "free",
        "lambda_free": "free",
        "fixed": "fixed",
        "constant": "fixed",
        "lambda_fixed": "fixed",
    }

    if text == "both":
        return "both"

    return aliases.get(text, text)


def lambda_mode_label(mode: str) -> str:
    mode = normalize_lambda_mode(mode)

    if mode == "free":
        return "lambda free"

    if mode == "fixed":
        return "lambda fixed"

    return f"lambda {mode}"


def _format_lambda_value(value: object) -> str:
    try:
        value = float(value)
    except Exception:
        return "?"

    if not np.isfinite(value):
        return "?"

    return f"{value:.4g}"


def _lambda_mode_items_for_method(item: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    """
    New summary format:
        method_summary["lambda_modes"][mode] = {
            "roc": ...,
            "threshold": ...
        }

    Legacy summary format:
        method_summary["roc"], method_summary["threshold"]
    """
    lambda_modes = item.get("lambda_modes", None)

    if isinstance(lambda_modes, dict) and lambda_modes:
        normalized: Dict[str, Dict[str, object]] = {}

        for mode, subitem in lambda_modes.items():
            if not isinstance(subitem, dict):
                continue

            norm_mode = normalize_lambda_mode(mode)
            if norm_mode == "both":
                continue

            if "roc" in subitem and "threshold" in subitem:
                normalized[norm_mode] = subitem

        if normalized:
            return normalized

    if "roc" in item and "threshold" in item:
        primary_mode = normalize_lambda_mode(item.get("primary_lambda_mode", "free"))
        if primary_mode == "both":
            primary_mode = "free"

        return {
            primary_mode: {
                "roc": item["roc"],
                "threshold": item["threshold"],
            }
        }

    return {}


def _get_method_summaries(contrast_summary: Dict[str, object]) -> Dict[str, object]:
    method_summaries = contrast_summary.get("methods", None)

    if isinstance(method_summaries, dict):
        normalized: Dict[str, object] = {}

        for method, item in method_summaries.items():
            try:
                normalized[normalize_method_name(str(method))] = item
            except Exception:
                continue

        if normalized:
            return normalized

    main_method = contrast_summary.get("main_glm_method", "bayes")

    if is_choose_best_mode(main_method):
        fallback_method = "bayes"
    else:
        fallback_method = normalize_method_name(str(main_method))

    if "roc" not in contrast_summary or "threshold" not in contrast_summary:
        return {}

    return {
        fallback_method: {
            "roc": contrast_summary["roc"],
            "threshold": contrast_summary["threshold"],
        }
    }


def _lambda_modes_from_summary(
    contrast_summary: Dict[str, object],
    method_summaries: Dict[str, object],
    requested_methods: List[str],
) -> List[str]:
    raw = contrast_summary.get("roc_lambda_optimization_modes", None)

    modes: List[str] = []

    if isinstance(raw, str):
        text = raw.lower().strip()
        if text == "both":
            modes = ["free", "fixed"]
        else:
            modes = [
                normalize_lambda_mode(part)
                for part in text.replace(";", ",").split(",")
                if part.strip()
            ]

    elif isinstance(raw, (list, tuple)):
        modes = [normalize_lambda_mode(x) for x in raw if str(x).strip()]

    if not modes:
        for method in requested_methods:
            item = method_summaries.get(method)
            if not isinstance(item, dict):
                continue

            for mode in _lambda_mode_items_for_method(item).keys():
                if mode not in modes:
                    modes.append(mode)

    expanded: List[str] = []
    for mode in modes:
        if mode == "both":
            for item in ("free", "fixed"):
                if item not in expanded:
                    expanded.append(item)
        else:
            if mode not in expanded:
                expanded.append(mode)

    available: List[str] = []
    for mode in expanded:
        for method in requested_methods:
            item = method_summaries.get(method)
            if isinstance(item, dict) and mode in _lambda_mode_items_for_method(item):
                available.append(mode)
                break

    return available or expanded or ["free"]


def _lambda_value_for_method_and_mode(
    method_summaries: Dict[str, object],
    method: str,
    lambda_mode: str,
) -> object:
    item = method_summaries.get(method)
    if not isinstance(item, dict):
        return None

    lambda_items = _lambda_mode_items_for_method(item)
    subitem = lambda_items.get(lambda_mode)

    if not isinstance(subitem, dict):
        return None

    roc = subitem.get("roc", {})
    if not isinstance(roc, dict):
        return None

    return roc.get("lambda", None)


def _lambda_values_for_title(
    method_summaries: Dict[str, object],
    requested_methods: List[str],
    lambda_mode: str,
) -> str:
    parts: List[str] = []

    for method in requested_methods:
        lambda_value = _lambda_value_for_method_and_mode(
            method_summaries=method_summaries,
            method=method,
            lambda_mode=lambda_mode,
        )

        if lambda_value is None:
            continue

        parts.append(f"{method_label(method)} λ={_format_lambda_value(lambda_value)}")

    return ", ".join(parts)


# ---------------------------------------------------------------------
# Selection helpers for choose_best_kappa
# ---------------------------------------------------------------------


def _safe_float(value: object, default: float = float("-inf")) -> float:
    try:
        out = float(value)
    except Exception:
        return default

    if not np.isfinite(out):
        return default

    return out


def _selected_method_and_lambda_mode(
    contrast_summary: Dict[str, object],
    method_summaries: Dict[str, object],
    requested_methods: List[str],
) -> tuple[Optional[str], Optional[str]]:
    """
    Return the pair (method, lambda_mode), for which to show
    Kappa and the operating point.

    Priority:
    1. selected_glm_method / selected_lambda_mode from summary;
    2. best_kappa_selection from summary;
    3. recompute the maximum threshold["kappa"] from method_summaries;
    Implementation detail.
    """
    for method_key in ("selected_glm_method", "selected_method", "best_glm_method"):
        raw_method = contrast_summary.get(method_key, None)
        if raw_method is None:
            continue

        try:
            method = normalize_method_name(str(raw_method))
        except Exception:
            continue

        raw_lambda = (
            contrast_summary.get("selected_lambda_mode", None)
            or contrast_summary.get("selected_lambda_optimization_mode", None)
            or contrast_summary.get("best_lambda_mode", None)
        )
        lambda_mode = normalize_lambda_mode(raw_lambda) if raw_lambda is not None else None
        if lambda_mode == "both":
            lambda_mode = None

        return method, lambda_mode

    best_info = contrast_summary.get("best_kappa_selection", None)
    if isinstance(best_info, dict):
        raw_method = (
            best_info.get("glm_method", None)
            or best_info.get("method", None)
            or best_info.get("selected_glm_method", None)
        )
        raw_lambda = (
            best_info.get("lambda_mode", None)
            or best_info.get("lambda_optimization_mode", None)
            or best_info.get("selected_lambda_mode", None)
        )

        if raw_method is not None:
            try:
                method = normalize_method_name(str(raw_method))
            except Exception:
                method = None

            lambda_mode = normalize_lambda_mode(raw_lambda) if raw_lambda is not None else None
            if lambda_mode == "both":
                lambda_mode = None

            if method is not None:
                return method, lambda_mode

    best_method: Optional[str] = None
    best_lambda_mode: Optional[str] = None
    best_kappa = float("-inf")

    for method in requested_methods:
        item = method_summaries.get(method)
        if not isinstance(item, dict):
            continue

        for lambda_mode, subitem in _lambda_mode_items_for_method(item).items():
            threshold = subitem.get("threshold", {})
            if not isinstance(threshold, dict):
                continue

            kappa = _safe_float(threshold.get("kappa", None))
            if kappa > best_kappa:
                best_kappa = kappa
                best_method = method
                best_lambda_mode = lambda_mode

    if best_method is not None:
        return best_method, best_lambda_mode

    main_method = contrast_summary.get("main_glm_method", "bayes")
    if not is_choose_best_mode(main_method):
        try:
            return normalize_method_name(str(main_method)), None
        except Exception:
            pass

    return None, None


def _should_show_auxiliary_for_curve(
    contrast_summary: Dict[str, object],
    method: str,
    lambda_mode: str,
    selected_method: Optional[str],
    selected_lambda_mode: Optional[str],
    requested_methods: List[str],
) -> bool:
    """
    Kappa/operating point are shown:
    - in choose_best_kappa: only for the selected best pair method+lambda_mode;
    - in regular multi-method mode: only for main_glm_method;
    - if exactly one method is requested: for it.
    """
    main_method_raw = contrast_summary.get("main_glm_method", "bayes")

    if is_choose_best_mode(main_method_raw):
        if selected_method is None:
            return False

        if method != selected_method:
            return False

        if selected_lambda_mode is not None and lambda_mode != selected_lambda_mode:
            return False

        return True

    if len(requested_methods) == 1:
        return True

    try:
        main_method = normalize_method_name(str(main_method_raw))
    except Exception:
        return False

    return method == main_method


# ---------------------------------------------------------------------
# Curve helpers
# ---------------------------------------------------------------------


def _roc_curve_for_plot(fp: np.ndarray, tp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Implementation detail.
    The original ROC points are not changed.
    """
    fp = np.asarray(fp, dtype=float)
    tp = np.asarray(tp, dtype=float)

    valid = np.isfinite(fp) & np.isfinite(tp)
    fp = fp[valid]
    tp = tp[valid]

    if len(fp) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    fp_full = np.concatenate([[0.0], fp, [1.0]])
    tp_full = np.concatenate([[0.0], tp, [1.0]])

    order = np.argsort(fp_full)
    fp_full = fp_full[order]
    tp_full = tp_full[order]

    return fp_full, tp_full


def _kappa_curve_for_plot(fp: np.ndarray, kappas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    For Kappa visualization, add endpoint points with kappa=0.
    """
    fp = np.asarray(fp, dtype=float)
    kappas = np.asarray(kappas, dtype=float)

    valid = np.isfinite(fp) & np.isfinite(kappas)
    fp = fp[valid]
    kappas = kappas[valid]

    if len(fp) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])

    fp_full = np.concatenate([[0.0], fp, [1.0]])
    kappa_full = np.concatenate([[0.0], kappas, [0.0]])

    order = np.argsort(fp_full)
    fp_full = fp_full[order]
    kappa_full = kappa_full[order]

    return fp_full, kappa_full


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------


def _plot_single_lambda_mode(
    ax,
    contrast_summary: Dict[str, object],
    method_summaries: Dict[str, object],
    requested_methods: List[str],
    lambda_mode: str,
    selected_method: Optional[str],
    selected_lambda_mode: Optional[str],
    show_curve: bool,
    show_points: bool,
    show_kappa: bool,
    show_optimal_point: bool,
    show_eroc: bool,
) -> None:
    ax.clear()

    ax2 = None
    if show_kappa:
        ax2 = ax.twinx()
        ax2.set_ylabel("Kappa")
        ax2.grid(False)

    for method in requested_methods:
        if method not in method_summaries:
            continue

        item = method_summaries[method]
        if not isinstance(item, dict):
            continue

        lambda_items = _lambda_mode_items_for_method(item)
        if lambda_mode not in lambda_items:
            continue

        subitem = lambda_items[lambda_mode]
        if not isinstance(subitem, dict):
            continue

        roc = subitem.get("roc", {})
        threshold = subitem.get("threshold", {})

        if not isinstance(roc, dict) or not isinstance(threshold, dict):
            continue

        if "false_alarm" not in roc or "sensitivity" not in roc:
            continue

        fp = np.asarray(roc["false_alarm"], dtype=float)
        tp = np.asarray(roc["sensitivity"], dtype=float)
        kappas = np.asarray(roc.get("kappa", []), dtype=float)

        label = method_label(method)

        show_auxiliary = _should_show_auxiliary_for_curve(
            contrast_summary=contrast_summary,
            method=method,
            lambda_mode=lambda_mode,
            selected_method=selected_method,
            selected_lambda_mode=selected_lambda_mode,
            requested_methods=requested_methods,
        )

        approximated = roc.get("approximated", {})
        if not isinstance(approximated, dict):
            approximated = {}

        approx_fp = np.asarray(approximated.get("false_alarm", []), dtype=float)
        approx_tp = np.asarray(approximated.get("sensitivity", []), dtype=float)
        approx_kappa = np.asarray(approximated.get("kappa", []), dtype=float)
        approximation_method = str(approximated.get("method", "none"))

        use_approximated = (
            approximation_method not in {"", "none", "raw"}
            and len(approx_fp) == len(approx_tp)
            and len(approx_fp) >= 2
        )

        if use_approximated:
            fp_curve, tp_curve = _roc_curve_for_plot(approx_fp, approx_tp)
        else:
            fp_curve, tp_curve = _roc_curve_for_plot(fp, tp)

        if show_curve:
            curve_suffix = f" ({approximation_method})" if use_approximated else ""
            ax.plot(fp_curve, tp_curve, linewidth=2, label=f"ROC {label}{curve_suffix}")

        if show_points:
            ax.scatter(fp, tp, s=25, alpha=0.75, label=f"Observed points {label}")

        if show_auxiliary and show_eroc and roc.get("england_eroc") is not None:
            eroc = roc["england_eroc"]
            if isinstance(eroc, dict) and "E" in eroc and "R" in eroc:
                E = float(eroc["E"])
                R = float(eroc["R"])

                fp_grid = np.linspace(0.0, 1.0, 200)
                tp_grid = EnglandEROC.tp(fp_grid, E=E, R=R)
                ax.plot(
                    fp_grid,
                    tp_grid,
                    linestyle="--",
                    linewidth=1.5,
                    label=f"England {label}",
                )

        if show_auxiliary and show_kappa and ax2 is not None:
            if use_approximated and len(approx_kappa) == len(approx_fp):
                fp_kappa, kappa_plot = _kappa_curve_for_plot(approx_fp, approx_kappa)
                kappa_suffix = f" ({approximation_method})"
            elif len(kappas) == len(fp):
                fp_kappa, kappa_plot = _kappa_curve_for_plot(fp, kappas)
                kappa_suffix = ""
            else:
                fp_kappa = kappa_plot = None
                kappa_suffix = ""

            if fp_kappa is not None:
                ax2.plot(
                    fp_kappa,
                    kappa_plot,
                    linestyle=":",
                    linewidth=2,
                    label=f"Kappa {label}{kappa_suffix}",
                )

        if show_auxiliary and show_optimal_point:
            try:
                opt_fp = float(threshold["false_alarm"])
                opt_tp = float(threshold["sensitivity"])
            except Exception:
                opt_fp = None
                opt_tp = None

            if opt_fp is not None and opt_tp is not None:
                ax.scatter(
                    [opt_fp],
                    [opt_tp],
                    s=90,
                    marker="*",
                    edgecolors="black",
                    zorder=5,
                    label=f"Working point {label}",
                )

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, alpha=0.4, label="Chance")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False alarm, P_I")
    ax.set_ylabel("Sensitivity, P_A")

    lambda_title = _lambda_values_for_title(
        method_summaries=method_summaries,
        requested_methods=requested_methods,
        lambda_mode=lambda_mode,
    )

    direction = str(contrast_summary.get("primary_direction", "")).strip()
    direction_suffix = f" — {direction}" if direction else ""
    base_title = f"ROC / Kappa — {lambda_mode_label(lambda_mode)}{direction_suffix}"

    if lambda_title:
        ax.set_title(f"{base_title}\n{lambda_title}")
    else:
        ax.set_title(base_title)

    ax.grid(True, alpha=0.25)

    lines, labels = ax.get_legend_handles_labels()

    if show_kappa and ax2 is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2

    if lines:
        ax.legend(lines, labels, loc="lower right", fontsize=8)


def plot_roc_summary_on_axis(
    ax,
    contrast_summary: Dict[str, object],
    methods: str = "both",
    show_curve: bool = True,
    show_points: bool = True,
    show_kappa: bool = True,
    show_optimal_point: bool = True,
    show_eroc: bool = True,
) -> None:
    """
    Draw ROC/Kappa.

    Implementation detail.
    Implementation detail.
        1) ROC / Kappa — lambda free
        2) ROC / Kappa — lambda fixed

    In choose_best_kappa mode, Kappa and the operating point are shown only for
    the selected best pair:
        selected_glm_method + selected_lambda_mode

    Other curves show only the ROC line and ROC points.
    """
    contrast_summary = _normalize_summary_schema(contrast_summary)
    method_summaries = _get_method_summaries(contrast_summary)

    if not method_summaries:
        ax.clear()
        ax.text(
            0.5,
            0.5,
            "No ROC summary available for plotting",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        return

    main_method = contrast_summary.get("main_glm_method", "bayes")
    available_methods = _available_methods_from_summary(method_summaries)
    requested_methods = methods_from_plot_mode(
        plot_mode=methods,
        main_method=str(main_method),
        available_methods=available_methods,
    )

    lambda_modes = _lambda_modes_from_summary(
        contrast_summary=contrast_summary,
        method_summaries=method_summaries,
        requested_methods=requested_methods,
    )

    selected_method, selected_lambda_mode = _selected_method_and_lambda_mode(
        contrast_summary=contrast_summary,
        method_summaries=method_summaries,
        requested_methods=requested_methods,
    )

    fig = ax.figure
    fig.clear()

    if len(lambda_modes) == 1:
        axes = [fig.add_subplot(111)]
    else:
        axes = list(fig.subplots(1, len(lambda_modes), squeeze=False)[0])

    for one_ax, one_lambda_mode in zip(axes, lambda_modes):
        _plot_single_lambda_mode(
            ax=one_ax,
            contrast_summary=contrast_summary,
            method_summaries=method_summaries,
            requested_methods=requested_methods,
            lambda_mode=one_lambda_mode,
            selected_method=selected_method,
            selected_lambda_mode=selected_lambda_mode,
            show_curve=show_curve,
            show_points=show_points,
            show_kappa=show_kappa,
            show_optimal_point=show_optimal_point,
            show_eroc=show_eroc,
        )

    fig.tight_layout()


def save_roc_summary_plot(
    contrast_summary: Dict[str, object],
    out_path: Path,
    methods: str = "both",
    show_curve: bool = True,
    show_points: bool = True,
    show_kappa: bool = True,
    show_optimal_point: bool = True,
    show_eroc: bool = True,
    dpi: int = 160,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))

    plot_roc_summary_on_axis(
        ax=ax,
        contrast_summary=contrast_summary,
        methods=methods,
        show_curve=show_curve,
        show_points=show_points,
        show_kappa=show_kappa,
        show_optimal_point=show_optimal_point,
        show_eroc=show_eroc,
    )

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

def save_selected_lambda_direction_plot(
    roc_models: Dict[str, object],
    selections_by_direction: Dict[str, object],
    out_path: Path,
    method: str,
    lambda_mode: str,
    directions: Optional[List[str]] = None,
    show_points: bool = True,
    show_optimal_point: bool = True,
    dpi: int = 160,
) -> None:
    """Save one ROC figure containing positive and negative directional curves.

    Only the lambda strategy selected for map generation is shown. This keeps
    the figure aligned with the actual maps and prevents a fixed-lambda map in
    one direction from being compared with a free-lambda map in the other.
    """
    method_key = normalize_method_name(method)
    mode = normalize_lambda_mode(lambda_mode)
    requested = list(directions or ["positive", "negative"])
    requested = [str(value).lower().strip() for value in requested]

    method_models = roc_models.get(method_key, {}) if isinstance(roc_models, dict) else {}
    if not isinstance(method_models, dict):
        method_models = {}

    fig, ax = plt.subplots(figsize=(9, 7))
    plotted = 0
    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D"]
    lambda_values: List[str] = []

    for index, direction in enumerate(requested):
        direction_models = method_models.get(direction, {})
        if not isinstance(direction_models, dict):
            continue
        item = direction_models.get(mode)
        if not isinstance(item, dict):
            continue
        curve = item.get("curve", {})
        if not isinstance(curve, dict):
            curve = {}

        raw_fp = np.asarray(curve.get("false_alarm", []), dtype=float)
        raw_tp = np.asarray(curve.get("sensitivity", []), dtype=float)
        approximated = curve.get("approximated", {})
        if not isinstance(approximated, dict):
            approximated = {}
        approx_fp = np.asarray(approximated.get("false_alarm", []), dtype=float)
        approx_tp = np.asarray(approximated.get("sensitivity", []), dtype=float)

        use_approx = (
            approx_fp.size >= 2
            and approx_tp.size == approx_fp.size
            and np.all(np.isfinite(approx_fp))
            and np.all(np.isfinite(approx_tp))
        )
        plot_fp, plot_tp = (approx_fp, approx_tp) if use_approx else (raw_fp, raw_tp)
        finite = np.isfinite(plot_fp) & np.isfinite(plot_tp)
        if np.count_nonzero(finite) < 2:
            continue

        selection = selections_by_direction.get(direction, {})
        if not isinstance(selection, dict):
            selection = {}
        operating = selection.get("operating_point", {})
        if not isinstance(operating, dict):
            operating = {}

        kappa_value = operating.get("kappa")
        threshold_value = operating.get("threshold_value")
        auc_value = item.get("auc")
        lambda_value = item.get("lambda")
        if lambda_value is not None:
            try:
                lambda_values.append(f"{direction}={float(lambda_value):.4g}")
            except (TypeError, ValueError):
                pass

        label_parts = [direction.capitalize()]
        try:
            label_parts.append(f"AUC={float(auc_value):.3f}")
        except (TypeError, ValueError):
            pass
        try:
            label_parts.append(f"Kappa={float(kappa_value):.3f}")
        except (TypeError, ValueError):
            pass
        try:
            label_parts.append(f"t={float(threshold_value):.2f}")
        except (TypeError, ValueError):
            pass

        ax.plot(
            plot_fp[finite],
            plot_tp[finite],
            linestyle=line_styles[index % len(line_styles)],
            linewidth=2.0,
            label=" | ".join(label_parts),
        )

        if show_points and raw_fp.size and raw_tp.size == raw_fp.size:
            raw_finite = np.isfinite(raw_fp) & np.isfinite(raw_tp)
            ax.scatter(
                raw_fp[raw_finite],
                raw_tp[raw_finite],
                marker=markers[index % len(markers)],
                s=18,
                alpha=0.55,
            )

        if show_optimal_point:
            try:
                op_fp = float(operating.get("false_alarm"))
                op_tp = float(operating.get("sensitivity"))
            except (TypeError, ValueError):
                op_fp = float("nan")
                op_tp = float("nan")
            if np.isfinite(op_fp) and np.isfinite(op_tp):
                ax.scatter([op_fp], [op_tp], marker="*", s=150, zorder=5)

        plotted += 1

    ax.plot([0, 1], [0, 1], linestyle=":", linewidth=1.2, label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False alarm, P_I")
    ax.set_ylabel("Sensitivity, P_A")
    ax.grid(True, alpha=0.25)

    mode_text = "fixed" if mode == "fixed" else "free"
    lambda_text = ", ".join(lambda_values)
    title = f"Directional ROC curves — {method_key}, lambda mode={mode_text}"
    if lambda_text:
        title += f"\nLambda: {lambda_text}"
    ax.set_title(title)

    if plotted:
        ax.legend(loc="lower right", fontsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            f"No {mode} ROC curves for method {method_key}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)
