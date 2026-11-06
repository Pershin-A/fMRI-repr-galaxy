from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.interpolate import PchipInterpolator

from config import AnalysisConfig
from data_types import KappaThreshold, ROCResult


def rates_from_roc(roc: ROCResult) -> tuple[np.ndarray, np.ndarray]:
    """Return sensitivity and false-alarm values at the stored thresholds."""
    n_thresholds = len(roc.thresholds)
    sensitivity = np.asarray(
        [np.sum(roc.p_active_bins[j + 1 :]) for j in range(n_thresholds)],
        dtype=float,
    )
    false_alarm = np.asarray(
        [np.sum(roc.p_inactive_bins[j + 1 :]) for j in range(n_thresholds)],
        dtype=float,
    )
    return sensitivity, false_alarm


def kappa_from_rates(
    lambda_: float,
    sensitivity: np.ndarray,
    false_alarm: np.ndarray,
) -> np.ndarray:
    """Calculate Cohen's Kappa for arbitrary ROC coordinates."""
    pa = np.asarray(sensitivity, dtype=float)
    pi = np.asarray(false_alarm, dtype=float)
    values = np.full(np.broadcast_shapes(pa.shape, pi.shape), np.nan, dtype=float)
    pa, pi = np.broadcast_arrays(pa, pi)

    observed = lambda_ * pa + (1.0 - lambda_) * (1.0 - pi)
    selected_active = lambda_ * pa + (1.0 - lambda_) * pi
    chance = lambda_ * selected_active + (1.0 - lambda_) * (1.0 - selected_active)
    denominator = 1.0 - chance

    valid = (
        np.isfinite(pa)
        & np.isfinite(pi)
        & np.isfinite(denominator)
        & (denominator > 0.0)
    )
    values[valid] = (observed[valid] - chance[valid]) / denominator[valid]
    return values


def kappa_curve(roc: ROCResult) -> np.ndarray:
    sensitivity, false_alarm = rates_from_roc(roc)
    return kappa_from_rates(float(roc.lambda_), sensitivity, false_alarm)


def _auc(false_alarm: np.ndarray, sensitivity: np.ndarray) -> float:
    x = np.asarray(false_alarm, dtype=float)
    y = np.asarray(sensitivity, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return float("nan")
    order = np.argsort(x)
    x, y = x[order], y[order]
    # Remove repeated x values while retaining the largest sensitivity at that x.
    unique_x = np.unique(x)
    unique_y = np.asarray([np.max(y[x == value]) for value in unique_x], dtype=float)
    return float(np.trapezoid(unique_y, unique_x))


def approximate_roc_curve(
    roc: ROCResult,
    config: Optional[AnalysisConfig],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, float]:
    """
    Approximate the discrete ROC coordinates on a dense threshold grid.

    The approximation is parameterized by the common t-threshold, not directly
    by false alarm.  PCHIP is shape preserving and therefore avoids the large
    oscillations possible with an unconstrained cubic spline.  Small numerical
    violations are removed by enforcing monotone non-increasing tail
    probabilities and stochastic dominance after interpolation.
    """
    thresholds = np.asarray(roc.thresholds, dtype=float)
    sensitivity, false_alarm = rates_from_roc(roc)
    raw_kappa = kappa_from_rates(float(roc.lambda_), sensitivity, false_alarm)

    method = (
        str(getattr(config, "roc_curve_approximation", "none")).lower().strip()
        if config is not None
        else "none"
    )
    use_approx = bool(
        getattr(config, "roc_kappa_use_approximated_curve", False)
    ) if config is not None else False

    valid = (
        np.isfinite(thresholds)
        & np.isfinite(sensitivity)
        & np.isfinite(false_alarm)
    )
    x = thresholds[valid]
    pa = sensitivity[valid]
    pi = false_alarm[valid]

    if (
        not use_approx
        or method in {"", "none", "raw", "off", "false"}
        or len(x) < 3
    ):
        roc.approximation_method = "none"
        roc.approximated_thresholds = thresholds.copy()
        roc.approximated_sensitivity = sensitivity.copy()
        roc.approximated_false_alarm = false_alarm.copy()
        roc.approximated_kappa = raw_kappa.copy()
        roc.approximated_auc = _auc(false_alarm, sensitivity)
        return (
            thresholds,
            sensitivity,
            false_alarm,
            raw_kappa,
            "raw",
            roc.approximated_auc,
        )

    order = np.argsort(x)
    x, pa, pi = x[order], pa[order], pi[order]
    x, unique_idx = np.unique(x, return_index=True)
    pa, pi = pa[unique_idx], pi[unique_idx]
    if len(x) < 3 or float(x[-1]) <= float(x[0]):
        roc.approximation_method = "none"
        roc.approximated_thresholds = thresholds.copy()
        roc.approximated_sensitivity = sensitivity.copy()
        roc.approximated_false_alarm = false_alarm.copy()
        roc.approximated_kappa = raw_kappa.copy()
        roc.approximated_auc = _auc(false_alarm, sensitivity)
        return thresholds, sensitivity, false_alarm, raw_kappa, "raw", roc.approximated_auc

    n_points = max(
        len(x),
        int(getattr(config, "roc_approximation_points", max(201, 10 * len(x)))),
    )
    dense_x = np.linspace(float(x[0]), float(x[-1]), n_points, dtype=float)

    if method in {"pchip", "shape_preserving", "monotone_cubic"}:
        dense_pa = PchipInterpolator(x, pa, extrapolate=False)(dense_x)
        dense_pi = PchipInterpolator(x, pi, extrapolate=False)(dense_x)
        actual_method = "pchip"
    elif method in {"linear", "interp", "interpolation"}:
        dense_pa = np.interp(dense_x, x, pa)
        dense_pi = np.interp(dense_x, x, pi)
        actual_method = "linear"
    else:
        raise ValueError(
            f"Unknown roc_curve_approximation={method!r}; expected pchip, linear or none"
        )

    dense_pa = np.clip(np.nan_to_num(dense_pa, nan=0.0), 0.0, 1.0)
    dense_pi = np.clip(np.nan_to_num(dense_pi, nan=0.0), 0.0, 1.0)

    # Tail probabilities must be non-increasing as the threshold rises.
    dense_pa = np.minimum.accumulate(dense_pa)
    dense_pi = np.minimum.accumulate(dense_pi)

    # Preserve first-order stochastic dominance after numerical interpolation.
    dense_pa = np.maximum(dense_pa, dense_pi)
    dense_pa = np.clip(dense_pa, 0.0, 1.0)

    dense_kappa = kappa_from_rates(float(roc.lambda_), dense_pa, dense_pi)
    dense_auc = _auc(dense_pi, dense_pa)

    roc.approximation_method = actual_method
    roc.approximated_thresholds = dense_x
    roc.approximated_sensitivity = dense_pa
    roc.approximated_false_alarm = dense_pi
    roc.approximated_kappa = dense_kappa
    roc.approximated_auc = dense_auc

    return dense_x, dense_pa, dense_pi, dense_kappa, actual_method, dense_auc


def choose_threshold_by_kappa(
    roc: ROCResult,
    config: Optional[AnalysisConfig] = None,
) -> KappaThreshold:
    (
        thresholds,
        sensitivity,
        false_alarm,
        kappas,
        curve_source,
        approximated_auc,
    ) = approximate_roc_curve(roc, config)

    n = len(thresholds)
    if n == 0:
        return KappaThreshold(
            threshold_index=-1,
            threshold_value=float("nan"),
            sensitivity=float("nan"),
            false_alarm=float("nan"),
            p_observed=float("nan"),
            p_chance=float("nan"),
            kappa=float("nan"),
            valid=False,
            invalid_reason="empty_threshold_grid",
            auc=float(getattr(roc, "auc", np.nan)),
            curve_source=curve_source,
        )

    min_kappa = float(getattr(config, "roc_min_kappa", 0.0)) if config else 0.0
    min_auc = float(getattr(config, "roc_min_auc", 0.5)) if config else 0.5
    reject_boundary = bool(
        getattr(config, "roc_reject_boundary_threshold", True)
    ) if config else True

    candidate = np.isfinite(kappas) & np.isfinite(sensitivity) & np.isfinite(false_alarm)
    candidate &= sensitivity > false_alarm
    candidate &= kappas > min_kappa
    if reject_boundary and n > 2:
        candidate[0] = False
        candidate[-1] = False

    global_reasons: list[str] = []
    if not bool(getattr(roc, "stochastic_dominance_valid", False)):
        global_reasons.append("stochastic_dominance_failed")

    auc = approximated_auc if np.isfinite(approximated_auc) else float(
        getattr(roc, "auc", np.nan)
    )
    if not np.isfinite(auc) or auc <= min_auc:
        global_reasons.append("auc_not_above_chance")
    if not bool(getattr(roc, "converged", False)):
        global_reasons.append("roc_em_not_converged")

    valid_indices = np.flatnonzero(candidate)
    valid = bool(len(valid_indices) and not global_reasons)

    if valid:
        best_index = int(valid_indices[np.argmax(kappas[valid_indices])])
        reason = None
    else:
        finite_indices = np.flatnonzero(np.isfinite(kappas))
        best_index = (
            int(finite_indices[np.argmax(kappas[finite_indices])])
            if len(finite_indices)
            else -1
        )
        reasons = list(global_reasons)
        if not np.any(np.isfinite(kappas)):
            reasons.append("all_kappa_nonfinite")
        elif float(np.nanmax(kappas)) <= min_kappa:
            reasons.append("no_positive_kappa")
        if not np.any(sensitivity > false_alarm):
            reasons.append("sensitivity_not_above_false_alarm")
        if reject_boundary and best_index in {0, n - 1}:
            reasons.append("best_threshold_on_grid_boundary")
        reason = ";".join(dict.fromkeys(reasons)) or "no_valid_operating_point"

    if best_index < 0:
        return KappaThreshold(
            threshold_index=-1,
            threshold_value=float("nan"),
            sensitivity=float("nan"),
            false_alarm=float("nan"),
            p_observed=float("nan"),
            p_chance=float("nan"),
            kappa=float("nan"),
            valid=False,
            invalid_reason=reason,
            auc=auc,
            curve_source=curve_source,
        )

    lambda_ = float(roc.lambda_)
    pa = float(sensitivity[best_index])
    pi = float(false_alarm[best_index])
    observed = lambda_ * pa + (1.0 - lambda_) * (1.0 - pi)
    selected_active = lambda_ * pa + (1.0 - lambda_) * pi
    chance = lambda_ * selected_active + (1.0 - lambda_) * (1.0 - selected_active)

    return KappaThreshold(
        threshold_index=best_index,
        threshold_value=float(thresholds[best_index]),
        sensitivity=pa,
        false_alarm=pi,
        p_observed=float(observed),
        p_chance=float(chance),
        kappa=float(kappas[best_index]),
        valid=valid,
        invalid_reason=reason,
        is_boundary=best_index in {0, n - 1},
        auc=auc,
        curve_source=curve_source,
    )
