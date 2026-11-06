from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

from config import AnalysisConfig
from data_types import ReproducibilityMap


def _normalize_direction(direction: str) -> str:
    value = str(direction).lower().strip()
    aliases = {
        "positive": "positive",
        "pos": "positive",
        "+": "positive",
        "negative": "negative",
        "neg": "negative",
        "-": "negative",
        "absolute": "absolute",
        "abs": "absolute",
        "absolute_liou": "absolute",
    }
    if value not in aliases:
        raise ValueError(f"Unknown reproducibility direction: {direction}")
    return aliases[value]


def compute_reproducibility_map(
    t_runwise: np.ndarray,
    T_values: np.ndarray,
    threshold: float,
    mask: np.ndarray,
    coords: np.ndarray,
    config: AnalysisConfig,
    direction: str = "positive",
) -> ReproducibilityMap:
    """Build a directional reproducibility map.

    positive: run is active when t >= threshold;
    negative: run is active when t <= -threshold;
    absolute: legacy Liou-compatible unsigned classification.
    """
    direction = _normalize_direction(direction)
    t_runwise = np.asarray(t_runwise, dtype=float)
    T_values = np.asarray(T_values, dtype=float)

    if direction == "positive":
        active = t_runwise >= float(threshold)
    elif direction == "negative":
        active = t_runwise <= -float(threshold)
    else:
        active = np.abs(t_runwise) >= float(threshold)

    finite = np.isfinite(t_runwise)
    active &= finite
    n_valid = np.sum(finite, axis=0)
    R_vec = np.sum(active, axis=0)
    rate_vec = np.divide(
        R_vec,
        n_valid,
        out=np.full(R_vec.shape, np.nan, dtype=float),
        where=n_valid > 0,
    )

    if direction == "positive":
        direction_gate = np.isfinite(T_values) & (T_values > 0.0)
    elif direction == "negative":
        direction_gate = np.isfinite(T_values) & (T_values < 0.0)
    else:
        direction_gate = np.isfinite(T_values)

    strong_vec = (rate_vec >= float(config.strong_reproducibility)) & direction_gate
    moderate_vec = (
        (rate_vec >= float(config.moderate_reproducibility))
        & (rate_vec < float(config.strong_reproducibility))
        & direction_gate
    )
    weak_vec = (
        (rate_vec >= float(config.weak_reproducibility))
        & (rate_vec < float(config.moderate_reproducibility))
        & direction_gate
    )

    R = vector_to_volume(R_vec, mask.shape, coords, fill=0)
    rate = vector_to_volume(rate_vec, mask.shape, coords, fill=np.nan)
    strong = vector_to_volume(strong_vec, mask.shape, coords, fill=False).astype(bool)
    moderate = vector_to_volume(moderate_vec, mask.shape, coords, fill=False).astype(bool)
    weak = vector_to_volume(weak_vec, mask.shape, coords, fill=False).astype(bool)

    if bool(config.include_moderate_neighbors):
        final = strong | (moderate & binary_dilation(strong) & np.asarray(mask, dtype=bool))
    else:
        final = strong

    mean_t_vec = np.nanmean(t_runwise, axis=0)
    T = vector_to_volume(T_values, mask.shape, coords, fill=np.nan)
    mean_t = vector_to_volume(mean_t_vec, mask.shape, coords, fill=np.nan)
    sign = np.sign(mean_t)
    final_bool = np.asarray(final, dtype=bool)

    if direction == "positive":
        R_pos = np.where(final_bool, R, 0.0)
        R_neg = np.zeros_like(R_pos)
        rate_pos = np.where(final_bool, rate, 0.0)
        rate_neg = np.zeros_like(rate_pos)
        T_pos = np.where(final_bool, np.maximum(T, 0.0), 0.0)
        T_neg = np.zeros_like(T_pos)
    elif direction == "negative":
        R_pos = np.zeros_like(R, dtype=float)
        R_neg = np.where(final_bool, R, 0.0)
        rate_pos = np.zeros_like(rate, dtype=float)
        rate_neg = np.where(final_bool, rate, 0.0)
        T_pos = np.zeros_like(T, dtype=float)
        T_neg = np.where(final_bool, np.minimum(T, 0.0), 0.0)
    else:
        positive = mean_t >= 0.0
        negative = ~positive
        R_pos = np.where(final_bool & positive, R, 0.0)
        R_neg = np.where(final_bool & negative, R, 0.0)
        rate_pos = np.where(final_bool & positive, rate, 0.0)
        rate_neg = np.where(final_bool & negative, rate, 0.0)
        T_pos = np.where(final_bool & (T >= 0.0), T, 0.0)
        T_neg = np.where(final_bool & (T < 0.0), T, 0.0)

    result = ReproducibilityMap(
        R=R,
        rate=rate,
        strong=strong,
        moderate=moderate,
        weak=weak,
        final=final,
        T=T,
        mean_t=mean_t,
        sign=sign,
        R_pos=R_pos,
        R_neg=R_neg,
        rate_pos=rate_pos,
        rate_neg=rate_neg,
        reproducibility_pos=rate_pos,
        reproducibility_neg=rate_neg,
        T_pos=T_pos,
        T_neg=T_neg,
    )
    result.direction = direction
    return result


def vector_to_volume(
    values: np.ndarray,
    shape: tuple[int, int, int],
    coords: np.ndarray,
    fill,
) -> np.ndarray:
    values = np.asarray(values)
    volume = np.full(shape, fill, dtype=values.dtype)
    if len(coords):
        coords = np.asarray(coords, dtype=int)
        volume[coords[:, 0], coords[:, 1], coords[:, 2]] = values
    return volume
