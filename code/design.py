from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.special import gammaln

from config import AppConfig
from data_types import DesignMatrices, ExperimentalEvent

logger = logging.getLogger(__name__)


def _gamma_pdf(t: np.ndarray | float, shape: float, scale: float) -> np.ndarray:
    values = np.asarray(t, dtype=float)
    result = np.zeros_like(values, dtype=float)
    positive = values > 0.0
    if np.any(positive):
        x = values[positive]
        log_pdf = (
            (shape - 1.0) * np.log(x)
            - x / scale
            - gammaln(shape)
            - shape * np.log(scale)
        )
        result[positive] = np.exp(log_pdf)
    return result


def hrf_spm_canonical(t: np.ndarray | float) -> np.ndarray:
    """SPM canonical double-gamma HRF before discrete normalization."""
    values = np.asarray(t, dtype=float)
    return _gamma_pdf(values, shape=6.0, scale=1.0) - _gamma_pdf(
        values, shape=16.0, scale=1.0
    ) / 6.0


def hrf_liou_spm_like(
    t: float,
    alpha1: float = 6,
    alpha2: float = 16,
    beta1: float = 1,
    beta2: float = 1,
    c: float = 1 / 6,
    A: float = 1,
) -> float:
    """Legacy-compatible double-gamma function with corrected gamma exponents."""
    if t < 0:
        return 0.0
    value = _gamma_pdf(np.asarray([t]), alpha1, beta1)[0]
    value -= c * _gamma_pdf(np.asarray([t]), alpha2, beta2)[0]
    return float(A * value)


def hrf_glover_like(
    t: float,
    a1: float = 6,
    a2: float = 12,
    b1: float = 0.9,
    b2: float = 0.9,
    c: float = 0.35,
) -> float:
    if t < 0:
        return 0.0
    value = _gamma_pdf(np.asarray([t]), a1, b1)[0]
    value -= c * _gamma_pdf(np.asarray([t]), a2, b2)[0]
    return float(value)


class DesignMatrixBuilder:
    """
    Build slice-specific first-level design matrices.

    The returned design is explicitly separated into:

        stimulus: X_stim, whose five condition coefficients enter the
                  hierarchical Liou model and determine the 5 x 5 Omega;
        nuisance: Z, containing drift, SPM motion parameters and intercept;
        full:     [X_stim, Z], used for AR estimation and prewhitening.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._design_cache: Dict[Tuple[str, int], DesignMatrices] = {}
        self.images_order: List[str] = []
        self._hrf_time: Optional[np.ndarray] = None
        self._hrf_cumulative: Optional[np.ndarray] = None

    def set_images_order_from_logs(
        self,
        logs_by_run: Dict[str, pd.DataFrame],
        run_names: List[str],
    ) -> None:
        stimulus_col = self.config.data.log_stimulus_column
        images_order: List[str] = []
        for run in run_names:
            log_df = logs_by_run.get(run)
            if log_df is None or stimulus_col not in log_df.columns:
                continue
            for image in log_df[stimulus_col].dropna().astype(str).values:
                if image not in images_order:
                    images_order.append(image)
        self.images_order = images_order[: self.config.scanner.nS]

    def clear_cache(self) -> None:
        self._design_cache.clear()

    def frame_times(self) -> np.ndarray:
        return np.arange(self.config.scanner.nT, dtype=float) * float(
            self.config.scanner.TR
        )

    def slice_times(self, run_name: Optional[str] = None) -> np.ndarray:
        n_z = int(self.config.scanner.nZ)

        if run_name is not None:
            by_run = getattr(self.config.scanner, "slice_timing_by_run", {}) or {}
            values = by_run.get(run_name)
            if values is not None:
                arr = np.asarray(values, dtype=float).reshape(-1)
                if len(arr) != n_z:
                    raise ValueError(
                        f"SliceTiming for {run_name} has {len(arr)} values; expected {n_z}"
                    )
                return arr

        explicit = getattr(self.config.scanner, "slice_timing", None)
        if explicit is not None:
            arr = np.asarray(explicit, dtype=float).reshape(-1)
            if len(arr) != n_z:
                raise ValueError(
                    f"scanner.slice_timing has {len(arr)} values; expected {n_z}"
                )
            return arr

        step = float(self.config.scanner.slice_time_step)
        if step <= 0.0:
            logger.warning(
                "No SliceTiming was resolved for run=%s and slice_time_step<=0; "
                "using zero offsets.",
                run_name,
            )
            return np.zeros(n_z, dtype=float)

        raw_times = [step * i for i in range(1, n_z + 1)]
        ordered = [0.0] * n_z
        idx = 0
        for z in range(n_z):
            if z % 2 == 0:
                ordered[z] = raw_times[idx]
                idx += 1
        for z in range(n_z):
            if z % 2 == 1:
                ordered[z] = raw_times[idx]
                idx += 1
        return np.asarray(ordered, dtype=float)

    def build_for_run_and_slice(
        self,
        run_name: str,
        log_df: Optional[pd.DataFrame],
        z: int,
    ) -> np.ndarray:
        """Backward-compatible access to the complete first-level design."""
        return self.build_matrices_for_run_and_slice(run_name, log_df, z).full

    def build_matrices_for_run_and_slice(
        self,
        run_name: str,
        log_df: Optional[pd.DataFrame],
        z: int,
    ) -> DesignMatrices:
        cache_key = (str(run_name), int(z))
        cached = self._design_cache.get(cache_key)
        if cached is not None:
            return cached

        if log_df is None:
            logger.warning(
                "Run %s has no events file. Creating a zero stimulus design.",
                run_name,
            )
            n_stim = int(self.config.scanner.nS)
            x_stim = np.zeros((self.config.scanner.nT, n_stim), dtype=float)
            stimulus_names = [f"stimulus_{i + 1}" for i in range(n_stim)]
        else:
            events = self.extract_events(log_df)
            x_stim, stimulus_names = self.build_stimulus_design_with_names(
                events=events,
                z=z,
                run_name=run_name,
            )

        nuisance, nuisance_names = self.build_nuisance_regressors(
            n_t=x_stim.shape[0],
            run_name=run_name,
        )
        full = (
            np.column_stack([x_stim, nuisance])
            if nuisance.shape[1]
            else x_stim.copy()
        )

        if self.config.design.normalize_columns:
            full = normalize_columns(full)
            x_stim = full[:, : x_stim.shape[1]]
            nuisance = full[:, x_stim.shape[1] :]

        result = DesignMatrices(
            stimulus=np.asarray(x_stim, dtype=float),
            nuisance=np.asarray(nuisance, dtype=float),
            full=np.asarray(full, dtype=float),
            stimulus_names=list(stimulus_names),
            nuisance_names=list(nuisance_names),
        )
        self._design_cache[cache_key] = result
        return result

    def extract_events(self, log_df: pd.DataFrame) -> List[ExperimentalEvent]:
        if self.config.design.design_type == "condition_regressors":
            return self._extract_condition_events(log_df)
        return self._extract_legacy_stimulus_events(log_df)

    def _extract_condition_events(
        self,
        log_df: pd.DataFrame,
    ) -> List[ExperimentalEvent]:
        onset_col = self.config.data.log_onset_column
        duration_col = self.config.data.log_duration_column
        condition_col = self.config.data.log_condition_column
        required = [onset_col, duration_col, condition_col]
        missing = [name for name in required if name not in log_df.columns]
        if missing:
            raise KeyError(
                f"Condition events are missing columns {missing}; "
                f"available={log_df.columns.tolist()}"
            )

        condition_order = list(self.config.data.condition_order)
        allowed = set(condition_order) if condition_order else None
        events: List[ExperimentalEvent] = []

        for row_index, row in log_df.reset_index(drop=True).iterrows():
            condition = str(row[condition_col])
            if allowed is not None and condition not in allowed:
                continue
            onset = float(row[onset_col])
            duration = float(row[duration_col])
            if not np.isfinite(onset) or not np.isfinite(duration):
                raise ValueError(f"Non-finite event at row {row_index}")
            if duration <= 0.0:
                raise ValueError(f"Non-positive event duration at row {row_index}")
            events.append(
                ExperimentalEvent(
                    event_id=len(events) + 1,
                    onset=onset,
                    duration=duration,
                    amplitude=1.0,
                    condition=condition,
                )
            )

        if not events:
            raise ValueError("No condition-level events were extracted")
        return events

    def _extract_legacy_stimulus_events(
        self,
        log_df: pd.DataFrame,
    ) -> List[ExperimentalEvent]:
        stimulus_col = self.config.data.log_stimulus_column
        prestim_col = self.config.data.log_prestim_column
        if stimulus_col not in log_df.columns or prestim_col not in log_df.columns:
            raise KeyError(
                f"Legacy log needs {stimulus_col!r} and {prestim_col!r}; "
                f"available={log_df.columns.tolist()}"
            )

        df = log_df.copy().reset_index(drop=True)
        images_order = list(self.images_order)
        if not images_order:
            for image in df[stimulus_col].dropna().astype(str).values:
                if image not in images_order:
                    images_order.append(image)
            images_order = images_order[: self.config.scanner.nS]

        by_index: Dict[int, ExperimentalEvent] = {}
        for s, row in df.iterrows():
            image = str(row[stimulus_col])
            if image not in images_order:
                continue
            real_n = images_order.index(image)
            onset = (
                float(self.config.data.onset_base_shift_sec)
                + s * float(self.config.data.stimulus_period_sec)
                + float(row[prestim_col]) / 1000.0
            )
            by_index[real_n] = ExperimentalEvent(
                event_id=real_n + 1,
                onset=onset,
                duration=float(self.config.data.default_event_duration),
                amplitude=1.0,
                condition=image,
            )

        events: List[ExperimentalEvent] = []
        for real_n in range(self.config.scanner.nS):
            events.append(
                by_index.get(
                    real_n,
                    ExperimentalEvent(
                        event_id=real_n + 1,
                        onset=-1.0,
                        duration=-1.0,
                        amplitude=0.0,
                        condition=None,
                    ),
                )
            )
        return events

    def build_stimulus_design(
        self,
        events: List[ExperimentalEvent],
        z: int,
        run_name: Optional[str] = None,
    ) -> np.ndarray:
        return self.build_stimulus_design_with_names(events, z, run_name)[0]

    def build_stimulus_design_with_names(
        self,
        events: List[ExperimentalEvent],
        z: int,
        run_name: Optional[str] = None,
    ) -> tuple[np.ndarray, List[str]]:
        if self.config.design.design_type == "event_regressors":
            n_s = int(self.config.scanner.nS)
            x = np.zeros((self.config.scanner.nT, n_s), dtype=float)
            names: List[str] = []
            for j, event in enumerate(events[:n_s]):
                x[:, j] = self._event_response_vector(event, z, run_name)
                names.append(event.condition or f"event_{j + 1}")
            while len(names) < n_s:
                names.append(f"event_{len(names) + 1}")
            return x, names

        if self.config.design.design_type != "condition_regressors":
            raise ValueError(f"Unknown design_type: {self.config.design.design_type}")

        conditions = list(self.config.data.condition_order)
        if not conditions:
            conditions = sorted(
                {event.condition for event in events if event.condition is not None}
            )
        if not conditions:
            raise ValueError("No conditions are available for the design")

        x = np.zeros((self.config.scanner.nT, len(conditions)), dtype=float)
        condition_to_col = {condition: i for i, condition in enumerate(conditions)}
        for event in events:
            col = condition_to_col.get(event.condition)
            if col is not None:
                x[:, col] += self._event_response_vector(event, z, run_name)
        return x, conditions

    def _event_response_vector(
        self,
        event: ExperimentalEvent,
        z: int,
        run_name: Optional[str],
    ) -> np.ndarray:
        frame_times = self.frame_times()
        if event.onset < 0.0 or event.duration <= 0.0 or event.amplitude == 0.0:
            return np.zeros_like(frame_times)

        slice_offset = 0.0
        if self.config.design.use_slice_timing:
            slice_offset = float(self.slice_times(run_name)[int(z)])
        sample_times = frame_times + slice_offset
        response = self._boxcar_convolved_hrf(
            sample_times=sample_times,
            onset=float(event.onset),
            duration=float(event.duration),
        )
        return float(event.amplitude) * response

    def _ensure_hrf_cumulative(self) -> tuple[np.ndarray, np.ndarray]:
        if self._hrf_time is not None and self._hrf_cumulative is not None:
            return self._hrf_time, self._hrf_cumulative

        length = max(1.0, float(self.config.design.hrf_length_sec))
        oversampling = max(1, int(self.config.design.hrf_oversampling))
        dt = float(self.config.scanner.TR) / float(oversampling)
        time = np.arange(0.0, length + dt, dt, dtype=float)

        model = str(self.config.design.hrf_model).lower().strip()
        if model in {"spm", "spm_canonical", "canonical", "double_gamma"}:
            hrf = hrf_spm_canonical(time)
        elif model == "liou_spm_like":
            hrf = np.asarray([hrf_liou_spm_like(float(t)) for t in time])
        elif model == "glover_like":
            hrf = np.asarray([hrf_glover_like(float(t)) for t in time])
        else:
            raise ValueError(f"Unknown HRF model: {self.config.design.hrf_model}")

        area = float(np.trapezoid(hrf, time))
        if not np.isfinite(area) or abs(area) < 1e-12:
            raise ValueError("The HRF has zero or non-finite integral")
        hrf = hrf / area
        cumulative = np.concatenate(
            [[0.0], cumulative_trapezoid(hrf, time)]
        )
        self._hrf_time = time
        self._hrf_cumulative = cumulative
        return time, cumulative

    def _hrf_integral(self, lag: np.ndarray) -> np.ndarray:
        time, cumulative = self._ensure_hrf_cumulative()
        values = np.asarray(lag, dtype=float)
        return np.interp(values, time, cumulative, left=0.0, right=float(cumulative[-1]))

    def _boxcar_convolved_hrf(
        self,
        sample_times: np.ndarray,
        onset: float,
        duration: float,
    ) -> np.ndarray:
        # Integral h(t-u) du, u in [onset, onset+duration], equals
        # H(t-onset) - H(t-onset-duration).
        upper_lag = np.asarray(sample_times, dtype=float) - onset
        lower_lag = upper_lag - duration
        return self._hrf_integral(upper_lag) - self._hrf_integral(lower_lag)

    def build_nuisance_regressors(
        self,
        n_t: int,
        run_name: str,
    ) -> tuple[np.ndarray, List[str]]:
        parts: List[np.ndarray] = []
        names: List[str] = []

        if self.config.design.add_drift_terms:
            drift = self._build_drift_terms(n_t)
            if drift.shape[1]:
                parts.append(drift)
                names.extend([f"drift_{i + 1}" for i in range(drift.shape[1])])

        if self.config.design.add_motion_regressors:
            motion = self._motion_regressors(run_name, n_t)
            if motion.shape[1]:
                parts.append(motion)
                base_names = ["motion_tx", "motion_ty", "motion_tz", "motion_rx", "motion_ry", "motion_rz"]
                names.extend(base_names[: motion.shape[1]])

        if self.config.design.add_intercept:
            parts.append(np.ones((n_t, 1), dtype=float))
            names.append("intercept")

        if not parts:
            return np.zeros((n_t, 0), dtype=float), []
        return np.column_stack(parts), names

    def _motion_regressors(self, run_name: str, n_t: int) -> np.ndarray:
        by_run = getattr(self.config.design, "motion_regressors_by_run", {}) or {}
        values = by_run.get(run_name)
        if values is None:
            raise FileNotFoundError(
                f"Motion regressors are enabled but no rp_bold parameters were loaded for {run_name}"
            )
        motion = np.asarray(values, dtype=float)
        if motion.ndim != 2 or motion.shape[1] < 6:
            raise ValueError(
                f"Motion parameters for {run_name} must have at least 6 columns; got {motion.shape}"
            )
        motion = motion[:, :6]
        if motion.shape[0] != int(n_t):
            raise ValueError(
                f"Motion parameters for {run_name} have {motion.shape[0]} rows; expected {n_t}"
            )
        model = str(self.config.design.motion_model).lower().strip()
        if model not in {"6", "spm6", "rigid6"}:
            raise ValueError(
                f"Unsupported motion_model={self.config.design.motion_model!r}; currently only '6' is supported"
            )
        if self.config.design.center_motion_regressors:
            motion = motion - np.mean(motion, axis=0, keepdims=True)
        return motion

    def _build_drift_terms(self, n_t: int) -> np.ndarray:
        kind = str(self.config.design.drift_kind).lower().strip()
        if kind == "polynomial":
            t = np.linspace(-1.0, 1.0, n_t)
            cols = [
                t**degree
                for degree in range(1, int(self.config.design.drift_order) + 1)
            ]
            return np.column_stack(cols) if cols else np.zeros((n_t, 0))

        if kind == "cosine":
            duration = n_t * float(self.config.scanner.TR)
            cutoff = float(self.config.design.high_pass_cutoff_sec)
            n_basis = int(np.floor(2.0 * duration / cutoff))
            if n_basis <= 0:
                return np.zeros((n_t, 0), dtype=float)
            t = np.arange(n_t, dtype=float)
            return np.column_stack(
                [
                    np.cos(np.pi * (t + 0.5) * k / n_t)
                    for k in range(1, n_basis + 1)
                ]
            )

        raise ValueError(f"Unknown drift_kind: {self.config.design.drift_kind}")


def residualize_nuisance_orthonormal(
    y: np.ndarray,
    x_stim: np.ndarray,
    nuisance: np.ndarray,
    rank_tolerance: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Apply the Frisch-Waugh-Lovell reduction in an orthonormal complement.

    With whitened errors e~N(0,sigma^2 I), Q_perp.T e remains spherical. This is
    preferable to retaining M_Z y in the original n-dimensional singular space.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    x_stim = np.asarray(x_stim, dtype=float)
    nuisance = np.asarray(nuisance, dtype=float)

    if nuisance.ndim != 2 or nuisance.shape[0] != len(y):
        raise ValueError("Nuisance matrix has incompatible shape")
    if x_stim.ndim != 2 or x_stim.shape[0] != len(y):
        raise ValueError("Stimulus matrix has incompatible shape")
    if nuisance.shape[1] == 0:
        return y, x_stim, 0

    q, r = np.linalg.qr(nuisance, mode="complete")
    diag = np.abs(np.diag(r[: min(r.shape), : min(r.shape)]))
    scale = float(np.max(diag)) if diag.size else 0.0
    tol = max(float(rank_tolerance), np.finfo(float).eps * max(nuisance.shape) * scale)
    rank = int(np.count_nonzero(diag > tol))
    q_perp = q[:, rank:]
    if q_perp.shape[1] <= x_stim.shape[1]:
        raise ValueError(
            f"Too few residual degrees of freedom after nuisance removal: {q_perp.shape[1]}"
        )
    return q_perp.T @ y, q_perp.T @ x_stim, rank


def normalize_columns(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    result = np.asarray(x, dtype=float).copy()
    for j in range(result.shape[1]):
        norm = float(np.linalg.norm(result[:, j]))
        if norm > eps:
            result[:, j] /= norm
    return result


def pad_contrast(values: List[float], target_len: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if len(arr) > target_len:
        raise ValueError(
            f"Contrast length {len(arr)} exceeds target dimension {target_len}"
        )
    if len(arr) < target_len:
        arr = np.pad(arr, (0, target_len - len(arr)))
    return arr
