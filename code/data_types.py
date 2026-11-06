# file: data_types.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Voxel:
    x: int
    y: int
    z: int

    def as_tuple(self) -> Tuple[int, int, int]:
        return self.x, self.y, self.z


@dataclass
class ExperimentalEvent:
    event_id: int
    onset: float
    duration: float
    amplitude: float = 1.0
    condition: Optional[str] = None


@dataclass
class RunData:
    name: str
    data: np.ndarray
    affine: np.ndarray
    header: nib.Nifti1Header
    source_files: List[Path]
    metadata: Optional[Dict[str, Any]] = None
    slice_timing: Optional[List[float]] = None

    @property
    def n_timepoints(self) -> int:
        return int(self.data.shape[0])

    @property
    def spatial_shape(self) -> Tuple[int, int, int]:
        return tuple(self.data.shape[1:4])

    def voxel_timeseries(self, voxel: Voxel) -> np.ndarray:
        return self.data[:, voxel.x, voxel.y, voxel.z].astype(float)


@dataclass
class PatientData:
    patient_id: str
    runs: Dict[str, RunData]
    logs: Dict[str, pd.DataFrame]
    run_order: Optional[List[str]] = None
    manifest: Optional[Dict[str, Any]] = None


@dataclass
class BrainMask:
    mask: np.ndarray
    affine: np.ndarray
    header: nib.Nifti1Header

    @property
    def n_voxels(self) -> int:
        return int(np.sum(self.mask))

    @property
    def coords(self) -> np.ndarray:
        return np.argwhere(self.mask)




@dataclass
class DesignMatrices:
    stimulus: np.ndarray
    nuisance: np.ndarray
    full: np.ndarray
    stimulus_names: List[str]
    nuisance_names: List[str]

    @property
    def n_stimulus(self) -> int:
        return int(self.stimulus.shape[1])

    @property
    def n_nuisance(self) -> int:
        return int(self.nuisance.shape[1])


@dataclass
class EBResult:
    h: np.ndarray
    H: np.ndarray
    B: np.ndarray
    Omega: np.ndarray
    sigma2: np.ndarray
    mu: np.ndarray
    n_iter: int
    converged: bool
    log_likelihood: float = float("nan")
    start_log_likelihood: float = float("nan")
    used_ols_fallback: bool = False
    bayes_converged: Optional[bool] = None
    diagnostics: Optional[Dict[str, Any]] = None


@dataclass
class ContrastResult:
    t_runwise: np.ndarray
    T_values: Dict[str, np.ndarray]


@dataclass
class ROCResult:
    thresholds: np.ndarray
    counts: np.ndarray
    lambda_: float
    p_active_bins: np.ndarray
    p_inactive_bins: np.ndarray
    gamma: np.ndarray
    converged: bool
    n_iter: int
    direction: str = "absolute"
    stochastic_dominance_valid: bool = False
    dominance_min_margin: float = float("nan")
    auc: float = float("nan")
    valid: bool = False
    invalid_reason: Optional[str] = None
    approximation_method: str = "none"
    approximated_thresholds: Optional[np.ndarray] = None
    approximated_sensitivity: Optional[np.ndarray] = None
    approximated_false_alarm: Optional[np.ndarray] = None
    approximated_kappa: Optional[np.ndarray] = None
    approximated_auc: float = float("nan")


@dataclass
class KappaThreshold:
    threshold_index: int
    threshold_value: float
    sensitivity: float
    false_alarm: float
    p_observed: float
    p_chance: float
    kappa: float
    valid: bool = True
    invalid_reason: Optional[str] = None
    is_boundary: bool = False
    auc: float = float("nan")
    curve_source: str = "raw"


@dataclass
class ReproducibilityMap:
    R: np.ndarray
    rate: np.ndarray
    strong: np.ndarray
    moderate: np.ndarray
    weak: np.ndarray
    final: np.ndarray
    T: np.ndarray
    mean_t: np.ndarray
    sign: np.ndarray
    R_pos: np.ndarray
    R_neg: np.ndarray
    rate_pos: np.ndarray
    rate_neg: np.ndarray
    reproducibility_pos: np.ndarray
    reproducibility_neg: np.ndarray
    T_pos: np.ndarray
    T_neg: np.ndarray