# file: prewhitening.py

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


def _residual_forming_matrix(X: np.ndarray) -> np.ndarray:
    """Return R = I - X X^+ for the ordinary least-squares fit."""
    X = np.asarray(X, dtype=float)
    n = int(X.shape[0])
    return np.eye(n, dtype=float) - X @ np.linalg.pinv(X)


def _ar1_bias_reduction_matrix(X: np.ndarray) -> np.ndarray:
    """
    Matrix used by the Worsley et al. approximate bias reduction.

    For AR(1), Appendix A.1 gives two linear equations

        a = M v,

    where a contains the empirical lag-0 and lag-1 residual autocovariances,
    and v contains the approximately unbiased covariance estimates. M depends
    only on the design matrix through R = I - X X^+, so it can be reused for
    all voxels with the same design matrix.
    """
    R = _residual_forming_matrix(X)
    n = R.shape[0]

    D0 = np.eye(n, dtype=float)
    D1 = np.zeros((n, n), dtype=float)
    if n > 1:
        D1[np.arange(n - 1), np.arange(1, n)] = 1.0

    D1_sym = D1 + D1.T

    M = np.array(
        [
            [np.trace(R @ D0 @ R), np.trace(R @ D0 @ R @ D1_sym)],
            [np.trace(R @ D1 @ R), np.trace(R @ D1 @ R @ D1_sym)],
        ],
        dtype=float,
    )

    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        logger.warning("AR(1) bias-reduction matrix is singular; using pseudo-inverse")
        return np.linalg.pinv(M)


def estimate_ar1_rho_from_residuals(
    y: np.ndarray,
    X: np.ndarray,
    rho_clip: float = 0.99,
    bias_correction: bool = True,
) -> float:
    """
    Estimate AR(1) rho from OLS residuals.

    If bias_correction=True, use the Worsley et al. Appendix A.1 approximate
    bias reduction for the lag-0/lag-1 autocovariances before taking rho=v1/v0.
    If False, use the simple residual autocorrelation from Section 3.2.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)

    beta = np.linalg.pinv(X) @ y
    residual = y - X @ beta

    if bias_correction:
        rho = estimate_ar1_rho_from_residuals_array(
            residuals=residual[:, None],
            X=X,
            rho_clip=rho_clip,
            bias_correction=True,
        )[0]
        return float(rho)

    denom = float(np.sum(residual**2))
    if denom <= 1e-20:
        return 0.0

    numer = float(np.sum(residual[1:] * residual[:-1]))
    rho = numer / denom

    return float(np.clip(rho, -rho_clip, rho_clip))


def estimate_ar1_rho_from_residuals_array(
    residuals: np.ndarray,
    X: np.ndarray,
    rho_clip: float = 0.99,
    bias_correction: bool = True,
) -> np.ndarray:
    """
    Vectorized AR(1) rho estimation from residuals.

    Parameters
    ----------
    residuals:
        Array with shape [n_timepoints, n_voxels].
    X:
        Design matrix used to obtain the residuals.
    rho_clip:
        Absolute clipping bound for numerical stability.
    bias_correction:
        If True, use the Worsley et al. approximate bias correction.
    """
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim == 1:
        residuals = residuals[:, None]

    a0 = np.sum(residuals * residuals, axis=0)
    a1 = np.sum(residuals[1:, :] * residuals[:-1, :], axis=0)

    if bias_correction:
        Minv = _ar1_bias_reduction_matrix(np.asarray(X, dtype=float))
        v0 = Minv[0, 0] * a0 + Minv[0, 1] * a1
        v1 = Minv[1, 0] * a0 + Minv[1, 1] * a1
        denom = v0
    else:
        v1 = a1
        denom = a0

    rho = np.zeros_like(a0, dtype=float)
    valid = np.isfinite(v1) & np.isfinite(denom) & (np.abs(denom) > 1e-20)
    rho[valid] = v1[valid] / denom[valid]
    rho[~np.isfinite(rho)] = 0.0

    return np.clip(rho, -rho_clip, rho_clip).astype(float)


def estimate_ar1_rho_volume_for_run(
    data: np.ndarray,
    mask: np.ndarray,
    design_by_z: dict[int, np.ndarray],
    rho_clip: float = 0.99,
    bias_correction: bool = True,
    smoothing_fwhm: float | None = None,
    voxel_sizes: tuple[float, float, float] | np.ndarray | None = None,
) -> np.ndarray:
    """
    Estimate a full AR(1) rho volume for one run.

    The estimate is computed slice-by-slice because the design matrix may depend
    on z when slice timing regressors are used. After voxelwise estimation, the
    rho map is optionally regularized by mask-normalized Gaussian smoothing, as
    proposed by Worsley et al. for autoregressive parameters.
    """
    data = np.asarray(data, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    if data.ndim != 4:
        raise ValueError(f"Expected run data with shape [T, X, Y, Z], got {data.shape}")

    _, nx, ny, nz = data.shape
    if mask.shape != (nx, ny, nz):
        raise ValueError(f"Mask shape {mask.shape} does not match run data spatial shape {(nx, ny, nz)}")

    rho_volume = np.zeros(mask.shape, dtype=float)

    for z in range(nz):
        slice_mask = mask[:, :, z]
        if not np.any(slice_mask):
            continue

        if z not in design_by_z:
            raise KeyError(f"Missing design matrix for z={z}")

        X = np.asarray(design_by_z[z], dtype=float)
        Y = data[:, :, :, z][:, slice_mask]

        beta = np.linalg.pinv(X) @ Y
        residuals = Y - X @ beta

        rho_slice = estimate_ar1_rho_from_residuals_array(
            residuals=residuals,
            X=X,
            rho_clip=rho_clip,
            bias_correction=bias_correction,
        )

        tmp = np.zeros((nx, ny), dtype=float)
        tmp[slice_mask] = rho_slice
        rho_volume[:, :, z] = tmp

    if smoothing_fwhm is not None and float(smoothing_fwhm) > 0:
        rho_volume = smooth_rho_volume(
            rho_volume=rho_volume,
            fwhm=float(smoothing_fwhm),
            mask=mask,
            voxel_sizes=voxel_sizes,
        )

    return np.clip(rho_volume, -rho_clip, rho_clip).astype(float)


def prewhiten_ar1_vector(y: np.ndarray, rho: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    yw = np.zeros_like(y)

    yw[0] = y[0]
    denom = np.sqrt(max(1e-12, 1.0 - rho**2))

    for i in range(1, len(y)):
        yw[i] = (y[i] - rho * y[i - 1]) / denom

    return yw


def prewhiten_ar1_matrix(X: np.ndarray, rho: float) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Xw = np.zeros_like(X)

    Xw[0, :] = X[0, :]
    denom = np.sqrt(max(1e-12, 1.0 - rho**2))

    for i in range(1, X.shape[0]):
        Xw[i, :] = (X[i, :] - rho * X[i - 1, :]) / denom

    return Xw


def prewhiten_ar1(y: np.ndarray, X: np.ndarray, rho: float) -> tuple[np.ndarray, np.ndarray]:
    return prewhiten_ar1_vector(y, rho), prewhiten_ar1_matrix(X, rho)


def smooth_rho_volume(
    rho_volume: np.ndarray,
    fwhm: float | None,
    mask: np.ndarray | None = None,
    voxel_sizes: tuple[float, float, float] | np.ndarray | None = None,
) -> np.ndarray:
    """
    Spatially regularize rho with a Gaussian FWHM.

    If a mask is supplied, smoothing is mask-normalized to avoid attenuation near
    mask boundaries and leakage from zeros outside the brain.
    """
    rho_volume = np.asarray(rho_volume, dtype=float)

    if fwhm is None or float(fwhm) <= 0:
        return rho_volume

    if voxel_sizes is None:
        sigma = float(fwhm) / np.sqrt(8.0 * np.log(2.0))
    else:
        voxel_sizes = np.asarray(voxel_sizes, dtype=float)
        voxel_sizes = np.where(voxel_sizes > 1e-12, voxel_sizes, 1.0)
        sigma = (float(fwhm) / voxel_sizes) / np.sqrt(8.0 * np.log(2.0))

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        weights = mask.astype(float)
        numerator = gaussian_filter(np.nan_to_num(rho_volume, nan=0.0) * weights, sigma=sigma)
        denominator = gaussian_filter(weights, sigma=sigma)

        smoothed = rho_volume.copy()
        valid = mask & (denominator > 1e-12)
        smoothed[valid] = numerator[valid] / denominator[valid]
        smoothed[~mask] = rho_volume[~mask]
        return smoothed

    return gaussian_filter(rho_volume, sigma=sigma)
