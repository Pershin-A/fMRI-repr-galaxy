# file: ordinary_least_squares.py

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from data_types import EBResult


class OrdinaryLeastSquaresGLM:
    """
    Ordinary Least Squares GLM.

    Implementation detail.

    Implementation detail.
        beta_i = pinv(X_i) @ y_i

    Implementation detail.
        cov(beta_i) = sigma_i^2 * pinv(X_i' X_i)

    Implementation detail.
        Implementation detail.
        Implementation detail.
        Implementation detail.
        Implementation detail.
        Implementation detail.
    """

    def __init__(
        self,
        min_sigma2: float = 1e-10,
        omega_ridge: float = 1e-6,
        omega_shrinkage_alpha: float = 0.0,
        omega_shrinkage_target: str = "diagonal",
        omega_eigenvalue_floor_relative: float = 1e-4,
    ):
        self.min_sigma2 = float(min_sigma2)
        self.omega_ridge = float(omega_ridge)
        self.omega_shrinkage_alpha = float(omega_shrinkage_alpha)
        self.omega_shrinkage_target = str(omega_shrinkage_target)
        self.omega_eigenvalue_floor_relative = float(omega_eigenvalue_floor_relative)

    def fit(self, X_list: List[np.ndarray], y_list: List[np.ndarray], X_star: np.ndarray) -> EBResult:
        self._validate_inputs(X_list, y_list, X_star)

        betas = []
        covs = []
        sigma2 = []

        for X, y in zip(X_list, y_list):
            beta, cov_beta, sig2 = self._fit_one_run(X, y)

            betas.append(beta)
            covs.append(cov_beta)
            sigma2.append(sig2)

        h = np.vstack(betas)
        H = np.stack(covs, axis=0)
        sigma2_arr = np.asarray(sigma2, dtype=float)

        B, Omega, mu = self._fit_second_level(h, X_star)

        eigvals = np.linalg.eigvalsh(Omega)
        return EBResult(
            h=h,
            H=H,
            B=B,
            Omega=Omega,
            sigma2=sigma2_arr,
            mu=mu,
            n_iter=1,
            converged=True,
            diagnostics={
                "omega_dimension": int(Omega.shape[0]),
                "omega_regularized_eigenvalues": eigvals.tolist(),
                "omega_regularized_condition_number": float(np.linalg.cond(Omega)),
                "omega_shrinkage_alpha": float(self.omega_shrinkage_alpha),
                "omega_shrinkage_target": self.omega_shrinkage_target,
            },
        )

    def _fit_one_run(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        XtX = X.T @ X
        XtX_inv = np.linalg.pinv(XtX)

        beta = np.linalg.pinv(X) @ y
        resid = y - X @ beta

        rank = np.linalg.matrix_rank(X)
        df = max(1, X.shape[0] - rank)

        sig2 = float(np.sum(resid ** 2) / df)
        sig2 = max(sig2, self.min_sigma2)

        cov_beta = sig2 * XtX_inv
        cov_beta = 0.5 * (cov_beta + cov_beta.T)

        return beta, cov_beta, sig2

    def _fit_second_level(self, h: np.ndarray, X_star: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        XtX = X_star.T @ X_star
        B = np.linalg.pinv(XtX) @ X_star.T @ h

        mu = X_star @ B
        residual = h - mu

        rank = np.linalg.matrix_rank(X_star)
        denom = max(1, h.shape[0] - rank)

        Omega = residual.T @ residual / denom
        Omega = self._regularize_cov(Omega)

        return B, Omega, mu

    def _regularize_cov(self, A: np.ndarray) -> np.ndarray:
        raw = 0.5 * (np.asarray(A, dtype=float) + np.asarray(A, dtype=float).T)
        k = raw.shape[0]
        alpha = float(np.clip(self.omega_shrinkage_alpha, 0.0, 1.0))
        target_name = self.omega_shrinkage_target.lower().strip()
        if target_name in {"diagonal", "diag"}:
            target = np.diag(np.maximum(np.diag(raw), 0.0))
        elif target_name in {"scaled_identity", "identity", "isotropic"}:
            target = max(float(np.trace(raw)) / max(1, k), 0.0) * np.eye(k)
        else:
            raise ValueError(
                "omega_shrinkage_target must be 'diagonal' or 'scaled_identity'"
            )
        regularized = (1.0 - alpha) * raw + alpha * target
        scale = max(
            float(np.trace(np.abs(regularized))) / max(1, k),
            self.omega_ridge,
        )
        floor = max(
            self.omega_ridge,
            self.omega_eigenvalue_floor_relative * scale,
        )
        eigvals, eigvecs = np.linalg.eigh(
            0.5 * (regularized + regularized.T)
        )
        eigvals = np.maximum(eigvals, floor)
        result = (eigvecs * eigvals) @ eigvecs.T
        return 0.5 * (result + result.T)

    @staticmethod
    def _validate_inputs(X_list: List[np.ndarray], y_list: List[np.ndarray], X_star: np.ndarray) -> None:
        if len(X_list) == 0:
            raise ValueError("X_list is empty")

        if len(X_list) != len(y_list):
            raise ValueError("X_list and y_list have different lengths")

        if X_star.shape[0] != len(X_list):
            raise ValueError("X_star must have one row per run")

        k = X_list[0].shape[1]

        for i, (X, y) in enumerate(zip(X_list, y_list)):
            if X.ndim != 2:
                raise ValueError(f"X[{i}] must be a two-dimensional matrix")

            if y.ndim != 1:
                raise ValueError(f"y[{i}] must be a one-dimensional time series")

            if X.shape[0] != len(y):
                raise ValueError(f"X[{i}] and y[{i}] have incompatible sizes")

            if X.shape[1] != k:
                raise ValueError("All X_i must have the same number of regressors")