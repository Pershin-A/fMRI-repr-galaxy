from __future__ import annotations

import hashlib
import logging
from typing import List, Optional, Tuple

import numpy as np

from config import EmpiricalBayesConfig
from data_types import EBResult

logger = logging.getLogger(__name__)


class EmpiricalBayesGLM:
    """Liou/Lindley-Smith hierarchical GLM for run-specific stimulus effects.

    Inputs must already have nuisance regressors removed. Consequently each X_i
    contains only the stimulus coefficients that belong to beta_i and Omega.
    """

    def __init__(self, config: EmpiricalBayesConfig):
        self.config = config

    def fit(
        self,
        X_list: List[np.ndarray],
        y_list: List[np.ndarray],
        X_star: np.ndarray,
    ) -> EBResult:
        self._validate_inputs(X_list, y_list, X_star)
        n_restarts = self._resolve_n_restarts()

        beta_ols, sigma2_ols = self._ols_beta_sigma(X_list, y_list)
        current_init = self._initialize_from_beta_samples(
            beta_samples=np.vstack(beta_ols),
            sigma2=sigma2_ols,
            X_star=X_star,
        )

        results: list[EBResult] = []
        restart_infos: list[dict[str, object]] = []

        for restart_id in range(n_restarts):
            init_mode = "ols" if restart_id == 0 else "random"
            if restart_id == 0:
                B0, Omega0, sigma20 = current_init
            else:
                rng = self._rng_for_restart(y_list, restart_id)
                B0, Omega0, sigma20 = self._initialize_random(
                    beta_ols=beta_ols,
                    sigma2_ols=sigma2_ols,
                    X_star=X_star,
                    rng=rng,
                )

            result = self._fit_single_start(
                X_list=X_list,
                y_list=y_list,
                X_star=X_star,
                B0=B0,
                Omega0=Omega0,
                sigma20=sigma20,
            )
            result.restart_id = int(restart_id)
            result.init_mode = init_mode
            result.n_restarts = int(n_restarts)
            results.append(result)
            restart_infos.append(
                {
                    "restart_id": int(restart_id),
                    "init_mode": init_mode,
                    "converged": bool(result.converged),
                    "n_iter": int(result.n_iter),
                    "start_log_likelihood": float(result.start_log_likelihood),
                    "log_likelihood": float(result.log_likelihood),
                }
            )

        finite = [r for r in results if np.isfinite(r.log_likelihood)]
        converged = [r for r in finite if r.converged]
        pool = converged if converged else finite
        if not pool:
            # Return a deterministic result rather than hiding the failure. The
            # pipeline will invoke its explicit OLS fallback.
            result = results[0]
            result.converged = False
            result.diagnostics = dict(result.diagnostics or {})
            result.diagnostics["selection_error"] = "no_finite_restart"
            return result

        best = max(pool, key=lambda item: float(item.log_likelihood))
        best.restart_infos = restart_infos
        best.best_restart_converged = bool(best.converged)
        best.any_restart_converged = bool(converged)
        best.diagnostics = dict(best.diagnostics or {})
        best.diagnostics.update(
            {
                "n_restarts": int(n_restarts),
                "n_converged_restarts": int(len(converged)),
                "restart_infos": restart_infos,
            }
        )
        return best

    def _fit_single_start(
        self,
        X_list: List[np.ndarray],
        y_list: List[np.ndarray],
        X_star: np.ndarray,
        B0: np.ndarray,
        Omega0: np.ndarray,
        sigma20: np.ndarray,
    ) -> EBResult:
        B = np.asarray(B0, dtype=float).copy()
        Omega, omega_diag = self._regularize_cov_with_diagnostics(Omega0)
        sigma2 = np.maximum(
            np.asarray(sigma20, dtype=float).copy(),
            float(self.config.min_sigma2),
        )
        mu = X_star @ B

        start_ll = self._observed_log_likelihood(
            X_list, y_list, X_star, B, Omega, sigma2
        )
        previous_checked_ll = start_ll
        final_ll = start_ll
        converged = False
        iteration = 0
        check_interval = max(1, int(self.config.likelihood_check_interval))
        likelihood_tol = max(0.0, float(self.config.likelihood_tol))
        last_parameter_delta = float("inf")
        likelihood_decreases = 0
        final_raw_omega = Omega.copy()

        for iteration in range(1, int(self.config.max_iter) + 1):
            h, H = self._e_step(X_list, y_list, B, Omega, sigma2, mu)
            B_new, Omega_raw, Omega_new, sigma2_new, mu_new, omega_diag = self._m_step(
                X_list, y_list, X_star, h, H
            )

            last_parameter_delta = max(
                self._relative_delta(B_new, B),
                self._relative_delta(Omega_new, Omega),
                self._relative_delta(sigma2_new, sigma2),
            )

            B, Omega, sigma2, mu = B_new, Omega_new, sigma2_new, mu_new
            final_raw_omega = Omega_raw

            should_check = (
                iteration % check_interval == 0
                or last_parameter_delta < float(self.config.tol)
                or iteration == int(self.config.max_iter)
            )
            if not should_check:
                continue

            final_ll = self._observed_log_likelihood(
                X_list, y_list, X_star, B, Omega, sigma2
            )
            if (
                np.isfinite(previous_checked_ll)
                and np.isfinite(final_ll)
                and final_ll < previous_checked_ll - 1e-7 * (1.0 + abs(previous_checked_ll))
            ):
                likelihood_decreases += 1

            relative_ll_delta = self._relative_scalar_delta(
                final_ll, previous_checked_ll
            )
            previous_checked_ll = final_ll

            if self.config.verbose:
                logger.debug(
                    "EB iter=%d parameter_delta=%.3e ll=%.6f relative_ll_delta=%.3e",
                    iteration,
                    last_parameter_delta,
                    final_ll,
                    relative_ll_delta,
                )

            if (
                last_parameter_delta < float(self.config.tol)
                and relative_ll_delta < likelihood_tol
            ):
                converged = True
                break

        h, H = self._e_step(X_list, y_list, B, Omega, sigma2, mu)
        final_ll = self._observed_log_likelihood(
            X_list, y_list, X_star, B, Omega, sigma2
        )

        raw_eigs = np.linalg.eigvalsh(0.5 * (final_raw_omega + final_raw_omega.T))
        reg_eigs = np.linalg.eigvalsh(Omega)
        diagnostics = {
            "omega_dimension": int(Omega.shape[0]),
            "omega_raw_rank": int(np.linalg.matrix_rank(final_raw_omega)),
            "omega_regularized_rank": int(np.linalg.matrix_rank(Omega)),
            "omega_raw_eigenvalues": raw_eigs.tolist(),
            "omega_regularized_eigenvalues": reg_eigs.tolist(),
            "omega_regularized_condition_number": float(np.linalg.cond(Omega)),
            "omega_shrinkage_alpha": float(self.config.omega_shrinkage_alpha),
            "omega_shrinkage_target": str(self.config.omega_shrinkage_target),
            "omega_eigenvalue_floor": float(omega_diag["eigenvalue_floor"]),
            "last_parameter_delta": float(last_parameter_delta),
            "likelihood_decreases": int(likelihood_decreases),
        }

        return EBResult(
            h=h,
            H=H,
            B=B,
            Omega=Omega,
            sigma2=sigma2,
            mu=mu,
            n_iter=int(iteration),
            converged=bool(converged),
            log_likelihood=float(final_ll),
            start_log_likelihood=float(start_ll),
            bayes_converged=bool(converged),
            diagnostics=diagnostics,
        )

    def _resolve_n_restarts(self) -> int:
        try:
            return max(1, int(self.config.n_restarts))
        except Exception:
            return 1

    def _ols_beta_sigma(
        self,
        X_list: List[np.ndarray],
        y_list: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        betas: list[np.ndarray] = []
        sigma2: list[float] = []
        for X, y in zip(X_list, y_list):
            beta = np.linalg.pinv(X) @ y
            resid = y - X @ beta
            df = max(1, X.shape[0] - np.linalg.matrix_rank(X))
            value = max(float(resid @ resid) / df, float(self.config.min_sigma2))
            betas.append(np.asarray(beta, dtype=float))
            sigma2.append(value)
        return betas, np.asarray(sigma2, dtype=float)

    def _initialize_from_beta_samples(
        self,
        beta_samples: np.ndarray,
        sigma2: np.ndarray,
        X_star: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Z = np.asarray(beta_samples, dtype=float)
        XtX = X_star.T @ X_star
        if float(self.config.b_ridge) > 0.0:
            XtX = XtX + float(self.config.b_ridge) * np.eye(XtX.shape[0])
        B = np.linalg.pinv(XtX) @ X_star.T @ Z
        residual = Z - X_star @ B
        if residual.shape[0] > 1:
            Omega_raw = residual.T @ residual / max(1, residual.shape[0] - 1)
        else:
            Omega_raw = np.eye(Z.shape[1], dtype=float)
        Omega, _ = self._regularize_cov_with_diagnostics(Omega_raw)
        sigma2 = np.maximum(np.asarray(sigma2, dtype=float), self.config.min_sigma2)
        return B, Omega, sigma2

    def _initialize_random(
        self,
        beta_ols: List[np.ndarray],
        sigma2_ols: np.ndarray,
        X_star: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Z_ols = np.vstack(beta_ols)
        finite = Z_ols[np.isfinite(Z_ols)]
        scale = float(np.nanmedian(np.abs(finite))) if finite.size else 1.0
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(np.nanstd(finite)) if finite.size else 1.0
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        scale *= max(float(self.config.random_init_scale), 1e-6)
        Z_random = Z_ols + rng.normal(0.0, 0.25 * scale, size=Z_ols.shape)
        B, Omega, _ = self._initialize_from_beta_samples(
            Z_random, sigma2_ols, X_star
        )
        sigma2 = np.maximum(
            sigma2_ols * rng.lognormal(0.0, 0.15, size=len(sigma2_ols)),
            self.config.min_sigma2,
        )
        return B, Omega, sigma2

    def _rng_for_restart(
        self,
        y_list: List[np.ndarray],
        restart_id: int,
    ) -> np.random.Generator:
        base_seed = self.config.random_seed
        if base_seed is None:
            return np.random.default_rng()
        h = hashlib.blake2b(digest_size=8)
        h.update(str(int(base_seed)).encode())
        h.update(str(int(restart_id)).encode())
        for y in y_list:
            arr = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
            h.update(arr.shape.__repr__().encode())
            h.update(arr.tobytes())
        return np.random.default_rng(
            int.from_bytes(h.digest(), byteorder="little", signed=False)
        )

    def _e_step(
        self,
        X_list: List[np.ndarray],
        y_list: List[np.ndarray],
        B: np.ndarray,
        Omega: np.ndarray,
        sigma2: np.ndarray,
        mu: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        M = len(X_list)
        k = X_list[0].shape[1]
        h = np.zeros((M, k), dtype=float)
        H = np.zeros((M, k, k), dtype=float)
        Omega_inv = self._inv_stable(Omega)

        for i, (X, y) in enumerate(zip(X_list, y_list)):
            sigma_i = max(float(sigma2[i]), float(self.config.min_sigma2))
            precision = Omega_inv + (X.T @ X) / sigma_i
            precision += float(self.config.omega_ridge) * np.eye(k)
            H_i = self._inv_stable(precision)
            rhs = Omega_inv @ mu[i] + (X.T @ y) / sigma_i
            h[i] = H_i @ rhs
            H[i] = H_i
        return h, H

    def _m_step(
        self,
        X_list: List[np.ndarray],
        y_list: List[np.ndarray],
        X_star: np.ndarray,
        h: np.ndarray,
        H: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        M, k = h.shape
        XtX = X_star.T @ X_star
        if float(self.config.b_ridge) > 0.0:
            XtX += float(self.config.b_ridge) * np.eye(XtX.shape[0])
        B_new = np.linalg.pinv(XtX) @ X_star.T @ h
        mu_new = X_star @ B_new

        Omega_num = np.zeros((k, k), dtype=float)
        for i in range(M):
            diff = h[i] - mu_new[i]
            Omega_num += H[i] + np.outer(diff, diff)

        if self.config.omega_denominator == "liou_reml":
            denominator = max(1, M - np.linalg.matrix_rank(X_star))
        elif self.config.omega_denominator == "ml":
            denominator = max(1, M)
        else:
            raise ValueError(
                f"Unknown omega_denominator: {self.config.omega_denominator}"
            )
        Omega_raw = Omega_num / denominator
        Omega_new, omega_diag = self._regularize_cov_with_diagnostics(Omega_raw)

        sigma2_new = np.zeros(M, dtype=float)
        for i, (X, y) in enumerate(zip(X_list, y_list)):
            resid = y - X @ h[i]
            if self.config.sigma_update == "liou_plugin":
                df = max(1, X.shape[0] - np.linalg.matrix_rank(X))
                value = float(resid @ resid) / df
            elif self.config.sigma_update == "strict_em":
                value = (
                    float(resid @ resid) + float(np.trace(X @ H[i] @ X.T))
                ) / X.shape[0]
            else:
                raise ValueError(f"Unknown sigma_update: {self.config.sigma_update}")
            sigma2_new[i] = max(value, float(self.config.min_sigma2))

        return B_new, Omega_raw, Omega_new, sigma2_new, mu_new, omega_diag

    def _regularize_cov_with_diagnostics(
        self,
        matrix: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        raw = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
        k = raw.shape[0]
        alpha = float(np.clip(self.config.omega_shrinkage_alpha, 0.0, 1.0))
        target_name = str(self.config.omega_shrinkage_target).lower().strip()

        if target_name in {"diagonal", "diag"}:
            target = np.diag(np.maximum(np.diag(raw), 0.0))
        elif target_name in {"scaled_identity", "identity", "isotropic"}:
            scale = max(float(np.trace(raw)) / max(1, k), 0.0)
            target = scale * np.eye(k)
        else:
            raise ValueError(
                "omega_shrinkage_target must be 'diagonal' or 'scaled_identity'"
            )

        regularized = (1.0 - alpha) * raw + alpha * target
        scale = max(
            float(np.trace(np.abs(regularized))) / max(1, k),
            float(self.config.omega_ridge),
        )
        floor = max(
            float(self.config.omega_ridge),
            float(self.config.omega_eigenvalue_floor_relative) * scale,
        )
        eigvals, eigvecs = np.linalg.eigh(0.5 * (regularized + regularized.T))
        eigvals = np.maximum(eigvals, floor)
        regularized = (eigvecs * eigvals) @ eigvecs.T
        regularized = 0.5 * (regularized + regularized.T)
        return regularized, {"eigenvalue_floor": float(floor)}

    def _observed_log_likelihood(
        self,
        X_list: List[np.ndarray],
        y_list: List[np.ndarray],
        X_star: np.ndarray,
        B: np.ndarray,
        Omega: np.ndarray,
        sigma2: np.ndarray,
    ) -> float:
        Omega_inv = self._inv_stable(Omega)
        sign_omega, logdet_omega = np.linalg.slogdet(Omega)
        if sign_omega <= 0 or not np.isfinite(logdet_omega):
            return float("-inf")

        total = 0.0
        for i, (X, y) in enumerate(zip(X_list, y_list)):
            sigma_i = max(float(sigma2[i]), float(self.config.min_sigma2))
            n = X.shape[0]
            residual = y - X @ (X_star[i] @ B)
            precision = Omega_inv + (X.T @ X) / sigma_i
            sign_precision, logdet_precision = np.linalg.slogdet(precision)
            if sign_precision <= 0 or not np.isfinite(logdet_precision):
                return float("-inf")
            H_i = self._inv_stable(precision)
            XTr = X.T @ residual
            logdet_v = n * np.log(sigma_i) + logdet_omega + logdet_precision
            quad = float(
                (residual @ residual) / sigma_i
                - (XTr @ H_i @ XTr) / (sigma_i**2)
            )
            if not np.isfinite(logdet_v) or not np.isfinite(quad):
                return float("-inf")
            total += -0.5 * (n * np.log(2.0 * np.pi) + logdet_v + quad)
        return float(total)

    @staticmethod
    def _relative_delta(new: np.ndarray, old: np.ndarray) -> float:
        numerator = float(np.max(np.abs(np.asarray(new) - np.asarray(old))))
        denominator = 1.0 + float(np.max(np.abs(np.asarray(old))))
        return numerator / denominator

    @staticmethod
    def _relative_scalar_delta(new: float, old: float) -> float:
        if not np.isfinite(new) or not np.isfinite(old):
            return float("inf")
        return abs(new - old) / (1.0 + abs(old))

    @staticmethod
    def _inv_stable(matrix: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(matrix)

    @staticmethod
    def _validate_inputs(
        X_list: List[np.ndarray],
        y_list: List[np.ndarray],
        X_star: np.ndarray,
    ) -> None:
        if not X_list:
            raise ValueError("X_list is empty")
        if len(X_list) != len(y_list):
            raise ValueError("X_list and y_list have different lengths")
        if X_star.ndim != 2 or X_star.shape[0] != len(X_list):
            raise ValueError("X_star must be a 2D matrix with one row per run")
        k = X_list[0].shape[1]
        for i, (X, y) in enumerate(zip(X_list, y_list)):
            if X.ndim != 2 or y.ndim != 1:
                raise ValueError(f"Invalid dimensions for run {i}: X={X.shape}, y={y.shape}")
            if X.shape[0] != len(y):
                raise ValueError(f"X/y length mismatch for run {i}")
            if X.shape[1] != k:
                raise ValueError("All X_i must have the same stimulus dimension")
            if np.linalg.matrix_rank(X) < k:
                raise ValueError(f"Stimulus design is rank deficient in run {i}")
