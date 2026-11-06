from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import logging
import multiprocessing as mp
import os
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from config import AnalysisConfig
from data_types import ROCResult

logger = logging.getLogger(__name__)


def _normalize_probability_vector(x: np.ndarray, eps: float) -> np.ndarray:
    values = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, eps)
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0.0:
        values = np.ones_like(values)
        total = float(values.sum())
    return values / total


def _apply_lambda_constraint_value(
    lambda_value: float,
    lambda_mode: str,
    lambda_upper_bound: float,
    eps: float,
) -> float:
    value = float(np.clip(lambda_value, eps, 1.0 - eps))
    if lambda_mode == "none":
        return value
    if lambda_mode == "indicator_upper_bound":
        if not 0.0 < lambda_upper_bound < 1.0:
            raise ValueError("lambda_upper_bound must be in (0,1)")
        return float(np.clip(value, eps, min(lambda_upper_bound, 1.0 - eps)))
    raise ValueError(f"Unknown lambda_likelihood_mode={lambda_mode}")


def _tail_probabilities(p: np.ndarray) -> np.ndarray:
    """Right-tail probabilities for all K thresholds (exclude the whole support)."""
    p = np.asarray(p, dtype=float)
    return np.cumsum(p[::-1])[::-1][1:]


def _dominance_margins(pA: np.ndarray, pI: np.ndarray) -> np.ndarray:
    return _tail_probabilities(pA) - _tail_probabilities(pI)


def _dominance_valid(pA: np.ndarray, pI: np.ndarray, margin: float) -> bool:
    margins = _dominance_margins(pA, pI)
    return bool(np.all(np.isfinite(margins)) and np.min(margins) >= margin - 1e-10)


def _ordered_random_initialization(
    n_bins: int,
    rng: np.random.Generator,
    eps: float,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate an MLR-ordered pair, which implies stochastic dominance."""
    pI = _normalize_probability_vector(rng.gamma(1.5, 1.0, n_bins), eps)
    index = np.linspace(0.0, 1.0, n_bins)
    for _ in range(100):
        theta = float(rng.uniform(1.0, 6.0))
        pA = _normalize_probability_vector(pI * np.exp(theta * index), eps)
        if _dominance_valid(pA, pI, margin):
            return pA, pI
    # Deterministic separated fallback.
    return _separated_initialization(n_bins, rng, eps, margin, randomize=True)


def _separated_initialization(
    n_bins: int,
    rng: np.random.Generator,
    eps: float,
    margin: float,
    randomize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if n_bins < 2:
        raise ValueError("ROC needs at least two bins")
    split = n_bins // 2
    if randomize and n_bins >= 4:
        split = int(rng.integers(1, n_bins))
    split = max(1, min(split, n_bins - 1))
    pI = np.full(n_bins, eps, dtype=float)
    pA = np.full(n_bins, eps, dtype=float)
    pI[:split] += rng.random(split) + 0.5 if randomize else 1.0
    pA[split:] += rng.random(n_bins - split) + 0.5 if randomize else 1.0
    pA = _normalize_probability_vector(pA, eps)
    pI = _normalize_probability_vector(pI, eps)
    if not _dominance_valid(pA, pI, margin):
        # Strong deterministic end-point pair always satisfies the ordering.
        pI = np.full(n_bins, eps, dtype=float)
        pA = np.full(n_bins, eps, dtype=float)
        pI[0] = 1.0
        pA[-1] = 1.0
        pA = _normalize_probability_vector(pA, eps)
        pI = _normalize_probability_vector(pI, eps)
    return pA, pI


def _initial_probability_vectors(
    n_bins: int,
    rng: np.random.Generator,
    eps: float,
    init_mode: str,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    mode = str(init_mode).lower().strip()
    if mode in {"random", "dirichlet", "simplex", "ordered_random", "ordered"}:
        return _ordered_random_initialization(n_bins, rng, eps, margin)
    if mode in {"split_random", "random_split", "random_separated"}:
        return _separated_initialization(n_bins, rng, eps, margin, randomize=True)
    if mode in {"ideal", "ideal_split", "separated"}:
        return _separated_initialization(n_bins, rng, eps, margin, randomize=False)
    if mode in {"uniform", "flat", "uniform_perturbation", "perturbed_uniform"}:
        # Symmetric uniform starts are not identifiable. Use a very weak ordered tilt.
        pI = np.ones(n_bins, dtype=float) / n_bins
        theta = max(0.05, 10.0 * margin)
        pA = _normalize_probability_vector(
            pI * np.exp(theta * np.linspace(0.0, 1.0, n_bins)), eps
        )
        if not _dominance_valid(pA, pI, margin):
            return _ordered_random_initialization(n_bins, rng, eps, margin)
        return pA, pI
    raise ValueError(f"Unknown ROC init mode: {init_mode}")


def _roc_log_likelihood(
    counts: np.ndarray,
    lambda_: float,
    pA: np.ndarray,
    pI: np.ndarray,
    lambda_likelihood_mode: str,
    lambda_upper_bound: float,
    eps: float,
) -> float:
    if lambda_likelihood_mode == "indicator_upper_bound" and lambda_ > lambda_upper_bound + 1e-15:
        return float("-inf")
    if not 0.0 < lambda_ < 1.0:
        return float("-inf")
    if np.any(pA <= 0.0) or np.any(pI <= 0.0):
        return float("-inf")
    if not np.isclose(np.sum(pA), 1.0, atol=1e-7) or not np.isclose(np.sum(pI), 1.0, atol=1e-7):
        return float("-inf")
    logA = counts @ np.log(pA + eps)
    logI = counts @ np.log(pI + eps)
    ll = np.sum(
        logsumexp(
            np.column_stack(
                [np.log(lambda_ + eps) + logA, np.log(1.0 - lambda_ + eps) + logI]
            ),
            axis=1,
        )
    )
    return float(ll)


def _q_probability_part(
    weighted_A: np.ndarray,
    weighted_I: np.ndarray,
    pA: np.ndarray,
    pI: np.ndarray,
    eps: float,
) -> float:
    return float(weighted_A @ np.log(pA + eps) + weighted_I @ np.log(pI + eps))


def _ordered_gem_probability_update(
    weighted_A: np.ndarray,
    weighted_I: np.ndarray,
    old_pA: np.ndarray,
    old_pI: np.ndarray,
    eps: float,
    margin: float,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    """Constrained GEM update with a fast feasible line step and SLSQP fallback."""
    mle_pA = _normalize_probability_vector(weighted_A, eps)
    mle_pI = _normalize_probability_vector(weighted_I, eps)
    if _dominance_valid(mle_pA, mle_pI, margin):
        return mle_pA, mle_pI, "unconstrained_feasible", 1.0

    old_margin = _dominance_margins(old_pA, old_pI)
    new_margin = _dominance_margins(mle_pA, mle_pI)
    delta = new_margin - old_margin
    alpha = 1.0
    for old_value, change in zip(old_margin, delta):
        if change < 0.0:
            bound = (old_value - margin) / (-change)
            alpha = min(alpha, float(bound))
    alpha = float(np.clip(0.995 * alpha, 0.0, 1.0))

    if alpha >= 1e-8:
        pA = (1.0 - alpha) * old_pA + alpha * mle_pA
        pI = (1.0 - alpha) * old_pI + alpha * mle_pI
        if _dominance_valid(pA, pI, margin):
            return pA, pI, "feasible_line_gem", alpha

    # At an active dominance boundary the direct line to the unconstrained MLE
    # can have no feasible positive step.  A generalized EM step is allowed to
    # keep the current feasible point; the outer convergence check then stops
    # this restart.  Multiple ordered starts provide the search over boundary
    # optima without an expensive generic optimizer at every voxel/report.
    return old_pA.copy(), old_pI.copy(), "stalled_at_constraint", 0.0


def _run_one_mixture_restart(args: Dict[str, Any]) -> Dict[str, Any]:
    restart_id = int(args["restart_id"])
    counts = np.asarray(args["counts"], dtype=float)
    max_iter = int(args["max_iter"])
    tol = float(args["tol"])
    lambda_likelihood_mode = str(args["lambda_likelihood_mode"])
    lambda_optimization_mode = str(args["lambda_optimization_mode"])
    lambda_upper_bound = float(args["lambda_upper_bound"])
    fixed_lambda = float(args["fixed_lambda"])
    lambda_init = float(args["lambda_init"])
    eps = float(args["eps"])
    seed = int(args["seed"])
    init_mode = str(args["init_mode"])
    use_dominance = bool(args["use_dominance"])
    margin = float(args["dominance_margin"]) if use_dominance else -np.inf

    try:
        _, n_bins = counts.shape
        rng = np.random.default_rng(seed)
        if lambda_optimization_mode == "fixed":
            lambda_ = _apply_lambda_constraint_value(
                fixed_lambda, lambda_likelihood_mode, lambda_upper_bound, eps
            )
        elif lambda_optimization_mode == "free":
            lambda_ = _apply_lambda_constraint_value(
                lambda_init, lambda_likelihood_mode, lambda_upper_bound, eps
            )
        else:
            raise ValueError("lambda_optimization_mode must be free or fixed")

        pA, pI = _initial_probability_vectors(n_bins, rng, eps, init_mode, max(0.0, margin))
        if use_dominance and not _dominance_valid(pA, pI, margin):
            raise RuntimeError("Could not create a dominance-valid initialization")

        start_ll = _roc_log_likelihood(
            counts, lambda_, pA, pI, lambda_likelihood_mode, lambda_upper_bound, eps
        )
        previous_ll = start_ll
        gamma = np.full(counts.shape[0], lambda_, dtype=float)
        converged = False
        update_types: dict[str, int] = {}
        monotonic_backtracks = 0
        iteration = 0

        for iteration in range(1, max_iter + 1):
            old_lambda, old_pA, old_pI = lambda_, pA.copy(), pI.copy()
            logA = counts @ np.log(pA + eps)
            logI = counts @ np.log(pI + eps)
            log_active = np.log(lambda_ + eps) + logA
            log_inactive = np.log(1.0 - lambda_ + eps) + logI
            denominator = logsumexp(np.column_stack([log_active, log_inactive]), axis=1)
            gamma_new = np.exp(log_active - denominator)

            if lambda_optimization_mode == "fixed":
                lambda_candidate = lambda_
            else:
                lambda_candidate = _apply_lambda_constraint_value(
                    float(np.mean(gamma_new)),
                    lambda_likelihood_mode,
                    lambda_upper_bound,
                    eps,
                )

            weighted_A = (gamma_new[:, None] * counts).sum(axis=0)
            weighted_I = ((1.0 - gamma_new)[:, None] * counts).sum(axis=0)
            if use_dominance:
                pA_candidate, pI_candidate, update_type, step = _ordered_gem_probability_update(
                    weighted_A, weighted_I, pA, pI, eps, margin
                )
            else:
                pA_candidate = _normalize_probability_vector(weighted_A, eps)
                pI_candidate = _normalize_probability_vector(weighted_I, eps)
                update_type, step = "unconstrained", 1.0
            update_types[update_type] = update_types.get(update_type, 0) + 1

            candidate_ll = _roc_log_likelihood(
                counts,
                lambda_candidate,
                pA_candidate,
                pI_candidate,
                lambda_likelihood_mode,
                lambda_upper_bound,
                eps,
            )

            # Numerical safeguard for the EM/GEM monotonicity property.
            if candidate_ll < previous_ll - 1e-8 * (1.0 + abs(previous_ll)):
                accepted = False
                alpha = 0.5
                for _ in range(40):
                    trial_lambda = (1.0 - alpha) * old_lambda + alpha * lambda_candidate
                    trial_pA = (1.0 - alpha) * old_pA + alpha * pA_candidate
                    trial_pI = (1.0 - alpha) * old_pI + alpha * pI_candidate
                    if use_dominance and not _dominance_valid(trial_pA, trial_pI, margin):
                        alpha *= 0.5
                        continue
                    trial_ll = _roc_log_likelihood(
                        counts,
                        trial_lambda,
                        trial_pA,
                        trial_pI,
                        lambda_likelihood_mode,
                        lambda_upper_bound,
                        eps,
                    )
                    if trial_ll >= previous_ll - 1e-10:
                        lambda_candidate, pA_candidate, pI_candidate = trial_lambda, trial_pA, trial_pI
                        candidate_ll = trial_ll
                        accepted = True
                        monotonic_backtracks += 1
                        break
                    alpha *= 0.5
                if not accepted:
                    lambda_candidate, pA_candidate, pI_candidate = old_lambda, old_pA, old_pI
                    candidate_ll = previous_ll

            delta = max(
                abs(lambda_candidate - lambda_),
                float(np.max(np.abs(pA_candidate - pA))),
                float(np.max(np.abs(pI_candidate - pI))),
            )
            ll_delta = abs(candidate_ll - previous_ll) / (1.0 + abs(previous_ll))
            lambda_, pA, pI, gamma = lambda_candidate, pA_candidate, pI_candidate, gamma_new
            previous_ll = candidate_ll
            if delta < tol and ll_delta < tol:
                converged = True
                break

        margins = _dominance_margins(pA, pI)
        dominance_valid = bool(not use_dominance or _dominance_valid(pA, pI, margin))
        return {
            "ok": bool(dominance_valid),
            "restart_id": restart_id,
            "seed": seed,
            "init_mode": init_mode,
            "lambda": float(lambda_),
            "pA": pA,
            "pI": pI,
            "gamma": gamma,
            "converged": bool(converged),
            "n_iter": int(iteration),
            "log_likelihood": float(previous_ll),
            "start_log_likelihood": float(start_ll),
            "dominance_valid": dominance_valid,
            "dominance_min_margin": float(np.min(margins)) if len(margins) else 0.0,
            "update_types": update_types,
            "monotonic_backtracks": int(monotonic_backtracks),
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "restart_id": restart_id,
            "seed": seed,
            "init_mode": init_mode,
            "lambda": np.nan,
            "pA": None,
            "pI": None,
            "gamma": None,
            "converged": False,
            "n_iter": 0,
            "log_likelihood": float("-inf"),
            "start_log_likelihood": float("-inf"),
            "dominance_valid": False,
            "dominance_min_margin": float("-inf"),
            "update_types": {},
            "monotonic_backtracks": 0,
            "error": repr(exc),
        }


class MixtureMultinomialROC:
    """Liou empirical ROC with identified active/inactive components."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.last_fit_info: Dict[str, Any] = {}

    @staticmethod
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
            "unsigned": "absolute",
            "absolute_liou": "absolute",
        }
        if value not in aliases:
            raise ValueError(f"Unknown reproducibility direction: {direction}")
        return aliases[value]

    @classmethod
    def evidence(cls, t_runwise: np.ndarray, direction: str) -> np.ndarray:
        direction = cls._normalize_direction(direction)
        t = np.asarray(t_runwise, dtype=float)
        if direction == "positive":
            return np.maximum(t, 0.0)
        if direction == "negative":
            return np.maximum(-t, 0.0)
        return np.abs(t)

    def fit(
        self,
        t_runwise: np.ndarray,
        lambda_optimization_mode: Optional[str] = None,
        direction: str = "positive",
    ) -> ROCResult:
        direction = self._normalize_direction(direction)
        evidence = self.evidence(t_runwise, direction)
        thresholds = self.choose_thresholds(evidence)
        counts = self.count_bins(evidence, thresholds)
        if lambda_optimization_mode is None:
            modes = self.lambda_optimization_modes()
            lambda_optimization_mode = "free" if "free" in modes else modes[0]

        lambda_, pA, pI, gamma, converged, n_iter, info = self._fit_mixture(
            counts, lambda_optimization_mode
        )
        result = ROCResult(
            thresholds=thresholds,
            counts=counts,
            lambda_=lambda_,
            p_active_bins=pA,
            p_inactive_bins=pI,
            gamma=gamma,
            converged=converged,
            n_iter=n_iter,
            direction=direction,
        )
        _, sensitivity, false_alarm = self.roc_points(result)
        result.stochastic_dominance_valid = bool(info["stochastic_dominance_valid"])
        result.dominance_min_margin = float(info["dominance_min_margin"])
        result.auc = self.auc(false_alarm, sensitivity)
        result.optimization_info = info
        result.lambda_optimization_mode = str(lambda_optimization_mode)
        self.last_fit_info = info
        return result

    def choose_thresholds(self, evidence: np.ndarray) -> np.ndarray:
        finite = np.asarray(evidence, dtype=float)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            raise ValueError("No finite statistics for ROC thresholds")
        K = max(1, int(self.config.n_thresholds))
        mode = str(self.config.threshold_grid).lower().strip()
        if mode == "fixed":
            low = float(self.config.fixed_threshold_min)
            high = float(self.config.fixed_threshold_max)
            if high <= low:
                raise ValueError("fixed_threshold_max must exceed fixed_threshold_min")
            # Include the configured endpoints to create one common scientific scale.
            thresholds = np.linspace(low, high, K, dtype=float)
        elif mode == "linspace":
            thresholds = np.linspace(float(np.min(finite)), float(np.max(finite)), K)
        elif mode == "quantiles":
            thresholds = np.quantile(finite, np.linspace(0.0, 1.0, K + 2)[1:-1])
        else:
            raise ValueError(f"Unknown threshold_grid={mode}")
        thresholds = np.unique(np.asarray(thresholds, dtype=float))
        thresholds = thresholds[np.isfinite(thresholds) & (thresholds >= 0.0)]
        if not len(thresholds):
            raise ValueError("Empty threshold grid")
        logger.info(
            "Threshold grid: mode=%s, n=%d, min=%.6f, max=%.6f",
            mode,
            len(thresholds),
            float(thresholds.min()),
            float(thresholds.max()),
        )
        return thresholds

    @staticmethod
    def count_bins(evidence: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        values = np.asarray(evidence, dtype=float)
        _, V = values.shape
        counts = np.zeros((V, len(thresholds) + 1), dtype=float)
        for v in range(V):
            finite = np.isfinite(values[:, v])
            classes = np.searchsorted(thresholds, values[finite, v], side="right")
            counts[v] = np.bincount(classes, minlength=len(thresholds) + 1)
        return counts

    def _lambda_likelihood_mode(self) -> str:
        mode = str(getattr(self.config, "lambda_likelihood_mode", "indicator_upper_bound")).lower().strip()
        if mode == "none" and bool(getattr(self.config, "use_lambda_upper_bound", False)):
            return "indicator_upper_bound"
        if mode not in {"none", "indicator_upper_bound"}:
            raise ValueError(f"Unknown lambda likelihood mode: {mode}")
        return mode

    def lambda_optimization_modes(self) -> List[str]:
        raw = getattr(self.config, "roc_lambda_optimization_mode", "both")
        if isinstance(raw, (list, tuple)):
            items = [str(x).lower().strip() for x in raw]
        else:
            text = str(raw).lower().strip()
            items = ["free", "fixed"] if text == "both" else [x.strip() for x in text.split(",")]
        result: List[str] = []
        for item in items:
            normalized = "free" if item in {"free", "variable", "estimated"} else "fixed" if item in {"fixed", "constant"} else None
            if normalized is None:
                raise ValueError(f"Invalid lambda optimization mode: {item}")
            if normalized not in result:
                result.append(normalized)
        return result or ["free"]

    def primary_lambda_optimization_mode(self, available_modes: Optional[List[str]] = None) -> str:
        available = available_modes or self.lambda_optimization_modes()
        raw = str(getattr(self.config, "roc_primary_lambda_mode", "best_valid_kappa")).lower().strip()
        if raw in {"best", "auto", "best_kappa", "best_valid_kappa"}:
            return "best_valid_kappa"
        return raw if raw in available else available[0]

    def _init_modes(self) -> List[str]:
        raw = getattr(self.config, "roc_init_modes", ["ordered_random", "split_random"])
        if isinstance(raw, (list, tuple)):
            return [str(x) for x in raw] or ["ordered_random"]
        return [x.strip() for x in str(raw).split(",") if x.strip()] or ["ordered_random"]

    def _eps(self) -> float:
        return max(1e-15, float(getattr(self.config, "roc_min_probability", 1e-12)))

    def _roc_n_restarts(self) -> int:
        return max(1, int(getattr(self.config, "roc_n_restarts", 30)))

    def _roc_restart_jobs(self, n_restarts: int) -> int:
        if not bool(getattr(self.config, "roc_parallel_restarts", True)):
            return 1
        raw = getattr(self.config, "roc_restart_jobs", None)
        jobs = int(raw if raw is not None else getattr(self.config, "n_jobs", 1))
        if jobs == -1:
            jobs = os.cpu_count() or 1
        return max(1, min(jobs, os.cpu_count() or 1, n_restarts))

    def _restart_seeds(self, n_restarts: int) -> List[int]:
        seed = getattr(self.config, "roc_random_seed", 12345)
        sequence = np.random.SeedSequence() if seed is None else np.random.SeedSequence(int(seed))
        return [int(child.generate_state(1)[0]) for child in sequence.spawn(n_restarts)]

    def _restart_backend(self) -> str:
        value = str(getattr(self.config, "roc_parallel_backend", "thread")).lower().strip()
        return "process" if value in {"process", "processes", "multiprocessing"} else "thread"

    def _fit_mixture(
        self,
        counts: np.ndarray,
        lambda_optimization_mode: str,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, bool, int, Dict[str, Any]]:
        counts = np.asarray(counts, dtype=float)
        eps = self._eps()
        n_restarts = self._roc_n_restarts()
        jobs = self._roc_restart_jobs(n_restarts)
        backend = self._restart_backend()
        likelihood_mode = self._lambda_likelihood_mode()
        upper = float(self.config.lambda_upper_bound)
        fixed_requested = float(self.config.roc_fixed_lambda)
        fixed = _apply_lambda_constraint_value(fixed_requested, likelihood_mode, upper, eps)
        lambda_init = _apply_lambda_constraint_value(float(self.config.lambda_init), likelihood_mode, upper, eps)
        use_dominance = bool(getattr(self.config, "roc_stochastic_dominance", True))
        margin = max(0.0, float(getattr(self.config, "roc_dominance_margin", 0.0)))
        modes = self._init_modes()
        seeds = self._restart_seeds(n_restarts)

        args_list = [
            {
                "restart_id": i,
                "counts": counts,
                "max_iter": int(self.config.roc_max_iter),
                "tol": float(self.config.roc_tol),
                "lambda_likelihood_mode": likelihood_mode,
                "lambda_optimization_mode": lambda_optimization_mode,
                "lambda_upper_bound": upper,
                "fixed_lambda": fixed,
                "lambda_init": lambda_init,
                "eps": eps,
                "seed": seeds[i],
                "init_mode": modes[i % len(modes)],
                "use_dominance": use_dominance,
                "dominance_margin": margin,
            }
            for i in range(n_restarts)
        ]

        logger.info(
            "ROC constrained mixture: lambda_mode=%s, dominance=%s, margin=%.3g, restarts=%d",
            lambda_optimization_mode,
            use_dominance,
            margin,
            n_restarts,
        )
        results: List[Dict[str, Any]] = []
        if jobs <= 1:
            results = [_run_one_mixture_restart(args) for args in args_list]
        else:
            executor_cls = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
            kwargs: dict[str, Any] = {"max_workers": jobs}
            if backend == "process":
                kwargs["mp_context"] = mp.get_context("spawn")
            with executor_cls(**kwargs) as executor:
                futures = [executor.submit(_run_one_mixture_restart, args) for args in args_list]
                for future in as_completed(futures):
                    results.append(future.result())

        successful = [
            item for item in results
            if item.get("ok") and np.isfinite(float(item.get("log_likelihood", -np.inf)))
        ]
        if not successful:
            raise RuntimeError(
                "All constrained ROC restarts failed. "
                + "; ".join(str(item.get("error")) for item in results[:5])
            )
        converged = [item for item in successful if item.get("converged")]
        best = max(converged or successful, key=lambda item: float(item["log_likelihood"]))
        finite_ll = np.asarray([float(item["log_likelihood"]) for item in successful])
        fit_info = {
            "optimizer": "constrained_multistart_gem",
            "lambda_optimization_mode": lambda_optimization_mode,
            "lambda_likelihood_mode": likelihood_mode,
            "stochastic_dominance_enforced": use_dominance,
            "stochastic_dominance_valid": bool(best["dominance_valid"]),
            "dominance_min_margin": float(best["dominance_min_margin"]),
            "dominance_required_margin": float(margin),
            "n_restarts": n_restarts,
            "successful_restarts": len(successful),
            "converged_restarts": len(converged),
            "failed_restarts": n_restarts - len(successful),
            "restart_jobs": jobs,
            "parallel_backend": backend,
            "best_restart_id": int(best["restart_id"]),
            "best_init_mode": str(best["init_mode"]),
            "start_log_likelihood": float(best["start_log_likelihood"]),
            "log_likelihood": float(best["log_likelihood"]),
            "delta_log_likelihood": float(best["log_likelihood"] - best["start_log_likelihood"]),
            "log_likelihood_min": float(finite_ll.min()),
            "log_likelihood_median": float(np.median(finite_ll)),
            "log_likelihood_max": float(finite_ll.max()),
            "fixed_lambda_requested": fixed_requested,
            "fixed_lambda_used": fixed,
            "update_types": best.get("update_types", {}),
            "monotonic_backtracks": int(best.get("monotonic_backtracks", 0)),
        }
        return (
            float(best["lambda"]),
            np.asarray(best["pA"], dtype=float),
            np.asarray(best["pI"], dtype=float),
            np.asarray(best["gamma"], dtype=float),
            bool(best["converged"]),
            int(best["n_iter"]),
            fit_info,
        )

    @staticmethod
    def roc_points(result: ROCResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sensitivity = _tail_probabilities(result.p_active_bins)
        false_alarm = _tail_probabilities(result.p_inactive_bins)
        return result.thresholds, sensitivity, false_alarm

    @staticmethod
    def auc(false_alarm: np.ndarray, sensitivity: np.ndarray) -> float:
        fp = np.asarray(false_alarm, dtype=float)
        tp = np.asarray(sensitivity, dtype=float)
        valid = np.isfinite(fp) & np.isfinite(tp)
        fp, tp = fp[valid], tp[valid]
        if not len(fp):
            return float("nan")
        fp = np.concatenate([[0.0], fp, [1.0]])
        tp = np.concatenate([[0.0], tp, [1.0]])
        order = np.argsort(fp)
        return float(np.trapezoid(tp[order], fp[order]))


class EnglandEROC:
    @staticmethod
    def tp(fp: np.ndarray, E: float, R: float) -> np.ndarray:
        fp = np.asarray(fp, dtype=float)
        return R * fp ** (1.0 / E) + (1.0 - R) * (1.0 - (1.0 - fp) ** E)

    @staticmethod
    def fit(
        false_alarm: np.ndarray,
        sensitivity: np.ndarray,
        loss: str = "rmse",
    ) -> tuple[float, float]:
        fp = np.asarray(false_alarm, dtype=float)
        tp = np.asarray(sensitivity, dtype=float)
        valid = np.isfinite(fp) & np.isfinite(tp)
        fp, tp = fp[valid], tp[valid]
        if len(fp) < 2:
            return 1.0, 0.5
        loss = str(loss).lower().strip()

        def objective(params: np.ndarray) -> float:
            E = np.exp(params[0])
            R = 1.0 / (1.0 + np.exp(-params[1]))
            error = EnglandEROC.tp(fp, E, R) - tp
            if loss in {"rmse", "mse"}:
                return float(np.mean(error**2))
            if loss == "mae":
                return float(np.mean(np.abs(error)))
            return float(np.max(np.abs(error)))

        result = minimize(
            objective,
            np.asarray([np.log(2.0), 0.0]),
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8},
        )
        E = float(np.exp(result.x[0]))
        R = float(1.0 / (1.0 + np.exp(-result.x[1])))
        return E, R
