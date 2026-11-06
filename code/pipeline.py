# file: pipeline.py

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import queue as queue_module
import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from datetime import datetime
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import nibabel as nib
import numpy as np
import yaml

from config import AppConfig
from contrasts import ContrastCalculator
from data_loader import ProjectDataLoader, sorted_common_run_names
from data_types import BrainMask, EBResult, PatientData, ReproducibilityMap, Voxel
from design import DesignMatrixBuilder, residualize_nuisance_orthonormal
from empirical_bayes import EmpiricalBayesGLM
from kappa import choose_threshold_by_kappa, kappa_curve
from ordinary_least_squares import OrdinaryLeastSquaresGLM
from prewhitening import estimate_ar1_rho_from_residuals, estimate_ar1_rho_volume_for_run, prewhiten_ar1
from reproducibility import compute_reproducibility_map
from roc import EnglandEROC, MixtureMultinomialROC
from roc_plotting import (
    methods_from_plot_mode,
    normalize_method_name,
    save_roc_summary_plot,
    save_selected_lambda_direction_plot,
)
from tvalue_plots import save_tvalue_plots
from spm_preprocessing import SPMPreprocessingManager

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None


logger = logging.getLogger(__name__)


def _glm_method_selection_mode(value: object) -> str:
    """
    Return the GLM selection mode for maps.

    fixed:
        Implementation detail.

    choose_best_kappa:
        compute Bayes and OLS and select the method+lambda mode by maximum
        Kappa value on the ROC curve.
    """
    text = str(value).lower().strip().replace("-", "_").replace(" ", "_")

    aliases = {
        "choose_best",
        "choose_best_kappa",
        "best",
        "best_kappa",
        "kappa_best",
        "auto",
        "auto_kappa",
    }

    if text in aliases:
        return "choose_best_kappa"

    return "fixed"


def _is_choose_best_kappa(value: object) -> bool:
    return _glm_method_selection_mode(value) == "choose_best_kappa"


def _worker_fit_glm(
    method: str,
    eb_model: EmpiricalBayesGLM,
    ols_model: OrdinaryLeastSquaresGLM,
    X_list: List[np.ndarray],
    y_list: List[np.ndarray],
    X_star: np.ndarray,
) -> EBResult:
    method = normalize_method_name(method)

    if method == "bayes":
        bayes_result = eb_model.fit(X_list, y_list, X_star)
        bayes_result.bayes_converged = bool(bayes_result.converged)
        bayes_result.used_ols_fallback = False
        if (
            not bayes_result.converged
            and bool(getattr(eb_model.config, "fallback_to_ols", True))
        ):
            ols_result = ols_model.fit(X_list, y_list, X_star)
            ols_result.bayes_converged = False
            ols_result.used_ols_fallback = True
            fallback_diagnostics = dict(ols_result.diagnostics or {})
            fallback_diagnostics.update(
                {
                    "fallback_reason": "bayes_not_converged",
                    "bayes_n_iter": int(bayes_result.n_iter),
                    "bayes_start_log_likelihood": float(
                        getattr(bayes_result, "start_log_likelihood", np.nan)
                    ),
                    "bayes_log_likelihood": float(
                        getattr(bayes_result, "log_likelihood", np.nan)
                    ),
                    "bayes_diagnostics": getattr(bayes_result, "diagnostics", None),
                }
            )
            ols_result.diagnostics = fallback_diagnostics
            return ols_result
        return bayes_result

    if method == "ols":
        result = ols_model.fit(X_list, y_list, X_star)
        result.bayes_converged = None
        result.used_ols_fallback = False
        return result

    raise ValueError(f"Unknown GLM method: {method}")


def _worker_prepare_voxel_inputs(
    config: AppConfig,
    design_builder: DesignMatrixBuilder,
    run_arrays: Dict[str, np.ndarray],
    logs_by_run: Dict[str, object],
    run_names: List[str],
    voxel: Voxel,
    rho_volumes: Optional[Dict[str, np.ndarray]] = None,
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for run_name in run_names:
        data = run_arrays[run_name]
        design = design_builder.build_matrices_for_run_and_slice(
            run_name=run_name,
            log_df=logs_by_run.get(run_name),
            z=voxel.z,
        )
        y = data[:, voxel.x, voxel.y, voxel.z].astype(float)
        full = design.full

        if config.preprocessing.prewhitening:
            if config.preprocessing.estimate_rho_from_residuals:
                if rho_volumes is not None and run_name in rho_volumes:
                    rho = float(rho_volumes[run_name][voxel.x, voxel.y, voxel.z])
                else:
                    rho = estimate_ar1_rho_from_residuals(
                        y=y,
                        X=full,
                        rho_clip=config.preprocessing.rho_clip,
                        bias_correction=bool(getattr(config.preprocessing, "rho_bias_correction", True)),
                    )
            else:
                rho = config.preprocessing.fixed_rho
            y, full = prewhiten_ar1(y, full, rho)

        n_stim = design.n_stimulus
        x_stim = full[:, :n_stim]
        nuisance = full[:, n_stim:]
        if nuisance.shape[1]:
            if not bool(getattr(config.design, "residualize_nuisance", True)):
                raise ValueError(
                    "The hierarchical Liou model requires nuisance residualization "
                    "so that Omega contains stimulus effects only."
                )
            y, x_stim, _ = residualize_nuisance_orthonormal(
                y, x_stim, nuisance,
                rank_tolerance=float(getattr(config.design, "nuisance_rank_tolerance", 1e-10)),
            )
        X_list.append(x_stim)
        y_list.append(y)

    return X_list, y_list


def _open_memmap_specs(specs: Dict[str, Dict[str, object]]) -> Dict[str, np.ndarray]:
    """
    Open read-only NumPy memmaps described by _build_voxel_series_memmaps().

    The arrays are shared by the OS page cache across worker processes.  Only the
    small time series rows actually requested by a worker are paged into memory.
    """
    arrays: Dict[str, np.ndarray] = {}

    for name, spec in specs.items():
        path = str(spec["path"])
        dtype = np.dtype(str(spec["dtype"]))
        shape = tuple(int(v) for v in spec["shape"])
        arrays[str(name)] = np.memmap(path, dtype=dtype, mode="r", shape=shape)

    return arrays


def _worker_prepare_voxel_inputs_from_series(
    config: AppConfig,
    design_builder: DesignMatrixBuilder,
    voxel_series_by_run: Dict[str, np.ndarray],
    logs_by_run: Dict[str, object],
    run_names: List[str],
    voxel: Voxel,
    voxel_index: int,
    rho_vectors_by_run: Optional[Dict[str, np.ndarray]] = None,
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for run_name in run_names:
        design = design_builder.build_matrices_for_run_and_slice(
            run_name=run_name,
            log_df=logs_by_run.get(run_name),
            z=voxel.z,
        )
        y = np.asarray(voxel_series_by_run[run_name][int(voxel_index), :], dtype=float)
        full = design.full

        if config.preprocessing.prewhitening:
            if config.preprocessing.estimate_rho_from_residuals:
                if rho_vectors_by_run is not None and run_name in rho_vectors_by_run:
                    rho = float(rho_vectors_by_run[run_name][int(voxel_index)])
                else:
                    rho = estimate_ar1_rho_from_residuals(
                        y=y, X=full, rho_clip=config.preprocessing.rho_clip,
                        bias_correction=bool(getattr(config.preprocessing, "rho_bias_correction", True)),
                    )
            else:
                rho = config.preprocessing.fixed_rho
            y, full = prewhiten_ar1(y, full, rho)

        n_stim = design.n_stimulus
        x_stim = full[:, :n_stim]
        nuisance = full[:, n_stim:]
        if nuisance.shape[1]:
            if not bool(getattr(config.design, "residualize_nuisance", True)):
                raise ValueError(
                    "The hierarchical Liou model requires nuisance residualization "
                    "so that Omega contains stimulus effects only."
                )
            y, x_stim, _ = residualize_nuisance_orthonormal(
                y, x_stim, nuisance,
                rank_tolerance=float(getattr(config.design, "nuisance_rank_tolerance", 1e-10)),
            )
        X_list.append(x_stim)
        y_list.append(y)

    return X_list, y_list


def _multiprocessing_voxel_worker(args: Dict[str, object]) -> Dict[str, object]:
    """
    Each process receives a preallocated subset of voxels.

    The split is performed in the main process using:
        worker_id = voxel_global_index % n_jobs
    """
    worker_id = int(args["worker_id"])
    indices = np.asarray(args["indices"], dtype=int)
    coords = np.asarray(args["coords"], dtype=int)

    config: AppConfig = args["config"]

    # Two input modes are supported:
    # 1) legacy in-memory arrays ("run_arrays") for sequential/thread/copy-process mode;
    # 2) process-safe masked memmaps ("voxel_series_specs") for full-brain multiprocessing.
    run_arrays: Optional[Dict[str, np.ndarray]] = args.get("run_arrays", None)
    voxel_series_by_run: Optional[Dict[str, np.ndarray]] = None
    rho_vectors_by_run: Optional[Dict[str, np.ndarray]] = None

    if "voxel_series_specs" in args:
        voxel_series_by_run = _open_memmap_specs(args["voxel_series_specs"])
        rho_specs = args.get("rho_vector_specs", None)
        if isinstance(rho_specs, dict) and rho_specs:
            rho_vectors_by_run = _open_memmap_specs(rho_specs)

    logs_by_run: Dict[str, object] = args["logs_by_run"]
    run_names: List[str] = args["run_names"]
    X_star: np.ndarray = args["X_star"]
    methods_to_run: List[str] = args["methods_to_run"]
    first_level_contrasts: Dict[str, np.ndarray] = args["first_level_contrasts"]
    second_level_contrasts: Dict[str, np.ndarray] = args["second_level_contrasts"]
    images_order: List[str] = args["images_order"]
    rho_volumes: Optional[Dict[str, np.ndarray]] = args.get("rho_volumes", None)

    progress_queue = args.get("progress_queue", None)
    progress_step = int(args.get("progress_step", 100))

    # Inside each process, BLAS/OpenMP should not inflate the thread count.
    # Parallelism is already handled at the process level.
    blas_threads = int(getattr(config.analysis, "blas_threads", 1))
    blas_threads = max(1, blas_threads)

    n_local = len(indices)
    n_runs = len(run_names)

    design_builder = DesignMatrixBuilder(config)
    design_builder.images_order = list(images_order)

    eb_model = EmpiricalBayesGLM(config.empirical_bayes)
    ols_model = OrdinaryLeastSquaresGLM(
        min_sigma2=config.empirical_bayes.min_sigma2,
        omega_ridge=config.empirical_bayes.omega_ridge,
    )
    contrast_calculator = ContrastCalculator(config)

    convergence = {
        method: np.zeros(n_local, dtype=bool)
        for method in methods_to_run
    }
    glm_diagnostics = {
        method: {
            "n_iter": np.zeros(n_local, dtype=np.int32),
            "used_ols_fallback": np.zeros(n_local, dtype=bool),
            "omega_condition_number": np.full(n_local, np.nan, dtype=float),
            "omega_min_eigenvalue": np.full(n_local, np.nan, dtype=float),
        }
        for method in methods_to_run
    }

    t_values_store = {
        method: {
            c_name: np.full((n_runs, n_local), np.nan, dtype=float)
            for c_name in first_level_contrasts
        }
        for method in methods_to_run
    }

    T_values_store = {
        method: {
            c_name: {
                d_name: np.full(n_local, np.nan, dtype=float)
                for d_name in second_level_contrasts
            }
            for c_name in first_level_contrasts
        }
        for method in methods_to_run
    }

    n_errors = 0

    def run_worker_body() -> None:
        nonlocal n_errors

        for local_pos, coord in enumerate(coords):
            voxel = Voxel(int(coord[0]), int(coord[1]), int(coord[2]))

            try:
                if voxel_series_by_run is not None:
                    X_list, y_list = _worker_prepare_voxel_inputs_from_series(
                        config=config,
                        design_builder=design_builder,
                        voxel_series_by_run=voxel_series_by_run,
                        logs_by_run=logs_by_run,
                        run_names=run_names,
                        voxel=voxel,
                        voxel_index=int(indices[local_pos]),
                        rho_vectors_by_run=rho_vectors_by_run,
                    )
                else:
                    if run_arrays is None:
                        raise RuntimeError("Worker received neither run_arrays nor voxel_series_specs")
                    X_list, y_list = _worker_prepare_voxel_inputs(
                        config=config,
                        design_builder=design_builder,
                        run_arrays=run_arrays,
                        logs_by_run=logs_by_run,
                        run_names=run_names,
                        voxel=voxel,
                        rho_volumes=rho_volumes,
                    )

                for method in methods_to_run:
                    glm_result = _worker_fit_glm(
                        method=method,
                        eb_model=eb_model,
                        ols_model=ols_model,
                        X_list=X_list,
                        y_list=y_list,
                        X_star=X_star,
                    )

                    convergence[method][local_pos] = bool(
                        getattr(glm_result, "bayes_converged", glm_result.converged)
                    )
                    glm_diagnostics[method]["n_iter"][local_pos] = int(glm_result.n_iter)
                    glm_diagnostics[method]["used_ols_fallback"][local_pos] = bool(
                        getattr(glm_result, "used_ols_fallback", False)
                    )
                    try:
                        eigvals = np.linalg.eigvalsh(np.asarray(glm_result.Omega, dtype=float))
                        glm_diagnostics[method]["omega_min_eigenvalue"][local_pos] = float(np.min(eigvals))
                        glm_diagnostics[method]["omega_condition_number"][local_pos] = float(
                            np.linalg.cond(np.asarray(glm_result.Omega, dtype=float))
                        )
                    except Exception:
                        pass

                    for c_name, c in first_level_contrasts.items():
                        t_values = contrast_calculator.compute_runwise_t(glm_result, c)
                        t_values_store[method][c_name][:, local_pos] = t_values

                        for d_name, d in second_level_contrasts.items():
                            T_value = contrast_calculator.compute_second_level_T(
                                glm_result,
                                X_star,
                                c,
                                d,
                            )
                            T_values_store[method][c_name][d_name][local_pos] = T_value

            except Exception:
                n_errors += 1

            done = local_pos + 1
            if progress_queue is not None and (done % progress_step == 0 or done == n_local):
                progress_queue.put(("progress", worker_id, done, n_local, n_errors))

    context = threadpool_limits(limits=blas_threads) if threadpool_limits is not None else nullcontext()

    with context:
        run_worker_body()

    if progress_queue is not None:
        progress_queue.put(("done", worker_id, n_local, n_errors))

    return {
        "worker_id": worker_id,
        "indices": indices,
        "convergence": convergence,
        "glm_diagnostics": glm_diagnostics,
        "t_values": t_values_store,
        "T_values": T_values_store,
        "n_errors": n_errors,
    }


class LiouReproducibilityPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.loader = ProjectDataLoader(config)
        self.design_builder = DesignMatrixBuilder(config)
        self.eb_model = EmpiricalBayesGLM(config.empirical_bayes)
        self.ols_model = OrdinaryLeastSquaresGLM(
            min_sigma2=config.empirical_bayes.min_sigma2,
            omega_ridge=config.empirical_bayes.omega_ridge,
            omega_shrinkage_alpha=config.empirical_bayes.omega_shrinkage_alpha,
            omega_shrinkage_target=config.empirical_bayes.omega_shrinkage_target,
            omega_eigenvalue_floor_relative=(
                config.empirical_bayes.omega_eigenvalue_floor_relative
            ),
        )
        self.contrast_calculator = ContrastCalculator(config)
        self.roc_model = MixtureMultinomialROC(config.analysis)
        self.spm_preprocessing = SPMPreprocessingManager(config)
        self._patient_output_dirs: Dict[str, Path] = {}
        self._run_metadata_by_patient: Dict[str, Dict[str, object]] = {}
        self._design_diagnostics: Dict[str, object] = {}
        self._glm_diagnostics_store: Dict[str, Dict[str, np.ndarray]] = {}

    def run(self) -> Dict[str, object]:
        self.spm_preprocessing.ensure_preprocessed()

        patient = self.loader.load_patient()
        mask = self.loader.load_or_compute_mask(patient)

        self._prepare_patient_output_dir(patient.patient_id)

        if self.config.analysis.save_nifti:
            self._save_brain_mask_nifti(mask, patient.patient_id)
            self._save_mean_functional_nifti(patient)

        run_names = sorted_common_run_names(patient, expected_n_runs=self.config.scanner.nR)
        logger.info("Runs used: %s", run_names)

        if self.config.design.design_type == "event_regressors":
            self.design_builder.set_images_order_from_logs(patient.logs, run_names)
            logger.info("Global stimulus order length: %d", len(self.design_builder.images_order))
        else:
            logger.info(
                "Condition-level design columns: %s",
                list(getattr(self.config.data, "condition_order", [])),
            )

        X_star = self._build_x_star(len(run_names))

        coords = mask.coords
        if self.config.analysis.max_voxels is not None:
            coords = coords[: self.config.analysis.max_voxels]

        logger.info("Voxels selected for analysis: %d", len(coords))

        first_design = self.design_builder.build_matrices_for_run_and_slice(
            run_name=run_names[0],
            log_df=patient.logs.get(run_names[0]),
            z=0,
        )
        k_stimulus = int(first_design.n_stimulus)
        k_nuisance = int(first_design.n_nuisance)
        k_full = int(first_design.full.shape[1])

        logger.info(
            "First-level design dimensions: stimulus=%d, nuisance=%d, full=%d; Omega=%d x %d",
            k_stimulus, k_nuisance, k_full, k_stimulus, k_stimulus,
        )
        logger.info("Stimulus regressors: %s", first_design.stimulus_names)
        logger.info("Nuisance regressors: %s", first_design.nuisance_names)
        self._design_diagnostics = {
            "stimulus_dimension": k_stimulus,
            "nuisance_dimension": k_nuisance,
            "full_dimension": k_full,
            "omega_dimension": [k_stimulus, k_stimulus],
            "stimulus_names": list(first_design.stimulus_names),
            "nuisance_names": list(first_design.nuisance_names),
            "motion_regressors_enabled": bool(getattr(self.config.design, "add_motion_regressors", False)),
            "motion_summary_by_run": dict(getattr(self.config.design, "motion_summary_by_run", {}) or {}),
        }

        configured_glm_method = str(getattr(self.config.analysis, "glm_method", "bayes")).lower().strip()
        selection_mode = _glm_method_selection_mode(configured_glm_method)

        if selection_mode == "choose_best_kappa":
            # To select the best model by Kappa, both GLM methods must be computed.
            fixed_main_method = "bayes"
            roc_methods = ["bayes", "ols"]
            methods_to_run = ["bayes", "ols"]
            logger.info(
                "Main GLM method for maps: choose_best_kappa "
                "(Bayes/OLS and lambda free/fixed will be selected by max Kappa)"
            )
        else:
            fixed_main_method = normalize_method_name(configured_glm_method)
            roc_methods = methods_from_plot_mode(self.config.analysis.roc_plot_methods, fixed_main_method)
            methods_to_run = sorted(set([fixed_main_method] + roc_methods))
            logger.info("Main GLM method for maps: %s", fixed_main_method)

        logger.info("GLM selection mode: %s", selection_mode)
        logger.info("GLM methods ROC visualization methods: %s", roc_methods)
        logger.info("GLM methods that will actually be computed: %s", methods_to_run)

        n_jobs = self._resolve_n_jobs(len(coords))
        logger.info("Voxel multiprocessing: n_jobs=%d", n_jobs)

        voxel_backend = self._resolve_voxel_parallel_backend(n_jobs)

        if n_jobs > 1:
            logger.info("Voxel parallel backend: %s", voxel_backend)
            if voxel_backend == "thread":
                logger.info(
                    "Thread backend shares loaded BOLD arrays, but Python voxel loops "
                    "can be limited by the GIL; use process_memmap for full CPU use."
                )
            elif voxel_backend == "process_memmap":
                logger.info(
                    "Memory-optimized multiprocessing: workers read masked voxel "
                    "time-series from shared NumPy memmaps instead of pickled BOLD arrays."
                )
            else:
                logger.warning(
                    "process_copy backend pickles BOLD arrays into workers and is not "
                    "recommended for full-brain Windows runs."
                )
        else:
            logger.info("n_jobs=1: computation will be sequential.")

        logger.info(
            "GPU acceleration is not enabled. "
            "This version uses CPU parallelism across voxels."
        )

        rho_volumes = self._estimate_prewhitening_rho_volumes(
            patient=patient,
            mask=mask,
            run_names=run_names,
        )

        if self.config.analysis.first_level_contrasts:
            first_level_contrasts = self.contrast_calculator.first_level_contrasts(k_stimulus)
        else:
            first_level_contrasts = self.contrast_calculator.build_me_vs_fr_contrast(
                k_design=k_stimulus,
                images_order=self.design_builder.images_order,
                logs_by_run=patient.logs,
                run_names=run_names,
            )

        second_level_contrasts = self.contrast_calculator.second_level_contrasts(X_star.shape[1])

        first_level_contrast_vectors = {
            name: np.asarray(vector, dtype=float).tolist()
            for name, vector in first_level_contrasts.items()
        }
        second_level_contrast_vectors = {
            name: np.asarray(vector, dtype=float).tolist()
            for name, vector in second_level_contrasts.items()
        }

        t_store: Dict[str, Dict[str, np.ndarray]] = {
            method: {
                c_name: np.full((len(run_names), len(coords)), np.nan, dtype=float)
                for c_name in first_level_contrasts
            }
            for method in methods_to_run
        }

        T_store: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {
            method: {
                c_name: {
                    d_name: np.full(len(coords), np.nan, dtype=float)
                    for d_name in second_level_contrasts
                }
                for c_name in first_level_contrasts
            }
            for method in methods_to_run
        }

        convergence_store: Dict[str, np.ndarray] = {
            method: np.zeros(len(coords), dtype=bool)
            for method in methods_to_run
        }
        self._glm_diagnostics_store = {
            method: {
                "n_iter": np.zeros(len(coords), dtype=np.int32),
                "used_ols_fallback": np.zeros(len(coords), dtype=bool),
                "omega_condition_number": np.full(len(coords), np.nan, dtype=float),
                "omega_min_eigenvalue": np.full(len(coords), np.nan, dtype=float),
            }
            for method in methods_to_run
        }

        if n_jobs <= 1:
            self._process_voxels_sequential(
                patient=patient,
                coords=coords,
                run_names=run_names,
                X_star=X_star,
                rho_volumes=rho_volumes,
                methods_to_run=methods_to_run,
                first_level_contrasts=first_level_contrasts,
                second_level_contrasts=second_level_contrasts,
                t_store=t_store,
                T_store=T_store,
                convergence_store=convergence_store,
            )
        else:
            if voxel_backend == "thread":
                self._process_voxels_threaded(
                    patient=patient,
                    coords=coords,
                    run_names=run_names,
                    X_star=X_star,
                    rho_volumes=rho_volumes,
                    methods_to_run=methods_to_run,
                    first_level_contrasts=first_level_contrasts,
                    second_level_contrasts=second_level_contrasts,
                    t_store=t_store,
                    T_store=T_store,
                    convergence_store=convergence_store,
                    n_jobs=n_jobs,
                )
            elif voxel_backend == "process_memmap":
                self._process_voxels_multiprocessing_memmap(
                    patient=patient,
                    coords=coords,
                    run_names=run_names,
                    X_star=X_star,
                    rho_volumes=rho_volumes,
                    methods_to_run=methods_to_run,
                    first_level_contrasts=first_level_contrasts,
                    second_level_contrasts=second_level_contrasts,
                    t_store=t_store,
                    T_store=T_store,
                    convergence_store=convergence_store,
                    n_jobs=n_jobs,
                )
            else:
                self._process_voxels_multiprocessing(
                    patient=patient,
                    coords=coords,
                    run_names=run_names,
                    X_star=X_star,
                    rho_volumes=rho_volumes,
                    methods_to_run=methods_to_run,
                    first_level_contrasts=first_level_contrasts,
                    second_level_contrasts=second_level_contrasts,
                    t_store=t_store,
                    T_store=T_store,
                    convergence_store=convergence_store,
                    n_jobs=n_jobs,
                )

        outputs = self._postprocess_and_save(
            patient=patient,
            mask=mask,
            coords=coords,
            run_names=run_names,
            fixed_main_method=fixed_main_method,
            configured_glm_method=configured_glm_method,
            selection_mode=selection_mode,
            roc_methods=roc_methods,
            t_store=t_store,
            T_store=T_store,
            convergence_store=convergence_store,
            first_level_contrast_vectors=first_level_contrast_vectors,
            second_level_contrast_vectors=second_level_contrast_vectors,
        )

        return outputs

    def _estimate_prewhitening_rho_volumes(
        self,
        patient: PatientData,
        mask: BrainMask,
        run_names: List[str],
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Estimate Worsley-style AR(1) rho maps for all runs before voxel GLM fitting.

        This enables spatial regularization of rho. The design can vary by slice,
        so design matrices are cached by z for each run.
        """
        if not self.config.preprocessing.prewhitening:
            return None

        if not self.config.preprocessing.estimate_rho_from_residuals:
            return None

        bias_correction = bool(getattr(self.config.preprocessing, "rho_bias_correction", True))
        smoothing_fwhm = getattr(self.config.preprocessing, "rho_spatial_smoothing_fwhm", None)
        rho_clip = float(getattr(self.config.preprocessing, "rho_clip", 0.99))
        voxel_sizes = tuple(float(v) for v in np.sqrt(np.sum(mask.affine[:3, :3] ** 2, axis=0)))

        logger.info(
            "Estimating AR(1) rho volumes: bias_correction=%s, smoothing_fwhm=%s mm, voxel_sizes=%s, rho_clip=%.3f",
            bias_correction,
            smoothing_fwhm,
            tuple(round(v, 4) for v in voxel_sizes),
            rho_clip,
        )

        rho_volumes: Dict[str, np.ndarray] = {}
        nz = int(mask.mask.shape[2])

        for run_name in run_names:
            run = patient.runs[run_name]
            log_df = patient.logs.get(run_name)

            design_by_z = {
                z: self.design_builder.build_for_run_and_slice(
                    run_name=run_name,
                    log_df=log_df,
                    z=z,
                )
                for z in range(nz)
            }

            rho_volume = estimate_ar1_rho_volume_for_run(
                data=run.data,
                mask=mask.mask,
                design_by_z=design_by_z,
                rho_clip=rho_clip,
                bias_correction=bias_correction,
                smoothing_fwhm=smoothing_fwhm,
                voxel_sizes=voxel_sizes,
            )

            rho_volumes[run_name] = rho_volume.astype(np.float32)

            inside = rho_volume[mask.mask]
            logger.info(
                "Run %s AR(1) rho: mean=%.4f, median=%.4f, min=%.4f, max=%.4f",
                run_name,
                float(np.nanmean(inside)) if inside.size else 0.0,
                float(np.nanmedian(inside)) if inside.size else 0.0,
                float(np.nanmin(inside)) if inside.size else 0.0,
                float(np.nanmax(inside)) if inside.size else 0.0,
            )

        return rho_volumes

    def _resolve_n_jobs(self, n_voxels: int) -> int:
        raw = getattr(self.config.analysis, "n_jobs", 1)

        try:
            n_jobs = int(raw)
        except Exception:
            n_jobs = 1

        cpu_count = os.cpu_count() or 1

        if n_jobs == -1:
            n_jobs = cpu_count

        if n_jobs <= 0:
            n_jobs = 1

        n_jobs = min(n_jobs, cpu_count)

        if n_voxels > 0:
            n_jobs = min(n_jobs, n_voxels)

        return max(1, n_jobs)

    def _resolve_voxel_parallel_backend(self, n_jobs: int) -> str:
        """
        Resolve the backend used for voxel-level parallelism.

        On Windows, ProcessPoolExecutor uses spawn. Passing large NumPy arrays
        directly to workers pickles/copies them and can exhaust RAM.  The
        process_memmap backend keeps real multiprocessing while sharing only
        on-disk NumPy memmaps between workers.
        """
        if n_jobs <= 1:
            return "sequential"

        raw = str(getattr(self.config.analysis, "voxel_parallel_backend", "process_memmap")).lower().strip()
        aliases = {
            "threads": "thread",
            "threading": "thread",
            "threadpool": "thread",
            "multiprocessing": "process_memmap",
            "processes": "process_memmap",
            "processpool": "process_memmap",
            "mp": "process_memmap",
            "process": "process_memmap",
            "memmap": "process_memmap",
            "process-memmap": "process_memmap",
            "process_memmapped": "process_memmap",
            "copy_process": "process_copy",
            "process_copy": "process_copy",
        }
        backend = aliases.get(raw, raw)

        if backend not in {"thread", "process_memmap", "process_copy"}:
            raise ValueError(
                "Unknown analysis.voxel_parallel_backend. "
                "Allowed values: 'process_memmap', 'thread', 'process_copy'. "
                f"Got: {raw}"
            )

        return backend

    @staticmethod
    def _split_indices_round_robin(n_voxels: int, n_jobs: int) -> List[np.ndarray]:
        """
        Round-robin distribution:

            worker 0: 0, n_jobs, 2*n_jobs, ...
            worker 1: 1, n_jobs+1, ...
            ...
        """
        return [
            np.arange(worker_id, n_voxels, n_jobs, dtype=int)
            for worker_id in range(n_jobs)
        ]

    @staticmethod
    def _split_indices_contiguous(n_voxels: int, n_jobs: int) -> List[np.ndarray]:
        """
        Contiguous chunks are better for memmapped voxel-series matrices:
        each worker reads long row ranges instead of jumping through the file.
        """
        if n_voxels <= 0:
            return []

        n_jobs = max(1, min(int(n_jobs), int(n_voxels)))
        return [
            arr.astype(int, copy=False)
            for arr in np.array_split(np.arange(n_voxels, dtype=int), n_jobs)
            if len(arr) > 0
        ]

    def _process_voxels_sequential(
        self,
        patient: PatientData,
        coords: np.ndarray,
        run_names: List[str],
        X_star: np.ndarray,
        rho_volumes: Optional[Dict[str, np.ndarray]],
        methods_to_run: List[str],
        first_level_contrasts: Dict[str, np.ndarray],
        second_level_contrasts: Dict[str, np.ndarray],
        t_store: Dict[str, Dict[str, np.ndarray]],
        T_store: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        convergence_store: Dict[str, np.ndarray],
    ) -> None:
        for idx, coord in enumerate(coords):
            voxel = Voxel(int(coord[0]), int(coord[1]), int(coord[2]))

            try:
                X_list, y_list = self._prepare_voxel_inputs(
                    patient=patient,
                    run_names=run_names,
                    voxel=voxel,
                    rho_volumes=rho_volumes,
                )

                for method in methods_to_run:
                    glm_result = self._fit_glm(method, X_list, y_list, X_star)
                    convergence_store[method][idx] = bool(
                        getattr(glm_result, "bayes_converged", glm_result.converged)
                    )
                    diag_store = self._glm_diagnostics_store[method]
                    diag_store["n_iter"][idx] = int(glm_result.n_iter)
                    diag_store["used_ols_fallback"][idx] = bool(
                        getattr(glm_result, "used_ols_fallback", False)
                    )
                    try:
                        eigvals = np.linalg.eigvalsh(np.asarray(glm_result.Omega, dtype=float))
                        diag_store["omega_min_eigenvalue"][idx] = float(np.min(eigvals))
                        diag_store["omega_condition_number"][idx] = float(
                            np.linalg.cond(np.asarray(glm_result.Omega, dtype=float))
                        )
                    except Exception:
                        pass

                    for c_name, c in first_level_contrasts.items():
                        t_values = self.contrast_calculator.compute_runwise_t(glm_result, c)
                        t_store[method][c_name][:, idx] = t_values

                        for d_name, d in second_level_contrasts.items():
                            T_value = self.contrast_calculator.compute_second_level_T(
                                glm_result,
                                X_star,
                                c,
                                d,
                            )
                            T_store[method][c_name][d_name][idx] = T_value

            except Exception as exc:
                logger.debug("Error in voxel %s: %s", voxel, exc)

            if (idx + 1) % 500 == 0 or (idx + 1) == len(coords):
                logger.info("Processed voxels: %d / %d", idx + 1, len(coords))

    def _process_voxels_threaded(
        self,
        patient: PatientData,
        coords: np.ndarray,
        run_names: List[str],
        X_star: np.ndarray,
        rho_volumes: Optional[Dict[str, np.ndarray]],
        methods_to_run: List[str],
        first_level_contrasts: Dict[str, np.ndarray],
        second_level_contrasts: Dict[str, np.ndarray],
        t_store: Dict[str, Dict[str, np.ndarray]],
        T_store: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        convergence_store: Dict[str, np.ndarray],
        n_jobs: int,
    ) -> None:
        """
        Memory-efficient parallel voxel processing.

        Unlike the process backend, this backend does not pickle/copy the large
        T x X x Y x Z BOLD arrays into every worker. Threads share the already
        loaded arrays inside one Python process, which is the safer default for
        full-brain fMRI runs on Windows.
        """
        n_voxels = len(coords)

        worker_indices = self._split_indices_round_robin(
            n_voxels=n_voxels,
            n_jobs=n_jobs,
        )
        worker_indices = [idxs for idxs in worker_indices if len(idxs) > 0]
        n_workers = len(worker_indices)

        logger.info(
            "Created threaded chunks: %d. Chunk sizes: %s",
            n_workers,
            [int(len(idxs)) for idxs in worker_indices],
        )

        run_arrays = {
            run_name: patient.runs[run_name].data
            for run_name in run_names
        }

        logs_by_run = {
            run_name: patient.logs.get(run_name)
            for run_name in run_names
        }

        progress_queue: queue_module.Queue = queue_module.Queue()
        progress_by_worker: Dict[int, int] = {
            worker_id: 0
            for worker_id in range(n_workers)
        }

        next_report = 500
        progress_step = 100

        futures = []

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for worker_id, indices in enumerate(worker_indices):
                args = {
                    "worker_id": worker_id,
                    "indices": indices,
                    "coords": coords[indices],
                    "config": self.config,
                    "run_arrays": run_arrays,
                    "logs_by_run": logs_by_run,
                    "run_names": run_names,
                    "X_star": X_star,
                    "rho_volumes": rho_volumes,
                    "methods_to_run": methods_to_run,
                    "first_level_contrasts": first_level_contrasts,
                    "second_level_contrasts": second_level_contrasts,
                    "images_order": list(self.design_builder.images_order),
                    "progress_queue": progress_queue,
                    "progress_step": progress_step,
                }

                futures.append(executor.submit(_multiprocessing_voxel_worker, args))

            pending = set(futures)

            while pending:
                next_report = self._drain_progress_queue(
                    progress_queue=progress_queue,
                    progress_by_worker=progress_by_worker,
                    total=n_voxels,
                    next_report=next_report,
                )

                done, pending = wait(
                    pending,
                    timeout=0.5,
                    return_when=FIRST_COMPLETED,
                )

                for future in done:
                    result = future.result()
                    worker_id = int(result["worker_id"])
                    n_local = int(len(result["indices"]))

                    self._store_worker_result(
                        result=result,
                        t_store=t_store,
                        T_store=T_store,
                        convergence_store=convergence_store,
                    )

                    progress_by_worker[worker_id] = n_local

                    logger.info(
                        "Thread worker %d finished its chunk: %d voxels, errors: %d",
                        worker_id + 1,
                        n_local,
                        int(result.get("n_errors", 0)),
                    )

                next_report = self._drain_progress_queue(
                    progress_queue=progress_queue,
                    progress_by_worker=progress_by_worker,
                    total=n_voxels,
                    next_report=next_report,
                )

        total_done = int(sum(progress_by_worker.values()))
        if total_done < n_voxels:
            logger.warning(
                "Threaded processing finished, but progress shows %d / %d",
                total_done,
                n_voxels,
            )
        else:
            logger.info("Processed voxels: %d / %d", n_voxels, n_voxels)


    def _process_voxels_multiprocessing_memmap(
        self,
        patient: PatientData,
        coords: np.ndarray,
        run_names: List[str],
        X_star: np.ndarray,
        rho_volumes: Optional[Dict[str, np.ndarray]],
        methods_to_run: List[str],
        first_level_contrasts: Dict[str, np.ndarray],
        second_level_contrasts: Dict[str, np.ndarray],
        t_store: Dict[str, Dict[str, np.ndarray]],
        T_store: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        convergence_store: Dict[str, np.ndarray],
        n_jobs: int,
    ) -> None:
        """
        CPU-parallel and memory-safe voxel processing for Windows.

        The main process writes one masked time-series memmap per run:
            shape = (n_selected_voxels, n_timepoints), dtype=float32

        Worker processes receive only filenames and small metadata.  This avoids
        the Windows spawn copy of 4D BOLD arrays while still using separate
        Python processes, so the voxel loop is not limited by the GIL.
        """
        n_voxels = int(len(coords))
        patient_out = self._patient_output_dir(patient.patient_id)

        worker_indices = self._split_indices_contiguous(
            n_voxels=n_voxels,
            n_jobs=n_jobs,
        )
        n_workers = len(worker_indices)

        logger.info(
            "Created process_memmap chunks: %d. Chunk sizes: %s",
            n_workers,
            [int(len(idxs)) for idxs in worker_indices],
        )

        logs_by_run = {
            run_name: patient.logs.get(run_name)
            for run_name in run_names
        }

        # Keep the temporary memmaps inside the current run folder so they are
        # easy to inspect during a running job.  They are removed automatically
        # after all workers finish.
        with tempfile.TemporaryDirectory(
            prefix="_voxel_memmap_",
            dir=str(patient_out),
        ) as tmp_dir:
            tmp_path = Path(tmp_dir)

            logger.info("Building masked voxel-series memmaps in %s", tmp_path)

            voxel_series_specs = self._build_voxel_series_memmaps(
                patient=patient,
                coords=coords,
                run_names=run_names,
                work_dir=tmp_path,
            )

            rho_vector_specs = self._build_rho_vector_memmaps(
                rho_volumes=rho_volumes,
                coords=coords,
                run_names=run_names,
                work_dir=tmp_path,
            )

            total_disk_mb = 0.0
            for specs in (voxel_series_specs, rho_vector_specs or {}):
                for spec in specs.values():
                    try:
                        total_disk_mb += Path(str(spec["path"])).stat().st_size / (1024.0 ** 2)
                    except OSError:
                        pass

            logger.info(
                "Memmap work files prepared: %.1f MB on disk. "
                "Workers will open them read-only.",
                total_disk_mb,
            )

            mp_context = mp.get_context("spawn")
            manager = mp_context.Manager()
            progress_queue = manager.Queue()

            progress_by_worker: Dict[int, int] = {
                worker_id: 0
                for worker_id in range(n_workers)
            }

            next_report = 500
            progress_step = 500

            executor = ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=mp_context,
            )

            try:
                futures = []

                for worker_id, indices in enumerate(worker_indices):
                    args = {
                        "worker_id": worker_id,
                        "indices": indices,
                        "coords": coords[indices],
                        "config": self.config,
                        "voxel_series_specs": voxel_series_specs,
                        "rho_vector_specs": rho_vector_specs,
                        "logs_by_run": logs_by_run,
                        "run_names": run_names,
                        "X_star": X_star,
                        "methods_to_run": methods_to_run,
                        "first_level_contrasts": first_level_contrasts,
                        "second_level_contrasts": second_level_contrasts,
                        "images_order": list(self.design_builder.images_order),
                        "progress_queue": progress_queue,
                        "progress_step": progress_step,
                    }

                    futures.append(executor.submit(_multiprocessing_voxel_worker, args))

                pending = set(futures)

                while pending:
                    next_report = self._drain_progress_queue(
                        progress_queue=progress_queue,
                        progress_by_worker=progress_by_worker,
                        total=n_voxels,
                        next_report=next_report,
                    )

                    done, pending = wait(
                        pending,
                        timeout=0.5,
                        return_when=FIRST_COMPLETED,
                    )

                    for future in done:
                        result = future.result()
                        worker_id = int(result["worker_id"])
                        n_local = int(len(result["indices"]))

                        self._store_worker_result(
                            result=result,
                            t_store=t_store,
                            T_store=T_store,
                            convergence_store=convergence_store,
                        )

                        progress_by_worker[worker_id] = n_local

                        logger.info(
                            "Memmap worker %d finished its chunk: %d voxels, errors: %d",
                            worker_id + 1,
                            n_local,
                            int(result.get("n_errors", 0)),
                        )

                    next_report = self._drain_progress_queue(
                        progress_queue=progress_queue,
                        progress_by_worker=progress_by_worker,
                        total=n_voxels,
                        next_report=next_report,
                    )

                total_done = int(sum(progress_by_worker.values()))
                if total_done < n_voxels:
                    logger.warning(
                        "process_memmap finished, but progress shows %d / %d",
                        total_done,
                        n_voxels,
                    )
                else:
                    logger.info("Processed voxels: %d / %d", n_voxels, n_voxels)

            except KeyboardInterrupt:
                logger.warning("Ctrl+C received. Stopping the process_memmap executor.")
                executor.shutdown(wait=False, cancel_futures=True)
                manager.shutdown()
                raise

            finally:
                executor.shutdown(wait=True, cancel_futures=False)
                manager.shutdown()

    def _build_voxel_series_memmaps(
        self,
        patient: PatientData,
        coords: np.ndarray,
        run_names: List[str],
        work_dir: Path,
    ) -> Dict[str, Dict[str, object]]:
        """
        Write masked time-series matrices used by process_memmap workers.

        For each run:
            memmap[row, :] = BOLD[:, x, y, z]
        where row corresponds to coords[row].
        """
        specs: Dict[str, Dict[str, object]] = {}
        coords = np.asarray(coords, dtype=np.int64)

        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords must have shape (n_voxels, 3); got {coords.shape}")

        n_voxels = int(coords.shape[0])
        if n_voxels == 0:
            return specs

        unique_z = np.unique(coords[:, 2])

        for run_name in run_names:
            data = np.asarray(patient.runs[run_name].data, dtype=np.float32)
            n_timepoints = int(data.shape[0])

            out_path = work_dir / f"voxel_series_{self._safe_filename(run_name)}.dat"
            mm = np.memmap(
                out_path,
                dtype=np.float32,
                mode="w+",
                shape=(n_voxels, n_timepoints),
            )

            for z_raw in unique_z:
                z = int(z_raw)
                row_idx = np.flatnonzero(coords[:, 2] == z)
                if row_idx.size == 0:
                    continue

                xs = coords[row_idx, 0]
                ys = coords[row_idx, 1]

                # data[:, xs, ys, z] gives (T, rows); transpose to rows x T.
                mm[row_idx, :] = np.asarray(data[:, xs, ys, z].T, dtype=np.float32)

            mm.flush()
            del mm

            specs[run_name] = {
                "path": str(out_path),
                "dtype": np.dtype(np.float32).str,
                "shape": (n_voxels, n_timepoints),
            }

            logger.info(
                "Voxel-series memmap written for %s: shape=(%d, %d), file=%s",
                run_name,
                n_voxels,
                n_timepoints,
                out_path,
            )

        return specs

    def _build_rho_vector_memmaps(
        self,
        rho_volumes: Optional[Dict[str, np.ndarray]],
        coords: np.ndarray,
        run_names: List[str],
        work_dir: Path,
    ) -> Dict[str, Dict[str, object]]:
        """
        Write one rho vector per run in the same row order as coords.
        """
        specs: Dict[str, Dict[str, object]] = {}

        if rho_volumes is None:
            return specs

        coords = np.asarray(coords, dtype=np.int64)
        n_voxels = int(coords.shape[0])

        if n_voxels == 0:
            return specs

        xs = coords[:, 0]
        ys = coords[:, 1]
        zs = coords[:, 2]

        for run_name in run_names:
            if run_name not in rho_volumes:
                continue

            rho_volume = np.asarray(rho_volumes[run_name], dtype=np.float32)

            out_path = work_dir / f"rho_{self._safe_filename(run_name)}.dat"
            mm = np.memmap(
                out_path,
                dtype=np.float32,
                mode="w+",
                shape=(n_voxels,),
            )
            mm[:] = np.asarray(rho_volume[xs, ys, zs], dtype=np.float32)
            mm.flush()
            del mm

            specs[run_name] = {
                "path": str(out_path),
                "dtype": np.dtype(np.float32).str,
                "shape": (n_voxels,),
            }

        if specs:
            logger.info("Rho memmap vectors written for %d runs", len(specs))

        return specs

    @staticmethod
    def _safe_filename(value: object) -> str:
        text = str(value)
        safe = []
        for ch in text:
            if ch.isalnum() or ch in {"-", "_", "."}:
                safe.append(ch)
            else:
                safe.append("_")
        return "".join(safe) or "item"


    def _process_voxels_multiprocessing(
        self,
        patient: PatientData,
        coords: np.ndarray,
        run_names: List[str],
        X_star: np.ndarray,
        rho_volumes: Optional[Dict[str, np.ndarray]],
        methods_to_run: List[str],
        first_level_contrasts: Dict[str, np.ndarray],
        second_level_contrasts: Dict[str, np.ndarray],
        t_store: Dict[str, Dict[str, np.ndarray]],
        T_store: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        convergence_store: Dict[str, np.ndarray],
        n_jobs: int,
    ) -> None:
        n_voxels = len(coords)

        worker_indices = self._split_indices_round_robin(
            n_voxels=n_voxels,
            n_jobs=n_jobs,
        )

        worker_indices = [idxs for idxs in worker_indices if len(idxs) > 0]
        n_workers = len(worker_indices)

        logger.info(
            "Created multiprocessing chunks: %d. Chunk sizes: %s",
            n_workers,
            [int(len(idxs)) for idxs in worker_indices],
        )

        run_arrays = {
            run_name: patient.runs[run_name].data
            for run_name in run_names
        }

        logs_by_run = {
            run_name: patient.logs.get(run_name)
            for run_name in run_names
        }

        # Windows always uses spawn. Use spawn explicitly for consistent behavior.
        mp_context = mp.get_context("spawn")
        manager = mp_context.Manager()
        progress_queue = manager.Queue()

        progress_by_worker: Dict[int, int] = {
            worker_id: 0
            for worker_id in range(n_workers)
        }

        next_report = 500
        progress_step = 100

        executor = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp_context,
        )

        try:
            futures = []

            for worker_id, indices in enumerate(worker_indices):
                args = {
                    "worker_id": worker_id,
                    "indices": indices,
                    "coords": coords[indices],
                    "config": self.config,
                    "run_arrays": run_arrays,
                    "logs_by_run": logs_by_run,
                    "run_names": run_names,
                    "X_star": X_star,
                    "rho_volumes": rho_volumes,
                    "methods_to_run": methods_to_run,
                    "first_level_contrasts": first_level_contrasts,
                    "second_level_contrasts": second_level_contrasts,
                    "images_order": list(self.design_builder.images_order),
                    "progress_queue": progress_queue,
                    "progress_step": progress_step,
                }

                futures.append(executor.submit(_multiprocessing_voxel_worker, args))

            pending = set(futures)

            while pending:
                next_report = self._drain_progress_queue(
                    progress_queue=progress_queue,
                    progress_by_worker=progress_by_worker,
                    total=n_voxels,
                    next_report=next_report,
                )

                done, pending = wait(
                    pending,
                    timeout=0.5,
                    return_when=FIRST_COMPLETED,
                )

                for future in done:
                    result = future.result()
                    worker_id = int(result["worker_id"])
                    n_local = int(len(result["indices"]))

                    self._store_worker_result(
                        result=result,
                        t_store=t_store,
                        T_store=T_store,
                        convergence_store=convergence_store,
                    )

                    progress_by_worker[worker_id] = n_local

                    logger.info(
                        "Worker %d finished its chunk: %d voxels, errors: %d",
                        worker_id + 1,
                        n_local,
                        int(result.get("n_errors", 0)),
                    )

                next_report = self._drain_progress_queue(
                    progress_queue=progress_queue,
                    progress_by_worker=progress_by_worker,
                    total=n_voxels,
                    next_report=next_report,
                )

            total_done = int(sum(progress_by_worker.values()))
            if total_done < n_voxels:
                logger.warning(
                    "Multiprocessing finished, but progress shows %d / %d",
                    total_done,
                    n_voxels,
                )
            else:
                logger.info("Processed voxels: %d / %d", n_voxels, n_voxels)

        except KeyboardInterrupt:
            logger.warning("Ctrl+C received. Stopping the multiprocessing executor.")
            executor.shutdown(wait=False, cancel_futures=True)
            manager.shutdown()
            raise

        finally:
            executor.shutdown(wait=True, cancel_futures=False)
            manager.shutdown()

    @staticmethod
    def _drain_progress_queue(
        progress_queue,
        progress_by_worker: Dict[int, int],
        total: int,
        next_report: int,
    ) -> int:
        while True:
            try:
                msg = progress_queue.get_nowait()
            except queue_module.Empty:
                break
            except Exception:
                break

            if not isinstance(msg, tuple) or len(msg) < 4:
                continue

            kind = msg[0]

            if kind == "progress":
                _, worker_id, done, _n_local, _n_errors = msg
                progress_by_worker[int(worker_id)] = int(done)

            elif kind == "done":
                _, worker_id, n_local, _n_errors = msg
                progress_by_worker[int(worker_id)] = int(n_local)

        total_done = int(sum(progress_by_worker.values()))

        if total_done >= next_report or total_done == total:
            logger.info(
                "Processed voxels: %d / %d",
                min(total_done, total),
                total,
            )

            if total_done >= next_report:
                next_report = ((total_done // 500) + 1) * 500
            else:
                next_report = total + 1

        return next_report

    def _store_worker_result(
        self,
        result: Dict[str, object],
        t_store: Dict[str, Dict[str, np.ndarray]],
        T_store: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        convergence_store: Dict[str, np.ndarray],
    ) -> None:
        indices = np.asarray(result["indices"], dtype=int)

        convergence = result["convergence"]
        glm_diagnostics = result.get("glm_diagnostics", {})
        t_values = result["t_values"]
        T_values = result["T_values"]

        for method, values in convergence.items():
            convergence_store[method][indices] = np.asarray(values, dtype=bool)

        for method, diagnostics in glm_diagnostics.items():
            if method not in self._glm_diagnostics_store:
                continue
            for name, values in diagnostics.items():
                if name in self._glm_diagnostics_store[method]:
                    self._glm_diagnostics_store[method][name][indices] = np.asarray(values)

        for method, contrasts in t_values.items():
            for c_name, arr in contrasts.items():
                t_store[method][c_name][:, indices] = np.asarray(arr, dtype=float)

        for method, contrasts in T_values.items():
            for c_name, second_level_values in contrasts.items():
                for d_name, arr in second_level_values.items():
                    T_store[method][c_name][d_name][indices] = np.asarray(arr, dtype=float)

    def _fit_glm(
        self,
        method: str,
        X_list: List[np.ndarray],
        y_list: List[np.ndarray],
        X_star: np.ndarray,
    ) -> EBResult:
        return _worker_fit_glm(
            method=method,
            eb_model=self.eb_model,
            ols_model=self.ols_model,
            X_list=X_list,
            y_list=y_list,
            X_star=X_star,
        )

    def _prepare_voxel_inputs(
        self,
        patient: PatientData,
        run_names: List[str],
        voxel: Voxel,
        rho_volumes: Optional[Dict[str, np.ndarray]] = None,
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        run_arrays = {name: patient.runs[name].data for name in run_names}
        logs_by_run = {name: patient.logs.get(name) for name in run_names}
        return _worker_prepare_voxel_inputs(
            config=self.config,
            design_builder=self.design_builder,
            run_arrays=run_arrays,
            logs_by_run=logs_by_run,
            run_names=run_names,
            voxel=voxel,
            rho_volumes=rho_volumes,
        )

    def _get_design(self, patient: PatientData, run_name: str, z: int) -> np.ndarray:
        log_df = patient.logs.get(run_name)
        return self.design_builder.build_for_run_and_slice(run_name=run_name, log_df=log_df, z=z)

    def _build_x_star(self, n_runs: int) -> np.ndarray:
        raw = self.config.analysis.x_star

        if raw is None:
            return np.ones((n_runs, 1), dtype=float)

        X_star = np.asarray(raw, dtype=float)

        # MATLAB jsondecode/jsonencode may flatten a single-column matrix
        # [[1.0], [1.0], ...] into [1.0, 1.0, ...] when writing the
        # preprocessed subject.json. Treat this as a valid n_runs x 1 X_star.
        if X_star.ndim == 1:
            X_star = X_star.reshape((-1, 1))

        if X_star.ndim != 2:
            raise ValueError(
                f"x_star must be a 2D matrix or a 1D single-column vector; "
                f"got shape={X_star.shape}"
            )

        if X_star.shape[0] != n_runs:
            raise ValueError(f"x_star has {X_star.shape[0]} rows, but there are {n_runs} runs")

        if X_star.shape[1] < 1:
            raise ValueError("x_star must contain at least one column")

        logger.info("X_star shape: %s", X_star.shape)
        return X_star



    # ------------------------------------------------------------------
    # Output run folder management
    # ------------------------------------------------------------------

    def _patient_output_base(self, patient_id: str) -> Path:
        return self.config.paths.output_path / str(patient_id)

    def _patient_output_dir(self, patient_id: str) -> Path:
        if patient_id not in self._patient_output_dirs:
            return self._prepare_patient_output_dir(patient_id)
        return self._patient_output_dirs[patient_id]

    def _prepare_patient_output_dir(self, patient_id: str) -> Path:
        patient_id = str(patient_id)
        if patient_id in self._patient_output_dirs:
            return self._patient_output_dirs[patient_id]

        output_cfg = getattr(self.config, "output", None)
        mode = str(getattr(output_cfg, "run_folder_mode", "timestamp_hash")).lower().strip()

        patient_base = self._patient_output_base(patient_id)
        patient_base.mkdir(parents=True, exist_ok=True)

        config_hash = self._short_config_hash()

        if mode in {"legacy", "old", "flat"}:
            patient_out = patient_base
            run_id = "legacy"
            run_rel = Path(".")
        elif mode in {"timestamp_hash", "timestamp", "date_hash"}:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_name = self._sanitize_run_name(getattr(output_cfg, "run_name", None))

            parts = [timestamp]
            if run_name:
                parts.append(run_name)
            if bool(getattr(output_cfg, "include_config_hash", True)):
                parts.append(f"cfg_{config_hash}")

            run_id = "_".join(parts)
            patient_out = patient_base / "runs" / run_id
            patient_out = self._apply_existing_run_policy(patient_out)
            run_id = patient_out.name
            run_rel = patient_out.relative_to(patient_base)
        else:
            raise ValueError(
                "Unknown output.run_folder_mode. "
                "Allowed values: 'legacy', 'timestamp_hash'. "
                f"Got: {mode}"
            )

        patient_out.mkdir(parents=True, exist_ok=True)

        metadata = {
            "run_folder_mode": mode,
            "run_id": run_id,
            "config_hash": config_hash,
            "patient_output_base": str(patient_base),
            "patient_output_dir": str(patient_out),
            "run_relative_path": run_rel.as_posix(),
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }

        self._patient_output_dirs[patient_id] = patient_out
        self._run_metadata_by_patient[patient_id] = metadata

        if bool(getattr(output_cfg, "save_settings_snapshot", True)):
            self._write_settings_snapshot(patient_out)

        if bool(getattr(output_cfg, "save_run_manifest", True)):
            self._write_run_manifest(patient_out=patient_out, patient_id=patient_id, metadata=metadata)

        logger.info("Current run output folder: %s", patient_out)
        logger.info("Output run_id=%s, config_hash=%s", run_id, config_hash)

        return patient_out

    def _apply_existing_run_policy(self, path: Path) -> Path:
        output_cfg = getattr(self.config, "output", None)
        policy = str(getattr(output_cfg, "existing_run_policy", "increment")).lower().strip()

        if not path.exists():
            return path

        if policy == "error":
            raise FileExistsError(f"Output folder already exists: {path}")

        if policy == "overwrite":
            shutil.rmtree(path)
            return path

        if policy == "increment":
            base = path
            for i in range(1, 10000):
                candidate = base.with_name(f"{base.name}_{i:03d}")
                if not candidate.exists():
                    return candidate
            raise RuntimeError(f"Could not find a free output folder name for: {base}")

        raise ValueError(
            "Unknown output.existing_run_policy. "
            "Allowed values: 'error', 'overwrite', 'increment'. "
            f"Got: {policy}"
        )

    @staticmethod
    def _sanitize_run_name(value: object) -> str:
        if value is None:
            return ""

        text = str(value).strip()
        if text == "" or text.lower() in {"none", "null"}:
            return ""

        allowed = []
        for ch in text:
            if ch.isalnum() or ch in {"-", "_"}:
                allowed.append(ch)
            elif ch in {" ", ".", ",", ";", ":", "/", "\\"}:
                allowed.append("_")

        sanitized = "".join(allowed).strip("_")
        while "__" in sanitized:
            sanitized = sanitized.replace("__", "_")

        return sanitized[:80]

    def _short_config_hash(self) -> str:
        output_cfg = getattr(self.config, "output", None)
        length = int(getattr(output_cfg, "config_hash_length", 8))
        length = max(4, min(length, 32))

        payload = self._config_hash_payload()
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()[:length]

    def _config_hash_payload(self) -> Dict[str, object]:
        """
        The hash is built from parameters that affect the numerical result.

        Exclude logging and output policy because they control
        where and how logs/folders are written, not the numerical result.
        """
        raw = asdict(self.config)
        raw.pop("logging", None)
        raw.pop("output", None)

        design = raw.get("design")
        if isinstance(design, dict):
            # Runtime-loaded 300x6 arrays are represented by their source hashes
            # in motion_summary_by_run and should not bloat the config hash.
            design.pop("motion_regressors_by_run", None)

        paths = raw.get("paths")
        if isinstance(paths, dict):
            # output_path does not affect the numerical result.
            paths.pop("output_path", None)

        spm = raw.get("spm_preprocessing")
        if isinstance(spm, dict):
            # Working/executable paths should not change the result hash.
            for key in ["work_root", "matlab_executable", "spm_dir"]:
                spm.pop(key, None)

        return self._to_jsonable(raw)

    @classmethod
    def _to_jsonable(cls, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): cls._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._to_jsonable(v) for v in value]
        return value

    def _write_settings_snapshot(self, patient_out: Path) -> None:
        snapshot_path = patient_out / "settings_used.yaml"
        data = self._to_jsonable(asdict(self.config))
        design = data.get("design") if isinstance(data, dict) else None
        if isinstance(design, dict):
            # Keep provenance/QC, not thousands of raw motion values.
            design.pop("motion_regressors_by_run", None)

        try:
            with open(snapshot_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
            logger.info("Configuration snapshot saved: %s", snapshot_path)
        except Exception as exc:
            logger.warning("Could not save settings_used.yaml: %s", exc)

    def _git_commit_hash(self) -> Optional[str]:
        try:
            root = Path(__file__).resolve().parents[1]
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(root),
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode == 0:
                return completed.stdout.strip()
        except Exception:
            return None
        return None

    def _write_run_manifest(self, patient_out: Path, patient_id: str, metadata: Dict[str, object]) -> None:
        manifest_path = patient_out / "run_manifest.json"
        manifest = dict(metadata)
        manifest.update(
            {
                "patient_id": str(patient_id),
                "git_commit": self._git_commit_hash(),
                "configured_glm_method": getattr(self.config.analysis, "glm_method", None),
                "roc_plot_methods": getattr(self.config.analysis, "roc_plot_methods", None),
                "n_jobs": getattr(self.config.analysis, "n_jobs", None),
                "eb_n_restarts": getattr(self.config.empirical_bayes, "n_restarts", None),
                "roc_n_restarts": getattr(self.config.analysis, "roc_n_restarts", None),
            }
        )

        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            logger.info("Run manifest saved: %s", manifest_path)
        except Exception as exc:
            logger.warning("Could not save run_manifest.json: %s", exc)

    def _update_latest_run_pointer(self, patient_id: str) -> None:
        output_cfg = getattr(self.config, "output", None)
        if not bool(getattr(output_cfg, "update_latest_pointer", True)):
            return

        patient_out = self._patient_output_dir(patient_id)
        patient_base = self._patient_output_base(patient_id)

        try:
            rel = patient_out.relative_to(patient_base).as_posix()
        except Exception:
            rel = str(patient_out)

        latest_path = patient_base / "latest_run.txt"
        try:
            latest_path.write_text(rel + "\n", encoding="utf-8")
            logger.info("latest_run.txt updated: %s -> %s", latest_path, rel)
        except Exception as exc:
            logger.warning("Could not update latest_run.txt: %s", exc)


    def _postprocess_and_save(
        self,
        patient: PatientData,
        mask: BrainMask,
        coords: np.ndarray,
        run_names: List[str],
        fixed_main_method: str,
        configured_glm_method: str,
        selection_mode: str,
        roc_methods: List[str],
        t_store: Dict[str, Dict[str, np.ndarray]],
        T_store: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        convergence_store: Dict[str, np.ndarray],
        first_level_contrast_vectors: Dict[str, List[float]],
        second_level_contrast_vectors: Dict[str, List[float]],
    ) -> Dict[str, object]:
        patient_out = self._patient_output_dir(patient.patient_id)
        patient_out.mkdir(parents=True, exist_ok=True)

        glm_diagnostics: Dict[str, object] = {}
        for method, convergence in convergence_store.items():
            diagnostics = self._glm_diagnostics_store.get(method, {})
            fallback = np.asarray(
                diagnostics.get("used_ols_fallback", np.zeros(len(coords), dtype=bool)),
                dtype=bool,
            )
            n_iter = np.asarray(diagnostics.get("n_iter", []), dtype=float)
            omega_condition = np.asarray(
                diagnostics.get("omega_condition_number", []), dtype=float
            )
            omega_min_eig = np.asarray(
                diagnostics.get("omega_min_eigenvalue", []), dtype=float
            )
            finite_condition = omega_condition[np.isfinite(omega_condition)]
            finite_eig = omega_min_eig[np.isfinite(omega_min_eig)]
            method_diag = {
                "n_voxels": int(len(coords)),
                "bayes_converged_voxels": int(np.count_nonzero(convergence))
                if method == "bayes"
                else None,
                "bayes_converged_fraction": float(np.mean(convergence))
                if method == "bayes" and len(convergence)
                else None,
                "ols_fallback_voxels": int(np.count_nonzero(fallback)),
                "ols_fallback_fraction": float(np.mean(fallback)) if len(fallback) else 0.0,
                "iteration_median": float(np.median(n_iter)) if len(n_iter) else None,
                "iteration_max": int(np.max(n_iter)) if len(n_iter) else None,
                "omega_condition_number_median": float(np.median(finite_condition))
                if len(finite_condition)
                else None,
                "omega_condition_number_max": float(np.max(finite_condition))
                if len(finite_condition)
                else None,
                "omega_min_eigenvalue_min": float(np.min(finite_eig))
                if len(finite_eig)
                else None,
            }
            glm_diagnostics[method] = method_diag

            if method == "bayes":
                warn_fraction = float(
                    getattr(self.config.empirical_bayes, "fallback_warn_fraction", 0.25)
                )
                if method_diag["ols_fallback_fraction"] > warn_fraction:
                    logger.warning(
                        "OLS fallback was used for %.2f%% of Bayes voxels (warning threshold %.2f%%)",
                        100.0 * float(method_diag["ols_fallback_fraction"]),
                        100.0 * warn_fraction,
                    )

        summary: Dict[str, object] = {
            "schema_version": "2.0",
            "patient_id": patient.patient_id,
            "output": dict(self._run_metadata_by_patient.get(patient.patient_id, {})),
            "run_names": list(run_names),
            "n_voxels_analyzed": int(len(coords)),
            "brain_mask_voxels": int(mask.n_voxels),
            "configured_glm_method": configured_glm_method,
            "glm_method_selection_mode": selection_mode,
            "glm_methods_computed": list(t_store.keys()),
            "first_level_contrast_vectors": first_level_contrast_vectors,
            "second_level_contrast_vectors": second_level_contrast_vectors,
            "diagnostics": {
                "design": dict(self._design_diagnostics),
                "glm": glm_diagnostics,
                "threshold_grid": {
                    "mode": str(self.config.analysis.threshold_grid),
                    "n_thresholds": int(self.config.analysis.n_thresholds),
                    "minimum": float(self.config.analysis.fixed_threshold_min),
                    "maximum": float(self.config.analysis.fixed_threshold_max),
                },
                "omega_regularization": {
                    "shrinkage_alpha": float(
                        self.config.empirical_bayes.omega_shrinkage_alpha
                    ),
                    "shrinkage_target": str(
                        self.config.empirical_bayes.omega_shrinkage_target
                    ),
                    "ridge": float(self.config.empirical_bayes.omega_ridge),
                    "eigenvalue_floor_relative": float(
                        self.config.empirical_bayes.omega_eigenvalue_floor_relative
                    ),
                },
            },
            "contrasts": {},
        }

        directions = self._reproducibility_directions()
        primary_direction = self._primary_reproducibility_direction(directions)
        contrast_names = list(next(iter(t_store.values())).keys())

        for c_name in contrast_names:
            logger.info("ROC/Kappa for contrast: %s", c_name)
            contrast_out = patient_out / c_name
            contrast_out.mkdir(parents=True, exist_ok=True)

            roc_models: Dict[str, object] = {}
            for method in t_store.keys():
                roc_models[method] = self._compute_roc_models_for_method(
                    method=method,
                    t_runwise=t_store[method][c_name],
                    directions=directions,
                )

            if selection_mode == "choose_best_kappa":
                allowed_methods = list(t_store.keys())
                selection_reason = "best_valid_kappa_across_glm_and_lambda"
            else:
                allowed_methods = [fixed_main_method]
                selection_reason = "fixed_glm_best_valid_lambda"

            configured_map_lambda_mode = str(
                getattr(self.config.analysis, "roc_map_lambda_mode", "")
            ).lower().strip()
            if configured_map_lambda_mode not in {"free", "fixed"}:
                legacy_mode = str(
                    getattr(self.config.analysis, "roc_primary_lambda_mode", "fixed")
                ).lower().strip()
                configured_map_lambda_mode = (
                    legacy_mode if legacy_mode in {"free", "fixed"} else "fixed"
                )

            computed_lambda_modes = set(self.roc_model.lambda_optimization_modes())
            if configured_map_lambda_mode not in computed_lambda_modes:
                raise ValueError(
                    "analysis.roc_map_lambda_mode must be included in "
                    "analysis.roc_lambda_optimization_mode. "
                    f"Requested map mode={configured_map_lambda_mode!r}, "
                    f"computed modes={sorted(computed_lambda_modes)}"
                )

            common_selection_reason = (
                "best_glm_with_configured_common_lambda_mode"
                if selection_mode == "choose_best_kappa"
                else "configured_common_lambda_mode"
            )
            selected_model = self._select_best_valid_operating_point(
                roc_models=roc_models,
                methods=allowed_methods,
                direction=primary_direction,
                lambda_modes=[configured_map_lambda_mode],
                selection_reason=common_selection_reason,
            )

            primary_d = self.config.analysis.primary_second_level_contrast
            selected_method = str(selected_model.get("method") or fixed_main_method)
            if selected_method not in T_store:
                selected_method = fixed_main_method
            if primary_d not in T_store[selected_method][c_name]:
                primary_d = next(iter(T_store[selected_method][c_name].keys()))

            maps_by_direction: Dict[str, object] = {}
            tvalue_plot_info = None

            for direction in directions:
                if direction == primary_direction:
                    direction_selection = selected_model
                else:
                    direction_selection = self._select_best_valid_operating_point(
                        roc_models=roc_models,
                        methods=[selected_method],
                        direction=direction,
                        lambda_modes=[configured_map_lambda_mode],
                        selection_reason="configured_common_lambda_mode",
                    )

                direction_info: Dict[str, object] = {
                    "valid": bool(direction_selection.get("valid", False)),
                    "selection": direction_selection,
                }
                if not direction_info["valid"]:
                    direction_info["maps_written"] = False
                    maps_by_direction[direction] = direction_info
                    logger.warning(
                        "No valid ROC operating point for contrast=%s, direction=%s: %s",
                        c_name,
                        direction,
                        direction_selection.get("invalid_reason"),
                    )
                    continue

                method = str(direction_selection["method"])
                threshold = float(direction_selection["operating_point"]["threshold_value"])
                selected_t_runwise = t_store[method][c_name]
                selected_T_values = T_store[method][c_name][primary_d]
                repro = compute_reproducibility_map(
                    t_runwise=selected_t_runwise,
                    T_values=selected_T_values,
                    threshold=threshold,
                    mask=mask.mask,
                    coords=coords,
                    config=self.config.analysis,
                    direction=direction,
                )

                direction_out = contrast_out / direction
                direction_out.mkdir(parents=True, exist_ok=True)
                if self.config.analysis.save_nifti:
                    self._save_reproducibility_niftis(repro, mask, direction_out)
                    if direction == primary_direction:
                        # Backward-compatible root maps for the GUI. These are the
                        # directional primary maps, not unsigned maps.
                        self._save_reproducibility_niftis(repro, mask, contrast_out)

                if direction == "positive":
                    active = selected_t_runwise >= threshold
                elif direction == "negative":
                    active = selected_t_runwise <= -threshold
                else:
                    active = np.abs(selected_t_runwise) >= threshold
                R_counts_vec = np.sum(active & np.isfinite(selected_t_runwise), axis=0).astype(int)

                if direction == primary_direction and getattr(
                    self.config.analysis, "save_tvalue_plots", True
                ):
                    try:
                        tvalue_plot_info = save_tvalue_plots(
                            T_values=selected_T_values,
                            R_counts=R_counts_vec,
                            n_runs=len(run_names),
                            config=self.config.analysis,
                            out_dir=contrast_out,
                        )
                    except Exception as exc:
                        logger.warning("Could not save T-value plots: %s", exc)

                direction_info.update(
                    {
                        "maps_written": bool(self.config.analysis.save_nifti),
                        "output_dir": str(direction_out),
                        "strong_voxels": int(np.sum(repro.strong)),
                        "moderate_voxels": int(np.sum(repro.moderate)),
                        "weak_voxels": int(np.sum(repro.weak)),
                        "final_voxels": int(np.sum(repro.final)),
                    }
                )
                maps_by_direction[direction] = direction_info

            contrast_summary: Dict[str, object] = {
                "schema_version": "2.0",
                "contrast": c_name,
                "configured_glm_method": configured_glm_method,
                "primary_second_level_contrast": primary_d,
                "primary_direction": primary_direction,
                "roc_map_lambda_mode": configured_map_lambda_mode,
                "selected_model": selected_model,
                "roc_models": roc_models,
                "maps_by_direction": maps_by_direction,
                "t_value_plots": tvalue_plot_info,
                "diagnostics": {
                    "design": dict(self._design_diagnostics),
                    "glm": glm_diagnostics,
                    "finite_voxels_by_method": {
                        method: int(
                            np.count_nonzero(
                                np.all(np.isfinite(t_store[method][c_name]), axis=0)
                            )
                        )
                        for method in t_store
                    },
                },
                "output": {
                    "patient_output_dir": str(patient_out),
                    "contrast_output_dir": str(contrast_out),
                },
            }

            safe_contrast_summary = self._json_safe(contrast_summary)
            with open(contrast_out / "summary.json", "w", encoding="utf-8") as f:
                json.dump(
                    safe_contrast_summary,
                    f,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )

            summary["contrasts"][c_name] = {
                "summary_file": str(contrast_out / "summary.json"),
                "roc_map_lambda_mode": safe_contrast_summary["roc_map_lambda_mode"],
                "selected_model": safe_contrast_summary["selected_model"],
                "maps_by_direction": safe_contrast_summary["maps_by_direction"],
            }
            summary["output"]["primary_contrast_dir"] = str(contrast_out)

            try:
                plot_methods = (
                    "both"
                    if selection_mode == "choose_best_kappa"
                    else self.config.analysis.roc_plot_methods
                )
                selections_by_direction = {
                    direction: maps_by_direction.get(direction, {}).get("selection", {})
                    for direction in directions
                }

                # Main plot used by the GUI: positive and negative ROC curves
                # together, always using the same lambda strategy as the maps.
                save_selected_lambda_direction_plot(
                    roc_models=roc_models,
                    selections_by_direction=selections_by_direction,
                    out_path=contrast_out / "roc_kappa.png",
                    method=selected_method,
                    lambda_mode=configured_map_lambda_mode,
                    directions=directions,
                    show_points=self.config.analysis.roc_plot_show_points,
                    show_optimal_point=self.config.analysis.roc_plot_show_optimal_point,
                    dpi=self.config.analysis.roc_plot_dpi,
                )

                # Keep direction-specific diagnostic files, but filter them to
                # the same configured lambda strategy instead of showing both.
                for plot_direction in directions:
                    direction_selected = selections_by_direction.get(plot_direction, {})
                    legacy_plot_summary = self._legacy_plot_summary_from_roc_models(
                        roc_models=roc_models,
                        selected_model=(
                            direction_selected
                            if isinstance(direction_selected, dict)
                            else {}
                        ),
                        direction=plot_direction,
                        configured_glm_method=configured_glm_method,
                        allowed_lambda_modes=[configured_map_lambda_mode],
                    )
                    save_roc_summary_plot(
                        contrast_summary=legacy_plot_summary,
                        out_path=contrast_out / f"roc_kappa_{plot_direction}.png",
                        methods=plot_methods,
                        show_curve=self.config.analysis.roc_plot_show_curve,
                        show_points=self.config.analysis.roc_plot_show_points,
                        show_kappa=self.config.analysis.roc_plot_show_kappa,
                        show_optimal_point=self.config.analysis.roc_plot_show_optimal_point,
                        show_eroc=self.config.analysis.roc_plot_show_eroc,
                        dpi=self.config.analysis.roc_plot_dpi,
                    )
            except Exception as exc:
                logger.warning("Could not save ROC/Kappa plot: %s", exc)

        safe_summary = self._json_safe(summary)
        with open(patient_out / "summary.json", "w", encoding="utf-8") as f:
            json.dump(safe_summary, f, ensure_ascii=False, indent=2, allow_nan=False)

        self._update_latest_run_pointer(patient.patient_id)
        logger.info("Results saved in %s", patient_out)
        return safe_summary

    def _reproducibility_directions(self) -> List[str]:
        raw = getattr(self.config.analysis, "reproducibility_directions", ["positive"])
        values = raw if isinstance(raw, (list, tuple)) else [raw]
        result: List[str] = []
        for value in values:
            normalized = self.roc_model._normalize_direction(str(value))
            if normalized not in result:
                result.append(normalized)
        return result or ["positive"]

    def _primary_reproducibility_direction(self, directions: List[str]) -> str:
        raw = getattr(self.config.analysis, "primary_reproducibility_direction", "positive")
        value = self.roc_model._normalize_direction(str(raw))
        return value if value in directions else directions[0]

    def _compute_roc_models_for_method(
        self,
        method: str,
        t_runwise: np.ndarray,
        directions: List[str],
    ) -> Dict[str, object]:
        result: Dict[str, object] = {}
        lambda_modes = self.roc_model.lambda_optimization_modes()

        for direction in directions:
            direction_models: Dict[str, object] = {}
            for lambda_mode in lambda_modes:
                logger.info(
                    "ROC/Kappa: method=%s, direction=%s, lambda_mode=%s",
                    method,
                    direction,
                    lambda_mode,
                )
                roc_result = self.roc_model.fit(
                    t_runwise,
                    lambda_optimization_mode=lambda_mode,
                    direction=direction,
                )
                threshold = choose_threshold_by_kappa(roc_result, self.config.analysis)
                thresholds, sensitivity, false_alarm = self.roc_model.roc_points(roc_result)
                kappas = kappa_curve(roc_result)

                eroc_params = None
                if self.config.analysis.fit_england_eroc:
                    E, R = EnglandEROC.fit(
                        false_alarm=false_alarm,
                        sensitivity=sensitivity,
                        loss=self.config.analysis.eroc_fit_loss,
                    )
                    eroc_params = {
                        "E": E,
                        "R": R,
                        "auc": E / (E + 1.0),
                        "loss": self.config.analysis.eroc_fit_loss,
                    }

                direction_models[lambda_mode] = {
                    "valid": bool(threshold.valid),
                    "invalid_reason": threshold.invalid_reason,
                    "lambda": float(roc_result.lambda_),
                    "em_converged": bool(roc_result.converged),
                    "em_iterations": int(roc_result.n_iter),
                    "stochastic_dominance_valid": bool(
                        roc_result.stochastic_dominance_valid
                    ),
                    "dominance_min_margin": float(roc_result.dominance_min_margin),
                    "auc": float(roc_result.auc),
                    "operating_point": threshold.__dict__,
                    "optimization": getattr(roc_result, "optimization_info", None),
                    "curve": {
                        "thresholds": thresholds.tolist(),
                        "sensitivity": sensitivity.tolist(),
                        "false_alarm": false_alarm.tolist(),
                        "kappa": kappas.tolist(),
                        "england_eroc": eroc_params,
                        "approximated": {
                            "method": str(getattr(roc_result, "approximation_method", "none")),
                            "thresholds": (
                                getattr(roc_result, "approximated_thresholds", None).tolist()
                                if getattr(roc_result, "approximated_thresholds", None) is not None
                                else []
                            ),
                            "sensitivity": (
                                getattr(roc_result, "approximated_sensitivity", None).tolist()
                                if getattr(roc_result, "approximated_sensitivity", None) is not None
                                else []
                            ),
                            "false_alarm": (
                                getattr(roc_result, "approximated_false_alarm", None).tolist()
                                if getattr(roc_result, "approximated_false_alarm", None) is not None
                                else []
                            ),
                            "kappa": (
                                getattr(roc_result, "approximated_kappa", None).tolist()
                                if getattr(roc_result, "approximated_kappa", None) is not None
                                else []
                            ),
                            "auc": float(getattr(roc_result, "approximated_auc", np.nan)),
                        },
                    },
                }
            result[direction] = direction_models
        return result

    def _select_best_valid_operating_point(
        self,
        roc_models: Dict[str, object],
        methods: List[str],
        direction: str,
        lambda_modes: Optional[List[str]],
        selection_reason: str,
    ) -> Dict[str, object]:
        candidates: List[Dict[str, object]] = []
        invalid_reasons: List[str] = []

        for method in methods:
            method_models = roc_models.get(method, {})
            if not isinstance(method_models, dict):
                continue
            direction_models = method_models.get(direction, {})
            if not isinstance(direction_models, dict):
                continue
            for lambda_mode, item in direction_models.items():
                if lambda_modes is not None and lambda_mode not in lambda_modes:
                    continue
                if not isinstance(item, dict):
                    continue
                operating_point = item.get("operating_point", {})
                if not item.get("valid", False):
                    invalid_reasons.append(
                        f"{method}/{lambda_mode}: {item.get('invalid_reason')}"
                    )
                    continue
                candidates.append(
                    {
                        "valid": True,
                        "method": method,
                        "direction": direction,
                        "lambda_mode": lambda_mode,
                        "lambda": item.get("lambda"),
                        "operating_point": operating_point,
                        "selection_reason": selection_reason,
                    }
                )

        if not candidates:
            return {
                "valid": False,
                "method": methods[0] if methods else None,
                "direction": direction,
                "lambda_mode": None,
                "operating_point": None,
                "selection_reason": selection_reason,
                "invalid_reason": "; ".join(invalid_reasons)
                or "no_valid_roc_operating_point",
            }

        tolerance = float(getattr(self.config.analysis, "roc_selection_tolerance", 1e-9))
        best = candidates[0]
        for candidate in candidates[1:]:
            a = candidate["operating_point"]
            b = best["operating_point"]
            ka, kb = float(a["kappa"]), float(b["kappa"])
            replace = ka > kb + tolerance
            if abs(ka - kb) <= tolerance:
                fa, fb = float(a["false_alarm"]), float(b["false_alarm"])
                replace = fa < fb - tolerance
                if abs(fa - fb) <= tolerance:
                    sa, sb = float(a["sensitivity"]), float(b["sensitivity"])
                    replace = sa > sb + tolerance
                    if abs(sa - sb) <= tolerance:
                        replace = (
                            candidate["lambda_mode"] == "fixed"
                            and best["lambda_mode"] != "fixed"
                        )
            if replace:
                best = candidate
        return best

    @staticmethod
    def _legacy_plot_summary_from_roc_models(
        roc_models: Dict[str, object],
        selected_model: Dict[str, object],
        direction: str,
        configured_glm_method: str,
        allowed_lambda_modes: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        methods: Dict[str, object] = {}
        for method, method_models in roc_models.items():
            if not isinstance(method_models, dict):
                continue
            direction_models = method_models.get(direction, {})
            if not isinstance(direction_models, dict):
                continue
            lambda_items: Dict[str, object] = {}
            for lambda_mode, item in direction_models.items():
                if allowed_lambda_modes is not None and lambda_mode not in allowed_lambda_modes:
                    continue
                if not isinstance(item, dict):
                    continue
                curve = item.get("curve", {})
                lambda_items[lambda_mode] = {
                    "threshold": item.get("operating_point", {}),
                    "roc": {
                        "lambda": item.get("lambda"),
                        "lambda_optimization_mode": lambda_mode,
                        "converged": item.get("em_converged"),
                        "n_iter": item.get("em_iterations"),
                        "optimization": item.get("optimization"),
                        "thresholds": curve.get("thresholds", []),
                        "sensitivity": curve.get("sensitivity", []),
                        "false_alarm": curve.get("false_alarm", []),
                        "kappa": curve.get("kappa", []),
                        "england_eroc": curve.get("england_eroc"),
                        "approximated": curve.get("approximated", {}),
                    },
                }
            methods[method] = {
                "lambda_modes": lambda_items,
                "primary_lambda_optimization_mode": selected_model.get("lambda_mode", "free"),
            }
        return {
            "configured_glm_method": configured_glm_method,
            "primary_direction": direction,
            "main_glm_method": selected_model.get("method", "bayes"),
            "selected_glm_method": selected_model.get("method", "bayes"),
            "selected_lambda_mode": selected_model.get("lambda_mode", "free"),
            "roc_lambda_optimization_modes": (
                list(allowed_lambda_modes)
                if allowed_lambda_modes is not None
                else ["free", "fixed"]
            ),
            "methods": methods,
        }

    @classmethod
    def _json_safe(cls, value):
        if isinstance(value, dict):
            return {str(k): cls._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return cls._json_safe(value.tolist())
        if isinstance(value, (np.floating, float)):
            numeric = float(value)
            return numeric if np.isfinite(numeric) else None
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value

    def _save_reproducibility_niftis(
        self,
        repro: ReproducibilityMap,
        mask: BrainMask,
        out_dir: Path,
    ) -> None:
        maps = {
            "R.nii.gz": repro.R.astype(np.float32),
            "rate.nii.gz": repro.rate.astype(np.float32),
            "strong.nii.gz": repro.strong.astype(np.uint8),
            "moderate.nii.gz": repro.moderate.astype(np.uint8),
            "weak.nii.gz": repro.weak.astype(np.uint8),
            "final.nii.gz": repro.final.astype(np.uint8),
            "T.nii.gz": repro.T.astype(np.float32),
            "mean_t.nii.gz": repro.mean_t.astype(np.float32),
            "sign.nii.gz": repro.sign.astype(np.float32),
            "R_pos.nii.gz": repro.R_pos.astype(np.float32),
            "R_neg.nii.gz": repro.R_neg.astype(np.float32),
            "rate_pos.nii.gz": repro.rate_pos.astype(np.float32),
            "rate_neg.nii.gz": repro.rate_neg.astype(np.float32),
            "reproducibility_pos.nii.gz": repro.reproducibility_pos.astype(np.float32),
            "reproducibility_neg.nii.gz": repro.reproducibility_neg.astype(np.float32),
            "T_pos.nii.gz": repro.T_pos.astype(np.float32),
            "T_neg.nii.gz": repro.T_neg.astype(np.float32),
        }

        for name, data in maps.items():
            img = nib.Nifti1Image(data, affine=mask.affine, header=mask.header)
            nib.save(img, str(out_dir / name))
            
    def _save_brain_mask_nifti(
        self,
        mask: BrainMask,
        patient_id: str,
    ) -> None:
        patient_out = self._patient_output_dir(patient_id)
        patient_out.mkdir(parents=True, exist_ok=True)

        mask_path = patient_out / "brain_mask.nii.gz"

        img = nib.Nifti1Image(
            mask.mask.astype(np.uint8),
            affine=mask.affine,
            header=mask.header,
        )
        nib.save(img, str(mask_path))

        logger.info(
            "Brain mask saved: %s, voxels=%d",
            mask_path,
            int(np.sum(mask.mask)),
        )
        
    def _save_mean_functional_nifti(self, patient: PatientData) -> None:
        """
        Save the mean functional image in functional space.

        This is needed for correct GUI visualization:
        activity maps, brain_mask.nii.gz, and mean_functional.nii.gz
        share the same grid and affine.
        """
        patient_out = self._patient_output_dir(patient.patient_id)
        patient_out.mkdir(parents=True, exist_ok=True)

        run_names = sorted(patient.runs.keys())
        if not run_names:
            return

        mean_sum = None
        n_mean = 0

        for run_name in run_names:
            data = np.asarray(patient.runs[run_name].data, dtype=np.float32)
            run_mean = np.nanmean(data, axis=0).astype(np.float32, copy=False)

            if mean_sum is None:
                mean_sum = np.zeros_like(run_mean, dtype=np.float32)

            mean_sum += run_mean
            n_mean += 1

            # Release the temporary mean before moving to the next run.
            del run_mean

        if mean_sum is None or n_mean == 0:
            return

        mean_functional = mean_sum / float(n_mean)

        first_run = patient.runs[run_names[0]]

        out_path = patient_out / "mean_functional.nii.gz"

        img = nib.Nifti1Image(
            mean_functional.astype(np.float32),
            affine=first_run.affine,
            header=first_run.header,
        )
        nib.save(img, str(out_path))

        logger.info("Mean functional image saved: %s", out_path)