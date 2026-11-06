# file: config.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


from spm_preprocessing import SPMPreprocessingConfig

import yaml


@dataclass
class PathsConfig:
    data_path: Path = Path("../data")
    logs_path: Path = Path("../logs")
    output_path: Path = Path("../results")
    mask_path: Optional[Path] = None
    patient_id: Optional[str] = None

    def resolve_relative(self, base_dir: Path) -> None:
        self.data_path = _resolve_path(self.data_path, base_dir, patient_id=self.patient_id)
        self.logs_path = _resolve_path(self.logs_path, base_dir, patient_id=self.patient_id)
        self.output_path = _resolve_path(self.output_path, base_dir, patient_id=self.patient_id)

        if self.mask_path is not None:
            mask_text = str(self.mask_path).strip()
            if mask_text.lower() in {"", "none", "null", "auto"}:
                self.mask_path = None
            else:
                self.mask_path = _resolve_path(self.mask_path, base_dir, patient_id=self.patient_id)

        self.output_path.mkdir(parents=True, exist_ok=True)


@dataclass
class ScannerConfig:
    nX: int = 53
    nY: int = 63
    nZ: int = 52
    nT: int = 72
    nS: int = 18
    nR: int = 6
    TR: float = 2.5
    slice_time_step: float = 0.1
    slice_timing: Optional[List[float]] = None
    # Optional run-specific slice timing arrays loaded from subject.json.
    # Keys are run_id values, values are seconds from volume onset.
    slice_timing_by_run: Dict[str, List[float]] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.nX, self.nY, self.nZ

    @property
    def n_voxels(self) -> int:
        return self.nX * self.nY * self.nZ



@dataclass
class DataConfig:
    run_mode: str = "+"
    banned_run_names: List[str] = field(default_factory=lambda: ["log", "logs", "structural"])
    skip_patient_ids: List[str] = field(default_factory=list)
    run_name_prefixes: List[str] = field(default_factory=lambda: ["fr", "me"])

    log_delimiter: str = "\t"

    # Log columns used by the legacy code
    log_block_column: str = "values.TestBlockNum"
    log_stimulus_column: str = "stimulusitem1"
    log_prestim_column: str = "values.PreStimInterval"
    log_trial_column: str = "values.TestTrialNum"
    log_response_column: str = "values.ResponseForSubj"

    # Fallback fields, currently rarely used
    log_me_morph_column: str = "values.MeMorphShare"
    log_fr_morph_column: str = "values.FrMorphShare"
    
    log_onset_column: str = "offset"
    log_duration_column: str = "duration"
    log_condition_column: str = "trial_type"

    # Fixed condition order for condition-level GLM designs.
    # If non-empty, these conditions define the first columns of X_i.
    condition_order: List[str] = field(default_factory=list)

    default_event_duration: float = 2.5

    # Legacy onset formula:
    # onset = onset_base_shift_sec + s * stimulus_period_sec + prestim / 1000
    stimulus_period_sec: float = 10.0
    onset_base_shift_sec: float = 0.0

    # Filter logs and scans by the common stimulus set
    enable_common_stimulus_filter: bool = True
    keep_even_log_rows_only: bool = True
    remove_zero_trial_rows: bool = True

    common_stimulus_patterns: List[str] = field(default_factory=lambda: ["MeFr", "FrMe"])

    stimulus_rename_map: Dict[str, str] = field(default_factory=lambda: {
        "FrMe_06.png": "MeFr_04.png",
        "FrMe_08.png": "MeFr_02.png",
        "FrMe_10.png": "MeFr_00.png",
        "FrMe_00.png": "MeFr_10.png",
        "FrMe_04.png": "MeFr_06.png",
        "FrMe_02.png": "MeFr_08.png",
    })

    append_trial_group_to_stimulus_name: bool = True
    trial_group_size: int = 12

    scans_per_trial: int = 4

    # Legacy code used:
    # scans_numbers = [i for i in range(trial_number * 4, trial_number * 4 + 4)]
    # Therefore the default offset is 0.
    # If later the required formula is (trial_number - 1) * 4, set this to -4.
    scan_index_offset: int = 0

    strict_scan_filtering: bool = True


@dataclass
class DesignConfig:
    design_type: str = "event_regressors"
    # Canonical SPM double-gamma HRF. Legacy models remain available in design.py.
    hrf_model: str = "spm_canonical"
    hrf_length_sec: float = 32.0
    hrf_oversampling: int = 16
    use_slice_timing: bool = True

    add_intercept: bool = True
    add_drift_terms: bool = False
    drift_kind: str = "cosine"
    drift_order: int = 3
    high_pass_cutoff_sec: float = 128.0

    # SPM realignment parameters are nuisance regressors, not random effects.
    add_motion_regressors: bool = False
    motion_model: str = "6"
    center_motion_regressors: bool = True
    motion_regressors_by_run: Dict[str, List[List[float]]] = field(default_factory=dict)
    motion_summary_by_run: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Remove nuisance effects in an orthonormal complement before fitting the
    # hierarchical five-parameter stimulus model. This keeps Omega at nS x nS.
    residualize_nuisance: bool = True
    nuisance_rank_tolerance: float = 1e-10
    normalize_columns: bool = False


@dataclass
class PreprocessingConfig:
    prewhitening: bool = True
    estimate_rho_from_residuals: bool = True
    rho_bias_correction: bool = True
    fixed_rho: float = 0.5
    rho_clip: float = 0.99
    rho_spatial_smoothing_fwhm: Optional[float] = 15.0

    mask_method: str = "spm_tissue"

    spm_mask_mode: str = "probability_sum"
    spm_mask_tissue_classes: List[int] = field(default_factory=lambda: [1, 2, 3])
    spm_mask_probability_eps: float = 0.0
    spm_mask_threshold: float = 0.5
    spm_mask_resample_to_functional: bool = True
    spm_mask_intersect_with_functional_nonzero: bool = True
    spm_mask_functional_nonzero_eps: float = 0.0

    # Functional mask parameters.
    mask_use_all_runs: bool = True
    mask_intensity_percentile: float = 35.0
    mask_otsu_n_bins: int = 256

    # Clean small components from the mask.
    mask_keep_largest_component: bool = True
    mask_opening_iterations: int = 0
    mask_closing_iterations: int = 0
    mask_fill_holes: bool = True
    mask_min_voxels: int = 1000

    # Reject automatically generated masks that cover too much of the fMRI
    # volume. A whole-FOV mask is usually caused by interpolation/background
    # values after SPM reslicing and leads to spurious full-slice activity maps.
    mask_reject_too_large: bool = True
    mask_max_volume_fraction: float = 0.80
    mask_min_volume_fraction: float = 0.01
    
    # How to transfer the structural brain mask to the functional grid.
    #
    # "fractional_occupancy":
    #     for each fMRI voxel, compute the fraction of structural voxel centers
    #     that fall into this fMRI voxel and lie inside the structural brain mask.
    #     The fMRI voxel is included in the mask if the fraction is >= spm_mask_fraction_threshold.
    #
    # "resample_probability":
    #     legacy mode: resample the probability map and threshold it.
    spm_mask_transfer_method: str = "fractional_occupancy"
    spm_mask_fraction_threshold: float = 0.5


@dataclass
class EmpiricalBayesConfig:
    max_iter: int = 400
    tol: float = 1e-6
    likelihood_tol: float = 1e-6
    likelihood_check_interval: int = 5
    omega_ridge: float = 1e-6
    # Convex shrinkage is substantive regularization; omega_ridge is only a
    # numerical floor. With four runs and five stimulus coefficients the raw
    # covariance is necessarily unstable without shrinkage.
    omega_shrinkage_alpha: float = 0.25
    omega_shrinkage_target: str = "diagonal"
    omega_eigenvalue_floor_relative: float = 1e-4
    b_ridge: float = 0.0
    min_sigma2: float = 1e-10
    fallback_to_ols: bool = True
    fallback_warn_fraction: float = 0.25

    # Number of EM GLM runs with different initializations.
    # 1: only the current OLS-based initialization.
    # 2: OLS-based initialization plus one random run.
    # N: OLS-based initialization plus (N-1) random runs.
    n_restarts: int = 1

    # Seed for random EM GLM initializations.
    # None/null: each run will be nondeterministic.
    random_seed: Optional[int] = 12345

    # Scale of random beta initializations relative to the OLS beta scale.
    # 1.0 is usually sufficient; larger values produce wider starts.
    random_init_scale: float = 1.0

    # Criterion for selecting the best EM GLM restart.
    # Currently only the observed-data log likelihood is supported.
    restart_selection_criterion: str = "log_likelihood"

    # liou_reml: divide Omega by M - rank(X_star)
    # ml: divide Omega by M
    omega_denominator: str = "liou_reml"

    # liou_plugin: sigma2 = RSS / (n - k)
    # strict_em: sigma2 = (RSS + tr(X H X^T)) / n
    sigma_update: str = "liou_plugin"

    verbose: bool = False


@dataclass
class NamedContrast:
    values: List[float]
    auto_pad_to_nS: bool = False
    auto_pad_to_k_star: bool = False


@dataclass
class AnalysisConfig:
    first_level_contrasts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Optional list of first-level contrast names to compute.
    # None or empty: compute all configured contrasts.
    first_level_contrast_names: Optional[List[str]] = None
    second_level_contrasts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    primary_second_level_contrast: str = "average"

    x_star: Optional[List[List[float]]] = None
    
    # GLM method for the main map calculation:
    # "bayes"             — empirical Bayes / Liou-style EM
    # "ols"               — ordinary least squares
    # "choose_best_kappa" — calculate Bayes and OLS and select the map by maximum Kappa
    glm_method: str = "bayes"

    # Which ROC curves to compute and plot:
    # "bayes"      — Bayes only
    # "ols"        — OLS only
    # "both"       — both curves on one plot
    # "glm_method" — only the method from glm_method
    roc_plot_methods: str = "both"

    roc_plot_show_curve: bool = True
    roc_plot_show_points: bool = True
    roc_plot_show_kappa: bool = True
    roc_plot_show_optimal_point: bool = True
    roc_plot_show_eroc: bool = True
    roc_plot_dpi: int = 160

    # Threshold grid for ROC:
    # "quantiles" — uniform quantiles of abs(t) from 0 to 1, without endpoints
    # "linspace"  — uniform grid from min(evidence) to max(evidence).
    # "fixed"     — one common grid including fixed_threshold_min/max.
    threshold_grid: str = "fixed"
    n_thresholds: int = 80
    fixed_threshold_min: float = 0.0
    fixed_threshold_max: float = 8.0

    # Variant of the lambda likelihood:
    # "none"                  — without the I(lambda <= rho) multiplier
    # "indicator_upper_bound" — with the I(lambda <= rho) multiplier.
    # Liou-style ROC uses the upper-bound indicator; keeping it enabled avoids
    # degenerate fits where free lambda expands to most of the mask.
    lambda_likelihood_mode: str = "indicator_upper_bound"

    # rho in Liou notation: upper bound for lambda
    lambda_upper_bound: float = 0.05

    # Initial lambda value for EM
    lambda_init: float = 0.025

    # Legacy parameter kept for compatibility.
    # If lambda_likelihood_mode is not set in YAML, use_lambda_upper_bound can still be used.
    use_lambda_upper_bound: bool = False

    roc_max_iter: int = 200
    roc_tol: float = 1e-6
    # ROC mixture likelihood optimization by multi-start EM.
    roc_n_restarts: int = 30
    roc_parallel_restarts: bool = True
    roc_restart_jobs: Optional[int] = None
    roc_random_seed: Optional[int] = 12345
    roc_min_probability: float = 1e-12
    fit_england_eroc: bool = False
    
    # Lambda optimization mode in the ROC mixture likelihood:
    # "free"  — EM estimates lambda, P_A, and P_I;
    # "fixed" — lambda is fixed; EM estimates only P_A and P_I;
    # "both"  — both modes are computed, and the plot shows two ROC curves.
    roc_lambda_optimization_mode: str = "both"
    # Lambda mode used for map generation in every reproducibility direction.
    # "fixed" uses roc_fixed_lambda for both positive and negative maps.
    # "free" estimates lambda separately for each direction, but uses the same
    # optimization strategy for both directions.
    roc_map_lambda_mode: str = "fixed"

    # Deprecated compatibility alias. It is used only when roc_map_lambda_mode
    # is absent in an older configuration.
    roc_primary_lambda_mode: str = "best_valid_kappa"
    roc_fixed_lambda: float = 0.05
    roc_stochastic_dominance: bool = True
    roc_dominance_margin: float = 1e-6
    roc_min_auc: float = 0.5
    roc_min_kappa: float = 0.0
    roc_reject_boundary_threshold: bool = True
    roc_selection_tolerance: float = 1e-9

    # Approximate the discrete ROC points before choosing the Kappa optimum.
    # "pchip" uses a shape-preserving cubic interpolation as a function of the
    # common t-threshold grid; "linear" uses linear interpolation; "none"
    # preserves the original discrete points.
    roc_curve_approximation: str = "pchip"
    roc_approximation_points: int = 801
    roc_kappa_use_approximated_curve: bool = True

    roc_parallel_backend: str = "thread"
    roc_log_restart_progress: bool = True

    # EM initialization modes for the ROC mixture likelihood.
    # ordered_random — random inactive distribution plus an ordered exponential
    #                  tilt for the active distribution (FOSD-valid).
    # split_random   — separated low-bin/high-bin FOSD-valid initialization.
    roc_init_modes: List[str] = field(default_factory=lambda: [
        "random",
        "split_random",
    ])

    # Loss function for fitting England EROC parameters to ROC points:
    # "rmse" — root mean squared error
    # "mae"  — mean absolute error
    # "max"  — maximum absolute error
    eroc_fit_loss: str = "rmse"

    # Directional reproducibility. Positive is the primary Self > Malleable
    # analysis; negative is evaluated separately as Malleable > Self.
    # Legacy projects default to unsigned Liou classification. Dataset-specific
    # configs such as settings_diva_sorpf.yaml explicitly request positive and
    # negative one-sided analyses.
    reproducibility_directions: List[str] = field(default_factory=lambda: ["absolute"])
    primary_reproducibility_direction: str = "absolute"

    strong_reproducibility: float = 0.90
    moderate_reproducibility: float = 0.70
    weak_reproducibility: float = 0.50
    include_moderate_neighbors: bool = True

    n_jobs: int = 1
    # Backend for voxel-level parallelism.
    # "process_memmap" uses real Python processes and shared read-only NumPy
    # memmaps, so CPU-bound voxel loops are not limited by the GIL and large
    # fMRI arrays are not pickled into every Windows worker.
    # "thread" is memory-light but may underuse CPU for Python-heavy voxel loops.
    # "process_copy" is kept only for debugging; it copies BOLD arrays into workers.
    voxel_parallel_backend: str = "process_memmap"
    # Limit BLAS/OpenMP inside each worker process. Voxel-level parallelism is
    # already handled by n_jobs; keeping this at 1 avoids CPU oversubscription.
    blas_threads: int = 1
    max_voxels: Optional[int] = None
    save_nifti: bool = True
    
    save_tvalue_plots: bool = True
    tvalue_hist_bins: int = 60
    tvalue_hist_density: bool = True
    tvalue_2d_bins_y: int = 80
    tvalue_2d_y_min: Optional[float] = None
    tvalue_2d_y_max: Optional[float] = None

    # MRI activity overlay colors in GUI.
    # Positive T-values are shown in green, negative T-values in yellow.
    # A voxel with |T| == activity_color_t_ref reaches full color intensity.
    activity_color_t_ref: float = 1.0
    activity_overlay_max_alpha: float = 0.85


@dataclass
class OutputConfig:
    # Result folder creation policy.
    # "legacy": legacy structure results/<patient>/<contrast>, files may be overwritten.
    # "timestamp_hash": new structure results/<patient>/runs/<timestamp>_cfg_<hash>/<contrast>.
    run_folder_mode: str = "timestamp_hash"

    # User-defined run name added to the run folder name.
    # None/null or an empty string: do not add a manual name.
    run_name: Optional[str] = None

    # Add a short configuration hash to the run folder name.
    include_config_hash: bool = True

    # Length of the short hash in the folder name.
    config_hash_length: int = 8

    # What to do if the target folder already exists:
    # "error": stop with an error;
    # "overwrite": delete the old folder and write again;
    # "increment": add suffix _001, _002, ...
    existing_run_policy: str = "increment"

    # Save a copy of the actually used configuration to the run folder.
    save_settings_snapshot: bool = True

    # Save run_manifest.json with date, patient_id, hash, and paths.
    save_run_manifest: bool = True

    # After successful completion, update results/<patient>/latest_run.txt.
    update_latest_pointer: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_to_file: bool = True
    log_file: Path = Path("../results/liou_repro.log")

    def resolve_relative(self, base_dir: Path) -> None:
        self.log_file = _resolve_path(self.log_file, base_dir)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    design: DesignConfig = field(default_factory=DesignConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    empirical_bayes: EmpiricalBayesConfig = field(default_factory=EmpiricalBayesConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    spm_preprocessing: SPMPreprocessingConfig = field(default_factory=SPMPreprocessingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        config_path = Path(path).resolve()

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        cfg = cls.from_dict(raw)
        cfg.paths.resolve_relative(config_path.parent)
        cfg.logging.resolve_relative(config_path.parent)
        cfg.spm_preprocessing.resolve_relative(config_path.parent)
        return cfg

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "AppConfig":
        return cls(
            paths=PathsConfig(**raw.get("paths", {})),
            scanner=ScannerConfig(**raw.get("scanner", {})),
            data=DataConfig(**raw.get("data", {})),
            design=DesignConfig(**raw.get("design", {})),
            preprocessing=PreprocessingConfig(**raw.get("preprocessing", {})),
            empirical_bayes=EmpiricalBayesConfig(**raw.get("empirical_bayes", {})),
            analysis=AnalysisConfig(**raw.get("analysis", {})),
            output=OutputConfig(**raw.get("output", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
            spm_preprocessing=SPMPreprocessingConfig(**raw.get("spm_preprocessing", {})),
        )


def _resolve_path(path: str | Path, base_dir: Path, patient_id: Optional[str] = None) -> Path:
    text = str(path)

    if patient_id is not None and str(patient_id).strip() != "":
        try:
            text = text.format(patient_id=str(patient_id).strip())
        except Exception:
            pass

    p = Path(text)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def load_config(path: str | Path = "settings.yaml") -> AppConfig:
    return AppConfig.from_yaml(path)
