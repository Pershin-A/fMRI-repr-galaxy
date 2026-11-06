from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading

import nibabel as nib
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SPMPreprocessingConfig:
    """Settings for automatic MATLAB/SPM preprocessing from Python."""

    enabled: bool = True
    auto_run_if_missing: bool = True

    # Number of MATLAB/SPM threads/cores. If None, config.analysis.n_jobs is used.
    n_jobs: Optional[int] = None

    # Root folders.
    data_root: Optional[Path] = None
    raw_root: Optional[Path] = None
    preprocessed_root: Optional[Path] = None
    work_root: Optional[Path] = None
    spm_dir: Optional[Path] = None

    # MATLAB/SPM runner.
    matlab_executable: str = "matlab"
    matlab_script: Optional[Path] = None
    use_matlab_batch: bool = True

    # After successful preprocessing, switch analysis to data/preprocessed.
    update_config_paths: bool = True
    set_mask_path: bool = True

    # Empty required_runs means: derive run_id values from subject.json.
    required_runs: list[str] = field(default_factory=list)
    min_functional_images_per_run: int = 1

    # Files expected in preprocessed/<subject>.
    required_root_files: list[str] = field(default_factory=lambda: [
        "brain_mask.nii",
        "mean_functional.nii",
    ])
    required_structural_files: list[str] = field(default_factory=lambda: [
        "T1w.nii",
        "T1w_coreg_native.nii",
        "anatomy_funcgrid.nii",
        "brain_mask_anat_native.nii",
        "brain_mask_anat_funcgrid.nii",
    ])

    # The analysis mask is built from SPM c1+c2+c3 tissue probabilities on the
    # coregistered native-resolution T1, then resliced to the functional grid.
    expected_brain_mask_method: str = "t1_tissue_probability_sum_coregistered_to_functional"
    expected_preprocessing_signature: str = ""
    apply_slice_timing_in_spm: bool = False
    functional_target_voxel_size_mm: list[float] = field(default_factory=list)
    functional_reslice_interpolation: int = 4

    # Optional SPM smoothing applied after MNI normalisation and before export
    # of the run BOLD files used by Python. The mean functional/reference and
    # all masks remain unsmoothed.
    apply_spatial_smoothing: bool = False
    smoothing_fwhm_mm: list[float] = field(default_factory=lambda: [6.0, 6.0, 6.0])

    t1_tissue_threshold: float = 0.20
    intersect_functional_coverage: bool = True
    functional_coverage_relative_threshold: float = 0.02

    # When realigned BOLD outputs already exist, missing/stale T1-derived mask
    # artifacts can be repaired without rerunning the expensive realignment.
    allow_anatomy_mask_only_repair: bool = True

    # Sanity check for generated masks.
    # reslicing can cover almost the whole FOV; such output is treated as
    # incomplete so preprocessing is regenerated with the safer mask builder.
    max_mask_volume_fraction: float = 0.80
    min_mask_volume_fraction: float = 0.01

    def resolve_relative(self, base_dir: Path) -> None:
        for attr in (
            "data_root",
            "raw_root",
            "preprocessed_root",
            "work_root",
            "spm_dir",
            "matlab_script",
        ):
            value = getattr(self, attr, None)
            if value is None:
                continue
            path = Path(value)
            setattr(self, attr, path if path.is_absolute() else (base_dir / path).resolve())


class SPMPreprocessingManager:
    """Check/run MATLAB/SPM preprocessing for manifest-format and legacy subjects."""

    def __init__(self, config):
        self.config = config
        self.settings = self._read_settings(config)

        patient_id = getattr(config.paths, "patient_id", None)
        if patient_id is None or str(patient_id).strip() == "":
            self.patient_id = ""
        else:
            self.patient_id = str(patient_id).strip()

        self.data_root = self._resolve_data_root()
        self.raw_root = self._resolve_path(self.settings.raw_root, self.data_root / "raw")
        self.preprocessed_root = self._resolve_path(
            self.settings.preprocessed_root,
            self.data_root / "preprocessed",
        )
        self.work_root = self._resolve_path(
            self.settings.work_root,
            self.data_root / "preprocessed_spm_work",
        )

        configured_spm_dir = self._resolve_path(self.settings.spm_dir, self.data_root / "SPM")
        self.spm_dir = self._resolve_spm_dir_with_fallbacks(configured_spm_dir)

        self.matlab_script = self._resolve_path(
            self.settings.matlab_script,
            self.data_root / "spm_preprocess_project.m",
        )

        self.n_jobs = self._resolve_n_jobs()

    def ensure_preprocessed(self) -> None:
        if not bool(getattr(self.settings, "enabled", False)):
            logger.info("SPM preprocessing is disabled")
            return

        if not self.patient_id:
            raise ValueError("SPM preprocessing requires paths.patient_id")

        missing = self.missing_outputs()
        if missing:
            logger.info("SPM preprocessing outputs are missing/incomplete for %s:", self.patient_id)
            for item in missing[:30]:
                logger.info("  missing: %s", item)
            if len(missing) > 30:
                logger.info("  ... and %d more", len(missing) - 30)

            if not bool(getattr(self.settings, "auto_run_if_missing", False)):
                raise RuntimeError(
                    "SPM preprocessing outputs are missing and auto_run_if_missing is false."
                )

            self._ensure_matlab_inputs_ready()
            anatomy_mask_only = self._can_repair_anatomy_mask_only(missing)
            if anatomy_mask_only:
                logger.info(
                    "Only T1-derived anatomy/mask artifacts are missing or stale for %s; "
                    "reusing existing motion-corrected BOLD outputs.",
                    self.patient_id,
                )
            self._run_matlab(anatomy_mask_only=anatomy_mask_only)

            missing_after = self.missing_outputs()
            if missing_after:
                raise RuntimeError(
                    "SPM preprocessing finished, but required outputs are still missing:\n"
                    + "\n".join(str(x) for x in missing_after[:80])
                )
        else:
            logger.info("SPM preprocessing outputs already exist for %s", self.patient_id)

        spatial_problems = self._validate_spatial_alignment_outputs()
        if spatial_problems:
            raise RuntimeError(
                "SPM spatial alignment validation failed for "
                f"{self.patient_id}:\n" + "\n".join(spatial_problems)
            )

        if bool(getattr(self.settings, "update_config_paths", True)):
            self._update_config_paths_to_preprocessed()

    def _effective_preprocessing_signature(self) -> str:
        """Return a signature that changes whenever smoothing settings change."""
        base = str(
            getattr(self.settings, "expected_preprocessing_signature", "")
        ).strip()
        if not base:
            base = "spm_preprocessing"

        enabled = bool(getattr(self.settings, "apply_spatial_smoothing", False))
        if not enabled:
            return f"{base}_nosmooth"

        fwhm = list(getattr(self.settings, "smoothing_fwhm_mm", [6.0, 6.0, 6.0]) or [])
        if len(fwhm) != 3:
            raise ValueError(
                "spm_preprocessing.smoothing_fwhm_mm must contain three values"
            )
        values = [float(v) for v in fwhm]
        if any((not np.isfinite(v)) or v <= 0 for v in values):
            raise ValueError(
                "spm_preprocessing.smoothing_fwhm_mm values must be finite and > 0"
            )

        def token(value: float) -> str:
            text = f"{value:g}".replace(".", "p")
            return text.replace("-", "m")

        return f"{base}_smooth_{'x'.join(token(v) for v in values)}"

    def missing_outputs(self) -> list[str]:
        patient_dir = self.patient_preprocessed_dir
        raw_manifest = self.raw_subject_dir / "subject.json"
        pre_manifest = patient_dir / "subject.json"

        if raw_manifest.exists() or pre_manifest.exists():
            return self._missing_manifest_outputs(raw_manifest=raw_manifest, pre_manifest=pre_manifest)

        return self._missing_legacy_outputs()

    def _missing_manifest_outputs(self, raw_manifest: Path, pre_manifest: Path) -> list[str]:
        missing: list[str] = []
        patient_dir = self.patient_preprocessed_dir

        manifest_path = pre_manifest if pre_manifest.exists() else raw_manifest
        if not manifest_path.exists():
            return [str(raw_manifest)]

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return [f"Could not read manifest {manifest_path}: {exc}"]

        if not patient_dir.exists():
            missing.append(str(patient_dir))
            return missing

        if not pre_manifest.exists():
            missing.append(str(pre_manifest))

        run_ids = [str(r.get("run_id")) for r in manifest.get("runs", [])]
        if not run_ids:
            missing.append(f"No runs in manifest: {manifest_path}")

        required_runs = list(self.settings.required_runs) or run_ids
        for run_id in required_runs:
            run_dir = patient_dir / "runs" / run_id
            if not run_dir.exists():
                missing.append(str(run_dir))
                continue

            bold = self._choose_existing_nii(run_dir / "bold.nii")
            if bold is None:
                missing.append(str(run_dir / "bold.nii"))

        logs_dir = patient_dir / "logs"
        if not logs_dir.exists():
            missing.append(str(logs_dir))
        else:
            for run_id in run_ids:
                log_path = logs_dir / f"{run_id}.tsv"
                if not log_path.exists():
                    missing.append(str(log_path))

        structural_dir = patient_dir / "structural"
        if not structural_dir.exists():
            missing.append(str(structural_dir))
        else:
            for name in self.settings.required_structural_files:
                if not self._exists_nii_variant(structural_dir / name):
                    missing.append(str(structural_dir / name))

        for name in self.settings.required_root_files:
            candidate = self._choose_existing_nii(patient_dir / name)
            if candidate is None:
                missing.append(str(patient_dir / name))

        mask_path = self._choose_existing_nii(patient_dir / "brain_mask.nii")
        if mask_path is not None:
            mask_problem = self._validate_existing_mask_file(mask_path)
            if mask_problem is not None:
                missing.append(mask_problem)

        if pre_manifest.exists():
            expected_method = str(getattr(self.settings, "expected_brain_mask_method", "")).strip()
            if expected_method:
                generated = manifest.get("generated_files", {}) if isinstance(manifest, dict) else {}
                actual_method = str(generated.get("brain_mask_method", "")).strip()
                if actual_method != expected_method:
                    missing.append(
                        f"brain_mask_method mismatch: expected '{expected_method}', "
                        f"found '{actual_method or '<missing>'}' in {pre_manifest}"
                    )

            expected_signature = self._effective_preprocessing_signature()
            if expected_signature:
                preprocessing = manifest.get("preprocessing", {}) if isinstance(manifest, dict) else {}
                actual_signature = str(preprocessing.get("signature", "")).strip()
                if actual_signature != expected_signature:
                    missing.append(
                        f"preprocessing_signature mismatch: expected '{expected_signature}', "
                        f"found '{actual_signature or '<missing>'}' in {pre_manifest}"
                    )

        return missing

    def _missing_legacy_outputs(self) -> list[str]:
        missing: list[str] = []
        patient_dir = self.patient_preprocessed_dir
        if not patient_dir.exists():
            return [str(patient_dir)]

        for run_name in self.settings.required_runs:
            run_dir = patient_dir / run_name
            if not run_dir.exists():
                missing.append(str(run_dir))
                continue
            nii_files = sorted(run_dir.glob("*.nii")) + sorted(run_dir.glob("*.nii.gz"))
            if len(nii_files) < int(self.settings.min_functional_images_per_run):
                missing.append(
                    f"{run_dir}: found {len(nii_files)} NIfTI, "
                    f"minimum required {self.settings.min_functional_images_per_run}"
                )

        log_dir = patient_dir / "log"
        if not log_dir.exists():
            missing.append(str(log_dir))

        structural_dir = patient_dir / "structural"
        if not structural_dir.exists():
            missing.append(str(structural_dir))
        else:
            for name in self.settings.required_structural_files:
                if not self._exists_nii_variant(structural_dir / name):
                    missing.append(str(structural_dir / name))

        for name in self.settings.required_root_files:
            if not self._exists_nii_variant(patient_dir / name):
                missing.append(str(patient_dir / name))

        return missing

    def _can_repair_anatomy_mask_only(self, missing: list[str]) -> bool:
        if not bool(getattr(self.settings, "allow_anatomy_mask_only_repair", True)):
            return False
        patient_dir = self.patient_preprocessed_dir
        structural_dir = patient_dir / "structural"

        # Only anatomy/mask artifacts and the mask-method marker may be missing.
        # A preprocessing signature mismatch must trigger a complete rebuild.
        for item in missing:
            item_text = str(item)
            allowed = (
                item_text.startswith(str(structural_dir))
                or "brain_mask" in item_text
                or "brain_mask_method mismatch" in item_text
            )
            if not allowed:
                return False

        manifest_path = patient_dir / "subject.json"
        if not manifest_path.exists() or not self._exists_nii_variant(patient_dir / "mean_functional.nii"):
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return False

        runs = manifest.get("runs", []) if isinstance(manifest, dict) else []
        if not runs:
            return False
        for run in runs:
            run_id = str(run.get("run_id", "")).strip()
            if not run_id:
                return False
            run_dir = patient_dir / "runs" / run_id
            if self._choose_existing_nii(run_dir / "bold.nii") is None:
                return False
            if not (run_dir / "bold.json").exists():
                return False
            if not (run_dir / "rp_bold.txt").exists():
                return False
            if not (patient_dir / "logs" / f"{run_id}.tsv").exists():
                return False

        raw_t1 = self._choose_existing_nii(self.raw_subject_dir / "structural" / "T1w.nii")
        return raw_t1 is not None

    @property
    def raw_subject_dir(self) -> Path:
        return self.raw_root / self.patient_id

    @property
    def patient_preprocessed_dir(self) -> Path:
        return self.preprocessed_root / self.patient_id

    def _ensure_matlab_inputs_ready(self) -> None:
        if not self.raw_subject_dir.exists():
            raise FileNotFoundError(f"Raw subject folder does not exist: {self.raw_subject_dir}")
        if not self.matlab_script.exists():
            raise FileNotFoundError(
                f"MATLAB preprocessing script was not found: {self.matlab_script}. "
                "Place spm_preprocess_project.m in data/ or set spm_preprocessing.matlab_script."
            )
        if not self._spm_dir_has_spm(self.spm_dir):
            raise FileNotFoundError(
                f"SPM folder does not contain spm.m: {self.spm_dir}. "
                "Set spm_preprocessing.spm_dir or install SPM in Documents/MATLAB/spm."
            )

    def _run_matlab(self, anatomy_mask_only: bool = False) -> None:
        self.work_root.mkdir(parents=True, exist_ok=True)
        subject_work = self.work_root / self.patient_id
        subject_work.mkdir(parents=True, exist_ok=True)

        runner_path = subject_work / f"run_spm_preprocess_{self.patient_id}.m"
        diary_path = subject_work / f"run_spm_preprocess_{self.patient_id}.diary.txt"
        stdout_path = subject_work / f"run_spm_preprocess_{self.patient_id}.stdout.log"
        stderr_path = subject_work / f"run_spm_preprocess_{self.patient_id}.stderr.log"
        command_path = subject_work / f"run_spm_preprocess_{self.patient_id}.command.txt"

        runner_path.write_text(
            self._make_matlab_runner(diary_path, anatomy_mask_only=anatomy_mask_only),
            encoding="utf-8",
        )

        if self.settings.use_matlab_batch:
            cmd = [
                str(self.settings.matlab_executable),
                "-batch",
                f"run('{self._matlab_path(runner_path)}')",
            ]
        else:
            cmd = [
                str(self.settings.matlab_executable),
                "-nosplash",
                "-nodesktop",
                "-r",
                (
                    f"try, run('{self._matlab_path(runner_path)}'); "
                    "catch ME, disp(getReport(ME,'extended')); exit(1); "
                    "end; exit(0);"
                ),
            ]

        command_path.write_text(
            " ".join(cmd)
            + "\n\n"
            + f"cwd={self.data_root}\n"
            + f"runner={runner_path}\n"
            + f"stdout={stdout_path}\n"
            + f"stderr={stderr_path}\n"
            + f"diary={diary_path}\n"
            + f"anatomy_mask_only={bool(anatomy_mask_only)}\n",
            encoding="utf-8",
        )

        logger.info("Starting MATLAB/SPM preprocessing: %s", " ".join(cmd))
        logger.info("MATLAB runner saved: %s", runner_path)
        logger.info("MATLAB command saved: %s", command_path)
        logger.info("MATLAB diary will be saved: %s", diary_path)
        logger.info("MATLAB/SPM n_jobs=%d", self.n_jobs)
        logger.info("MATLAB/SPM anatomy_mask_only=%s", bool(anatomy_mask_only))
        logger.info("SPM dir: %s", self.spm_dir)
        logger.info("Raw subject dir: %s", self.raw_subject_dir)

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(self.n_jobs)
        env["MKL_NUM_THREADS"] = str(self.n_jobs)
        env["OPENBLAS_NUM_THREADS"] = str(self.n_jobs)
        env["NUMEXPR_NUM_THREADS"] = str(self.n_jobs)

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.data_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Could not start MATLAB executable '{self.settings.matlab_executable}'. "
                "Set the full path in spm_preprocessing.matlab_executable or add MATLAB to PATH."
            ) from exc

        stdout_thread = threading.Thread(
            target=self._stream_process_pipe,
            args=(proc.stdout, stdout_lines, "MATLAB stdout"),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=self._stream_process_pipe,
            args=(proc.stderr, stderr_lines, "MATLAB stderr"),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        try:
            returncode = proc.wait()
            stdout_thread.join()
            stderr_thread.join()
        except KeyboardInterrupt:
            logger.warning("Ctrl+C received. Stopping MATLAB/SPM preprocessing...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("MATLAB did not exit after terminate(); killing it.")
                proc.kill()
                proc.wait()
            stdout_thread.join(timeout=2)
            stderr_thread.join(timeout=2)
            stdout_path.write_text("".join(stdout_lines), encoding="utf-8")
            stderr_path.write_text("".join(stderr_lines), encoding="utf-8")
            raise

        stdout_text = "".join(stdout_lines)
        stderr_text = "".join(stderr_lines)
        stdout_path.write_text(stdout_text, encoding="utf-8")
        stderr_path.write_text(stderr_text, encoding="utf-8")

        logger.info("MATLAB stdout saved: %s", stdout_path)
        if stderr_text.strip():
            logger.warning("MATLAB stderr saved: %s", stderr_path)
        else:
            logger.info("MATLAB stderr saved: %s", stderr_path)

        if returncode != 0:
            diary_text = self._read_text_if_exists(diary_path)
            combined = "\n".join(
                part
                for part in [
                    "===== STDERR =====\n" + stderr_text if stderr_text.strip() else "",
                    "===== STDOUT =====\n" + stdout_text if stdout_text.strip() else "",
                    "===== DIARY =====\n" + diary_text if diary_text.strip() else "",
                ]
                if part.strip()
            )
            raise RuntimeError(
                f"MATLAB/SPM preprocessing finished with code {returncode}.\n"
                f"runner: {runner_path}\n"
                f"stdout: {stdout_path}\n"
                f"stderr: {stderr_path}\n"
                f"diary: {diary_path}\n"
                f"command: {command_path}\n"
                f"Last lines:\n{self._tail_text(combined, n_lines=140)}"
            )

        logger.info("MATLAB/SPM preprocessing finished successfully")

    def _make_matlab_runner(self, diary_path: Path, anatomy_mask_only: bool = False) -> str:
        matlab_bool = "true" if anatomy_mask_only else "false"
        t1_threshold = float(getattr(self.settings, "t1_tissue_threshold", 0.20))
        intersect_bool = "true" if bool(getattr(self.settings, "intersect_functional_coverage", True)) else "false"
        coverage_threshold = float(
            getattr(self.settings, "functional_coverage_relative_threshold", 0.02)
        )
        apply_stc_bool = (
            "true" if bool(getattr(self.settings, "apply_slice_timing_in_spm", False)) else "false"
        )
        target_voxels = list(getattr(self.settings, "functional_target_voxel_size_mm", []) or [])
        if target_voxels and len(target_voxels) != 3:
            raise ValueError(
                "spm_preprocessing.functional_target_voxel_size_mm must contain three values"
            )
        if target_voxels:
            target_voxel_matlab = "[" + " ".join(f"{float(v):.12g}" for v in target_voxels) + "]"
        else:
            target_voxel_matlab = "[]"
        reslice_interp = int(getattr(self.settings, "functional_reslice_interpolation", 4))
        apply_smoothing_bool = (
            "true" if bool(getattr(self.settings, "apply_spatial_smoothing", False)) else "false"
        )
        smoothing_fwhm = list(
            getattr(self.settings, "smoothing_fwhm_mm", [6.0, 6.0, 6.0]) or []
        )
        if len(smoothing_fwhm) != 3:
            raise ValueError(
                "spm_preprocessing.smoothing_fwhm_mm must contain three values"
            )
        smoothing_values = [float(v) for v in smoothing_fwhm]
        if any((not np.isfinite(v)) or v <= 0 for v in smoothing_values):
            raise ValueError(
                "spm_preprocessing.smoothing_fwhm_mm values must be finite and > 0"
            )
        smoothing_fwhm_matlab = "[" + " ".join(
            f"{value:.12g}" for value in smoothing_values
        ) + "]"
        preprocessing_signature = self._effective_preprocessing_signature()
        return f"""
try
    diary('{self._matlab_path(diary_path)}');
    diary on;

    fprintf('=== Python generated SPM runner ===\\n');
    fprintf('Subject: {self._matlab_quote(self.patient_id)}\\n');
    fprintf('Data root: {self._matlab_path(self.data_root)}\\n');
    fprintf('Raw root: {self._matlab_path(self.raw_root)}\\n');
    fprintf('Work root: {self._matlab_path(self.work_root)}\\n');
    fprintf('Python/preprocessed root: {self._matlab_path(self.preprocessed_root)}\\n');
    fprintf('SPM dir: {self._matlab_path(self.spm_dir)}\\n');
    fprintf('MATLAB pwd: %s\\n', pwd);
    fprintf('MATLAB version: %s\\n', version);

    if exist('{self._matlab_path(self.spm_dir)}', 'dir') ~= 7
        error('SPM directory does not exist: {self._matlab_path(self.spm_dir)}');
    end
    if exist('{self._matlab_path(self.raw_subject_dir)}', 'dir') ~= 7
        error('Raw subject folder does not exist: {self._matlab_path(self.raw_subject_dir)}');
    end
    if exist('{self._matlab_path(self.matlab_script.parent)}', 'dir') ~= 7
        error('MATLAB script folder does not exist: {self._matlab_path(self.matlab_script.parent)}');
    end

    addpath('{self._matlab_path(self.spm_dir)}');
    addpath('{self._matlab_path(self.matlab_script.parent)}');

    fprintf('which spm -all:\\n');
    which spm -all;
    fprintf('which spm_preprocess_project -all:\\n');
    which spm_preprocess_project -all;

    if isempty(which('spm'))
        error('SPM function spm.m is not on MATLAB path after addpath: {self._matlab_path(self.spm_dir)}');
    end
    if isempty(which('spm_preprocess_project'))
        error('spm_preprocess_project.m is not on MATLAB path after addpath: {self._matlab_path(self.matlab_script.parent)}');
    end

    spm_preprocess_project( ...
        'data_root', '{self._matlab_path(self.data_root)}', ...
        'raw_root', '{self._matlab_path(self.raw_root)}', ...
        'work_root', '{self._matlab_path(self.work_root)}', ...
        'python_root', '{self._matlab_path(self.preprocessed_root)}', ...
        'spm_dir', '{self._matlab_path(self.spm_dir)}', ...
        'n_jobs', {int(self.n_jobs)}, ...
        'anatomy_mask_only', {matlab_bool}, ...
        'preprocessing_signature', '{self._matlab_quote(preprocessing_signature)}', ...
        'apply_slice_timing_in_spm', {apply_stc_bool}, ...
        'functional_target_voxel_size_mm', {target_voxel_matlab}, ...
        'functional_reslice_interpolation', {reslice_interp}, ...
        'apply_spatial_smoothing', {apply_smoothing_bool}, ...
        'smoothing_fwhm_mm', {smoothing_fwhm_matlab}, ...
        't1_tissue_threshold', {t1_threshold:.12g}, ...
        'intersect_functional_coverage', {intersect_bool}, ...
        'functional_coverage_relative_threshold', {coverage_threshold:.12g}, ...
        'subjects', {{'{self._matlab_quote(self.patient_id)}'}} ...
    );

    fprintf('=== SPM preprocessing finished successfully ===\\n');
    diary off;
    exit(0);
catch ME
    fprintf(2, '\\n=== MATLAB/SPM ERROR ===\\n');
    fprintf(2, '%s\\n', getReport(ME, 'extended', 'hyperlinks', 'off'));
    try
        diary off;
    catch
    end
    exit(1);
end
""".strip() + "\n"

    @staticmethod
    def _nifti_grid_record(path: Path) -> dict:
        img = nib.load(str(path))
        affine = np.asarray(img.affine, dtype=float)
        return {
            "path": str(path),
            "shape": [int(v) for v in img.shape[:3]],
            "zooms_mm": [float(v) for v in nib.affines.voxel_sizes(affine)],
            "affine": affine.tolist(),
        }

    @staticmethod
    def _same_spatial_grid(a: nib.spatialimages.SpatialImage, b: nib.spatialimages.SpatialImage, atol: float = 1e-4) -> tuple[bool, float]:
        same_shape = tuple(int(v) for v in a.shape[:3]) == tuple(int(v) for v in b.shape[:3])
        max_affine_difference = float(np.max(np.abs(np.asarray(a.affine) - np.asarray(b.affine))))
        return bool(same_shape and max_affine_difference <= float(atol)), max_affine_difference

    def _validate_spatial_alignment_outputs(self) -> list[str]:
        """Validate that every analysis image shares exactly one functional grid.

        The previous GUI failure was caused by valid-looking files living in
        different spatial coordinate systems.  This validation is run before
        the data loader and writes a compact report beside subject.json.
        """
        patient_dir = self.patient_preprocessed_dir
        problems: list[str] = []
        report: dict = {
            "schema_version": "1.0",
            "patient_id": self.patient_id,
            "status": "unknown",
            "checks": [],
            "images": {},
        }

        def existing(base: Path) -> Optional[Path]:
            if base.exists():
                return base
            gz = Path(str(base) + ".gz")
            return gz if gz.exists() else None

        mean_path = existing(patient_dir / "mean_functional.nii")
        mask_path = existing(patient_dir / "brain_mask.nii")
        reference_path = existing(patient_dir / "functional_reference.nii")
        if mean_path is None:
            return [f"Missing mean functional: {patient_dir / 'mean_functional.nii'}"]
        if mask_path is None:
            return [f"Missing brain mask: {patient_dir / 'brain_mask.nii'}"]

        mean_img = nib.load(str(mean_path))
        mask_img = nib.load(str(mask_path))
        report["images"]["mean_functional"] = self._nifti_grid_record(mean_path)
        report["images"]["brain_mask"] = self._nifti_grid_record(mask_path)

        def check_grid(label: str, candidate_path: Optional[Path], required: bool = False) -> Optional[nib.spatialimages.SpatialImage]:
            if candidate_path is None:
                if required:
                    problems.append(f"Missing required image for spatial check: {label}")
                return None
            try:
                img = nib.load(str(candidate_path))
                same, max_diff = self._same_spatial_grid(mean_img, img)
                report["images"][label] = self._nifti_grid_record(candidate_path)
                report["checks"].append({
                    "label": label,
                    "same_grid_as_mean": bool(same),
                    "max_affine_difference": max_diff,
                })
                if not same:
                    problems.append(
                        f"{label} does not share the mean-functional grid: "
                        f"shape={img.shape[:3]} vs {mean_img.shape[:3]}, "
                        f"max affine difference={max_diff:.9g}; file={candidate_path}"
                    )
                return img
            except Exception as exc:
                problems.append(f"Could not validate {label} ({candidate_path}): {exc}")
                return None

        check_grid("brain_mask", mask_path, required=True)
        if reference_path is not None:
            check_grid("functional_reference", reference_path)

        structural_dir = patient_dir / "structural"
        anatomy_funcgrid_path = existing(structural_dir / "anatomy_funcgrid.nii")
        anat_mask_funcgrid_path = existing(structural_dir / "brain_mask_anat_funcgrid.nii")
        anatomy_func_img = check_grid("anatomy_funcgrid", anatomy_funcgrid_path)
        anat_mask_img = check_grid("brain_mask_anat_funcgrid", anat_mask_funcgrid_path)

        manifest_path = patient_dir / "subject.json"
        manifest: dict = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                problems.append(f"Could not read preprocessed manifest for spatial validation: {exc}")

        run_ids = [str(item.get("run_id")) for item in manifest.get("runs", []) if item.get("run_id")]
        for run_id in run_ids:
            run_path = existing(patient_dir / "runs" / run_id / "bold.nii")
            check_grid(f"run:{run_id}", run_path, required=True)

        try:
            mask = np.asarray(mask_img.dataobj) > 0
            report["brain_mask_voxels"] = int(mask.sum())
            report["brain_mask_fraction"] = float(mask.mean())
            if not np.any(mask):
                problems.append("brain_mask.nii is empty")
            if anat_mask_img is not None and tuple(anat_mask_img.shape[:3]) == tuple(mask.shape):
                anat_mask = np.asarray(anat_mask_img.dataobj) > 0
                outside = int(np.count_nonzero(mask & ~anat_mask))
                report["brain_mask_voxels_outside_anatomical_mask"] = outside
                if outside != 0:
                    problems.append(
                        f"Final functional mask contains {outside} voxels outside "
                        "brain_mask_anat_funcgrid.nii"
                    )
        except Exception as exc:
            problems.append(f"Could not inspect mask containment: {exc}")

        anatomy_mni_path = existing(structural_dir / "anatomy_mni_1mm.nii")
        mask_mni_path = existing(structural_dir / "brain_mask_anat_mni_1mm.nii")
        if anatomy_mni_path is not None or mask_mni_path is not None:
            if anatomy_mni_path is None or mask_mni_path is None:
                problems.append("Only one of anatomy_mni_1mm / brain_mask_anat_mni_1mm exists")
            else:
                try:
                    a = nib.load(str(anatomy_mni_path))
                    m = nib.load(str(mask_mni_path))
                    same, max_diff = self._same_spatial_grid(a, m)
                    report["images"]["anatomy_mni_1mm"] = self._nifti_grid_record(anatomy_mni_path)
                    report["images"]["brain_mask_anat_mni_1mm"] = self._nifti_grid_record(mask_mni_path)
                    report["checks"].append({
                        "label": "highres_mni_anatomy_mask",
                        "same_grid": bool(same),
                        "max_affine_difference": max_diff,
                    })
                    if not same:
                        problems.append(
                            "anatomy_mni_1mm and brain_mask_anat_mni_1mm do not share one grid"
                        )
                except Exception as exc:
                    problems.append(f"Could not validate high-resolution MNI anatomy/mask: {exc}")

        preprocessing = manifest.get("preprocessing", {}) if isinstance(manifest, dict) else {}
        report["preprocessing_space"] = preprocessing.get("space")
        report["preprocessing_signature"] = preprocessing.get("signature")
        report["status"] = "passed" if not problems else "failed"
        report["problems"] = problems
        qc_path = patient_dir / "spatial_qc.json"
        try:
            qc_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("Spatial QC report saved: %s", qc_path)
        except Exception as exc:
            problems.append(f"Could not write spatial QC report: {exc}")

        if not problems:
            logger.info(
                "Spatial alignment validation passed: shape=%s, mask_voxels=%d",
                tuple(int(v) for v in mean_img.shape[:3]),
                int(np.asarray(mask_img.dataobj).astype(bool).sum()),
            )
        return problems

    def _validate_existing_mask_file(self, mask_path: Path) -> Optional[str]:
        try:
            img = nib.load(str(mask_path))
            data = np.asarray(img.dataobj)
            mask = np.asarray(data != 0, dtype=bool)
            n_mask = int(mask.sum())
            n_total = int(mask.size)
            fraction = n_mask / float(n_total) if n_total > 0 else 0.0
        except Exception as exc:
            return f"Could not read generated mask {mask_path}: {exc}"

        min_fraction = float(getattr(self.settings, "min_mask_volume_fraction", 0.01))
        max_fraction = float(getattr(self.settings, "max_mask_volume_fraction", 0.80))

        if fraction < min_fraction or fraction > max_fraction:
            return (
                f"Generated brain_mask.nii has implausible volume fraction "
                f"{fraction:.6f} ({n_mask}/{n_total}); expected between "
                f"{min_fraction:.6f} and {max_fraction:.6f}: {mask_path}"
            )

        return None

    def _update_config_paths_to_preprocessed(self) -> None:
        patient_dir = self.patient_preprocessed_dir
        mask_path = self._choose_existing_nii(patient_dir / "brain_mask.nii")

        logger.info("Switching config.paths.data_path to preprocessed_root: %s", self.preprocessed_root)
        self.config.paths.data_path = self.preprocessed_root
        self.config.paths.logs_path = self.preprocessed_root

        if bool(getattr(self.settings, "set_mask_path", True)) and mask_path is not None:
            mask_problem = self._validate_existing_mask_file(mask_path)
            if mask_problem is None:
                logger.info("Setting config.paths.mask_path: %s", mask_path)
                self.config.paths.mask_path = mask_path
            else:
                logger.warning(
                    "Not setting config.paths.mask_path because the generated mask failed sanity checks: %s",
                    mask_problem,
                )
                self.config.paths.mask_path = None

    def _resolve_n_jobs(self) -> int:
        value = getattr(self.settings, "n_jobs", None)
        if value is None:
            analysis_cfg = getattr(self.config, "analysis", None)
            value = getattr(analysis_cfg, "n_jobs", 1)
        try:
            n_jobs = int(value)
        except Exception:
            n_jobs = 1
        return max(1, n_jobs)

    @staticmethod
    def _read_settings(config) -> SPMPreprocessingConfig:
        raw = getattr(config, "spm_preprocessing", None)
        if raw is None:
            return SPMPreprocessingConfig(enabled=False)
        if isinstance(raw, SPMPreprocessingConfig):
            return raw
        known = {f.name for f in SPMPreprocessingConfig.__dataclass_fields__.values()}
        values = {name: getattr(raw, name) for name in known if hasattr(raw, name)}
        return SPMPreprocessingConfig(**values)

    @staticmethod
    def _resolve_path(value: Optional[Path], default: Path) -> Path:
        if value is None or str(value).strip() == "":
            return default.expanduser().resolve()
        return Path(value).expanduser().resolve()

    def _resolve_data_root(self) -> Path:
        if self.settings.data_root is not None:
            return Path(self.settings.data_root).expanduser().resolve()
        data_path = Path(self.config.paths.data_path).expanduser().resolve()
        if data_path.name.lower() in {"preprocessed", "raw"}:
            return data_path.parent.resolve()
        if self.patient_id and data_path.name == self.patient_id and data_path.parent.name.lower() in {"preprocessed", "raw"}:
            return data_path.parent.parent.resolve()
        return data_path.resolve()

    def _candidate_spm_dirs(self, configured_spm_dir: Path) -> list[Path]:
        candidates: list[Path] = []
        def add(path: Path) -> None:
            path = Path(path).expanduser()
            if path not in candidates:
                candidates.append(path)
        add(configured_spm_dir)
        home = Path.home()
        add(home / "Documents" / "MATLAB" / "spm")
        add(home / "Documents" / "MATLAB" / "spm12")
        add(self.data_root / "SPM")
        add(self.data_root / "spm")
        return candidates

    @staticmethod
    def _spm_dir_has_spm(path: Path) -> bool:
        return path.exists() and path.is_dir() and (path / "spm.m").exists()

    def _resolve_spm_dir_with_fallbacks(self, configured_spm_dir: Path) -> Path:
        for candidate in self._candidate_spm_dirs(configured_spm_dir):
            if self._spm_dir_has_spm(candidate):
                if candidate.resolve() != configured_spm_dir.resolve():
                    logger.warning(
                        "SPM folder from settings.yaml was not found or does not contain spm.m: %s. "
                        "Using the found fallback: %s",
                        configured_spm_dir,
                        candidate.resolve(),
                    )
                return candidate.resolve()
        return configured_spm_dir.resolve()

    @staticmethod
    def _exists_nii_variant(path: Path) -> bool:
        path = Path(path)
        if path.exists():
            return True
        if str(path).endswith(".nii") and Path(str(path) + ".gz").exists():
            return True
        return False

    @staticmethod
    def _choose_existing_nii(path: Path) -> Optional[Path]:
        path = Path(path)
        if path.exists():
            return path
        if str(path).endswith(".nii"):
            gz = Path(str(path) + ".gz")
            if gz.exists():
                return gz
        return None

    @staticmethod
    def _stream_process_pipe(pipe, lines: list[str], prefix: str) -> None:
        if pipe is None:
            return
        try:
            for line in iter(pipe.readline, ""):
                lines.append(line)
                msg = line.rstrip("\r\n")
                if msg:
                    if "stderr" in prefix.lower():
                        logger.warning("[%s] %s", prefix, msg)
                    else:
                        logger.info("[%s] %s", prefix, msg)
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    @staticmethod
    def _tail_text(text: str, n_lines: int = 80) -> str:
        lines = str(text).splitlines()
        return "\n".join(lines[-int(n_lines):]) if lines else ""

    @staticmethod
    def _read_text_if_exists(path: Path) -> str:
        try:
            if Path(path).exists():
                return Path(path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""
        return ""

    @staticmethod
    def _matlab_quote(value: str) -> str:
        return str(value).replace("'", "''")

    def _matlab_path(self, path: Path) -> str:
        return self._matlab_quote(Path(path).resolve().as_posix())
