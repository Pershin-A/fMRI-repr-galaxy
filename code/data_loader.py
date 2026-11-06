from __future__ import annotations

import json
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import nibabel as nib
from nibabel.affines import apply_affine
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from nibabel.processing import resample_from_to

from config import AppConfig
from data_types import BrainMask, PatientData, RunData

logger = logging.getLogger(__name__)


class ProjectDataLoader:
    """
    Implementation detail.

    Implementation detail.

    data/
    └── M_Y3_003/
        ├── fr_1/
        ├── fr_2/
        ├── fr_3/
        ├── me_1/
        ├── me_2/
        ├── me_3/
        ├── log/
        └── structural/

    Implementation detail.
    Implementation detail.
    Implementation detail.
    Implementation detail.
    Implementation detail.
    """

    def __init__(self, config: AppConfig):
        self.config = config

    def load_patient(self) -> PatientData:
        patient_data_path = self._resolve_patient_path(self.config.paths.data_path)

        manifest_path = patient_data_path / "subject.json"
        if manifest_path.exists():
            return self.load_patient_from_manifest(
                patient_data_path=patient_data_path,
                manifest_path=manifest_path,
            )

        patient_logs_path = self._resolve_logs_path(patient_data_path)
        patient_id = self.config.paths.patient_id or patient_data_path.name

        runs = self.load_patient_runs(patient_data_path)
        logs = self.load_patient_logs(patient_logs_path)

        if self.config.data.enable_common_stimulus_filter:
            logs, runs = self.apply_common_stimulus_filter(logs=logs, runs=runs)

        logger.info(
            "Patient %s: runs loaded: %d, logs loaded: %d",
            patient_id,
            len(runs),
            len(logs),
        )

        for run_name, run_data in runs.items():
            logger.info("Run %s: data shape = %s", run_name, run_data.data.shape)

        return PatientData(
            patient_id=patient_id,
            runs=runs,
            logs=logs,
            run_order=sorted(runs.keys()),
            manifest=None,
        )

    def load_patient_from_manifest(
        self,
        patient_data_path: Path,
        manifest_path: Path,
    ) -> PatientData:
        """
        Load a subject in the new strict manifest-based raw-data format.

        Expected structure:

            data/raw/<subject_id>/
                subject.json
                runs/<run_id>/bold.nii.gz
                runs/<run_id>/bold.json
                logs/<run_id>.tsv
                structural/T1w.nii.gz

        The manifest controls the run order, X_star, condition order and
        contrast definitions. No run-name guessing is used in this mode.
        """
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest: Dict[str, Any] = json.load(f)

        patient_id = str(manifest.get("subject_id") or patient_data_path.name)
        run_specs = list(manifest.get("runs", []))

        if not run_specs:
            raise ValueError(f"Manifest contains no runs: {manifest_path}")

        run_order = [str(run["run_id"]) for run in run_specs]
        if len(run_order) != len(set(run_order)):
            raise ValueError(f"Duplicate run_id values in manifest: {manifest_path}")

        runs: Dict[str, RunData] = {}
        logs: Dict[str, pd.DataFrame] = {}

        log_delimiter = getattr(self.config.data, "log_delimiter", "\t")

        for run in run_specs:
            run_id = str(run["run_id"])
            bold_path = patient_data_path / str(run["bold"])
            log_path = patient_data_path / str(run["log"])

            if not bold_path.exists():
                raise FileNotFoundError(f"Manifest BOLD file does not exist: {bold_path}")

            if not log_path.exists():
                raise FileNotFoundError(f"Manifest log/events file does not exist: {log_path}")

            runs[run_id] = self.load_manifest_run_data(
                run_id=run_id,
                bold_path=bold_path,
                run_manifest=run,
            )
            logs[run_id] = pd.read_csv(log_path, sep=log_delimiter)

        self._apply_manifest_to_config(
            manifest=manifest,
            runs=runs,
            patient_data_path=patient_data_path,
        )

        logger.info(
            "Patient %s loaded from manifest %s: runs=%d",
            patient_id,
            manifest_path,
            len(runs),
        )
        logger.info("Manifest run order: %s", run_order)
        logger.info(
            "Scanner updated from manifest: shape=(%d, %d, %d), nT=%d, nR=%d, TR=%.6g",
            self.config.scanner.nX,
            self.config.scanner.nY,
            self.config.scanner.nZ,
            self.config.scanner.nT,
            self.config.scanner.nR,
            self.config.scanner.TR,
        )

        return PatientData(
            patient_id=patient_id,
            runs=runs,
            logs=logs,
            run_order=run_order,
            manifest=manifest,
        )

    def load_manifest_run_data(
        self,
        run_id: str,
        bold_path: Path,
        run_manifest: Dict[str, Any],
    ) -> RunData:
        img = nib.load(str(bold_path))
        data = np.asanyarray(img.dataobj)

        if data.dtype != np.float32:
            data = data.astype(np.float32, copy=False)

        if data.ndim == 4:
            # NIfTI is stored as X × Y × Z × T, while the pipeline expects
            # T × X × Y × Z.
            data = np.moveaxis(data, -1, 0)
        elif data.ndim == 3:
            data = data[np.newaxis, ...]
        else:
            raise ValueError(f"Unexpected BOLD dimensionality {bold_path}: {data.shape}")

        slice_timing = self._slice_timing_from_run_manifest(run_manifest)

        return RunData(
            name=run_id,
            data=data.astype(np.float32, copy=False),
            affine=img.affine,
            header=img.header.copy(),
            source_files=[bold_path],
            metadata=dict(run_manifest),
            slice_timing=slice_timing,
        )

    def _apply_manifest_to_config(
        self,
        manifest: Dict[str, Any],
        runs: Dict[str, RunData],
        patient_data_path: Path,
    ) -> None:
        run_order = [str(run["run_id"]) for run in manifest.get("runs", [])]
        first_run = runs[run_order[0]]

        nT, nX, nY, nZ = first_run.data.shape

        self.config.paths.patient_id = str(manifest.get("subject_id") or patient_data_path.name)
        self.config.scanner.nX = int(nX)
        self.config.scanner.nY = int(nY)
        self.config.scanner.nZ = int(nZ)
        self.config.scanner.nT = int(nT)
        self.config.scanner.nR = int(len(run_order))

        first_run_manifest = manifest["runs"][0]
        if "tr_sec" in first_run_manifest:
            self.config.scanner.TR = float(first_run_manifest["tr_sec"])

        self._apply_manifest_slice_timing(manifest=manifest)
        self._load_manifest_motion_regressors(
            manifest=manifest,
            patient_data_path=patient_data_path,
            expected_n_timepoints=nT,
        )

        events_cfg = manifest.get("events", {})
        self.config.data.log_onset_column = str(
            events_cfg.get("onset_column", self.config.data.log_onset_column)
        )
        self.config.data.log_duration_column = str(
            events_cfg.get("duration_column", getattr(self.config.data, "log_duration_column", "duration"))
        )
        self.config.data.log_condition_column = str(
            events_cfg.get("condition_column", getattr(self.config.data, "log_condition_column", "trial_type"))
        )
        self.config.data.log_stimulus_column = str(
            events_cfg.get("stimulus_column", self.config.data.log_stimulus_column)
        )

        first_level_design = manifest.get("first_level_design", {})
        conditions = list(first_level_design.get("conditions", []))
        self.config.data.condition_order = [str(c) for c in conditions]
        if conditions:
            self.config.scanner.nS = int(len(conditions))

        design_type = str(first_level_design.get("type", "")).lower().strip()
        if design_type in {"condition_level_glm", "condition_regressors"}:
            self.config.design.design_type = "condition_regressors"

        if "include_intercept" in first_level_design:
            self.config.design.add_intercept = bool(first_level_design["include_intercept"])

        if "include_drift_terms" in first_level_design:
            self.config.design.add_drift_terms = bool(first_level_design["include_drift_terms"])

        # Common-stimulus filtering is a legacy preprocessing step for M_Y3_*.
        # Manifest-format data already contain complete BIDS events for each run.
        self.config.data.enable_common_stimulus_filter = False

        if "first_level_contrasts" in manifest:
            self.config.analysis.first_level_contrasts = dict(manifest["first_level_contrasts"])

        x_star = manifest.get("x_star", {})
        if isinstance(x_star, dict) and "values" in x_star:
            values = x_star["values"]
            # MATLAB may flatten a single-column JSON matrix in the
            # preprocessed manifest. Convert [1, 1, 1, 1] back to
            # [[1], [1], [1], [1]] for consistency with the raw manifest.
            if isinstance(values, list) and values and all(
                not isinstance(row, (list, tuple, dict)) for row in values
            ):
                values = [[row] for row in values]
            self.config.analysis.x_star = values

        if "second_level_contrasts" in manifest:
            self.config.analysis.second_level_contrasts = dict(manifest["second_level_contrasts"])

        if "primary_second_level_contrast" in manifest:
            self.config.analysis.primary_second_level_contrast = str(
                manifest["primary_second_level_contrast"]
            )

    @staticmethod
    def _slice_timing_from_run_manifest(run_manifest: Dict[str, Any]) -> Optional[List[float]]:
        st = run_manifest.get("slice_timing", {})
        if isinstance(st, dict) and st.get("available") and st.get("values") is not None:
            return [float(v) for v in st.get("values", [])]
        return None

    def _apply_manifest_slice_timing(self, manifest: Dict[str, Any]) -> None:
        """Resolve SliceTiming by the canonical runs[].run_id values.

        MATLAB jsondecode/jsonencode converts JSON object keys containing '-'
        into valid MATLAB field names with '_'. Therefore the manifest-level
        slice_timings.by_run dictionary is not trusted when per-run values are
        available. The per-run blocks preserve run_id and are authoritative.
        """
        by_run: Dict[str, List[float]] = {}

        for run in manifest.get("runs", []):
            run_id = str(run.get("run_id"))
            values = self._slice_timing_from_run_manifest(run)
            if values is not None:
                by_run[run_id] = values

        if not by_run:
            st_summary = manifest.get("slice_timings", {})
            raw_by_run = st_summary.get("by_run", {}) if isinstance(st_summary, dict) else {}
            run_ids = [str(run.get("run_id")) for run in manifest.get("runs", [])]
            for run_id in run_ids:
                candidates = [
                    run_id,
                    run_id.replace("-", "_"),
                    run_id.replace("_", "-"),
                ]
                values = None
                for candidate in candidates:
                    if isinstance(raw_by_run, dict) and candidate in raw_by_run:
                        values = raw_by_run[candidate]
                        break
                if values is not None:
                    by_run[run_id] = [float(v) for v in values]

        self.config.scanner.slice_timing_by_run = by_run
        if not by_run:
            self.config.scanner.slice_timing = None
            logger.warning("Manifest contains no usable SliceTiming values")
            return

        expected_runs = [str(run.get("run_id")) for run in manifest.get("runs", [])]
        missing = [run_id for run_id in expected_runs if run_id not in by_run]
        if missing:
            raise ValueError(f"SliceTiming is missing for manifest runs: {missing}")

        arrays = [by_run[run_id] for run_id in expected_runs]
        first = arrays[0]
        all_same = all(
            len(arr) == len(first) and np.allclose(arr, first, rtol=0.0, atol=1e-9)
            for arr in arrays[1:]
        )
        self.config.scanner.slice_timing = first if all_same else None

        if all_same:
            logger.info("Manifest SliceTiming: one shared array for %d runs", len(by_run))
        else:
            logger.info(
                "Manifest SliceTiming: run-specific arrays detected for %d runs; "
                "canonical run_id keys verified.",
                len(by_run),
            )

    def _load_manifest_motion_regressors(
        self,
        manifest: Dict[str, Any],
        patient_data_path: Path,
        expected_n_timepoints: int,
    ) -> None:
        """Load six SPM realignment parameters for each run into the config.

        Realignment changes image geometry, while these regressors model residual
        signal changes correlated with motion. They are nuisance coefficients and
        are deliberately excluded from the hierarchical Omega matrix.
        """
        if not bool(getattr(self.config.design, "add_motion_regressors", False)):
            self.config.design.motion_regressors_by_run = {}
            self.config.design.motion_summary_by_run = {}
            return

        by_run: Dict[str, List[List[float]]] = {}
        summaries: Dict[str, Dict[str, Any]] = {}
        missing: List[str] = []

        for run in manifest.get("runs", []):
            run_id = str(run.get("run_id"))
            preprocessing = run.get("preprocessing", {})
            rel_path = None
            if isinstance(preprocessing, dict):
                rel_path = preprocessing.get("motion_parameters")
            if not rel_path:
                rel_path = f"runs/{run_id}/rp_bold.txt"

            path = patient_data_path / str(rel_path)
            if not path.exists():
                missing.append(f"{run_id}: {path}")
                continue

            values = np.loadtxt(path, dtype=float)
            values = np.atleast_2d(values)
            if values.shape[0] != int(expected_n_timepoints):
                raise ValueError(
                    f"Motion file {path} has {values.shape[0]} rows; "
                    f"expected {expected_n_timepoints}"
                )
            if values.shape[1] < 6:
                raise ValueError(f"Motion file {path} has only {values.shape[1]} columns")
            values = values[:, :6]
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Motion file contains non-finite values: {path}")

            by_run[run_id] = values.tolist()
            diff = np.diff(values, axis=0, prepend=values[[0], :])
            fd = np.sum(np.abs(diff[:, :3]), axis=1) + 50.0 * np.sum(
                np.abs(diff[:, 3:6]), axis=1
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            summaries[run_id] = {
                "source": str(path),
                "sha256": digest,
                "n_timepoints": int(values.shape[0]),
                "max_abs_translation_mm": np.max(np.abs(values[:, :3]), axis=0).tolist(),
                "max_abs_rotation_rad": np.max(np.abs(values[:, 3:6]), axis=0).tolist(),
                "fd_mean_mm": float(np.mean(fd)),
                "fd_max_mm": float(np.max(fd)),
                "n_fd_gt_0_2_mm": int(np.count_nonzero(fd > 0.2)),
                "n_fd_gt_0_5_mm": int(np.count_nonzero(fd > 0.5)),
            }

        if missing:
            raise FileNotFoundError(
                "SPM motion regressors are enabled but files are missing: " + "; ".join(missing)
            )

        self.config.design.motion_regressors_by_run = by_run
        self.config.design.motion_summary_by_run = summaries
        logger.info("Loaded six SPM motion regressors for %d runs", len(by_run))
        for run_id, info in summaries.items():
            logger.info(
                "Motion %s: mean FD=%.4f mm, max FD=%.4f mm, FD>0.2=%d, FD>0.5=%d",
                run_id,
                info["fd_mean_mm"],
                info["fd_max_mm"],
                info["n_fd_gt_0_2_mm"],
                info["n_fd_gt_0_5_mm"],
            )

    def _resolve_patient_path(self, root_path: Path) -> Path:
        """
        Implementation detail.
        Implementation detail.
        """
        if self.config.paths.patient_id:
            candidate = root_path / self.config.paths.patient_id
            if candidate.exists():
                return candidate
            raise FileNotFoundError(f"Patient folder not found: {candidate}")

        return root_path

    def _resolve_logs_path(self, patient_data_path: Path) -> Path:
        """
        Implementation detail.

        1. logs_path == data_path:
           Implementation detail.

        Implementation detail.
           Implementation detail.
        """
        if self.config.paths.logs_path == self.config.paths.data_path:
            patient_logs_path = patient_data_path / "log"
        else:
            patient_logs_path = self._resolve_patient_path(self.config.paths.logs_path)

            if (patient_logs_path / "log").exists():
                patient_logs_path = patient_logs_path / "log"

        return patient_logs_path

    def load_patient_runs(self, patient_data_path: Path) -> Dict[str, RunData]:
        runs: Dict[str, RunData] = {}

        for run_dir in sorted(patient_data_path.iterdir()):
            if not run_dir.is_dir():
                continue

            if run_dir.name in self.config.data.banned_run_names:
                continue

            if not self._is_allowed_run_name(run_dir.name):
                continue

            if not self._passes_run_mode(run_dir.name):
                continue

            try:
                runs[run_dir.name] = self.load_run_data(run_dir)
            except Exception as exc:
                logger.warning("Could not load run %s: %s", run_dir, exc)

        if not runs:
            raise RuntimeError(f"{patient_data_path} contains no suitable run folders")

        return runs

    def load_run_data(self, run_dir: Path) -> RunData:
        nii_files = sorted(run_dir.glob("*.nii")) + sorted(run_dir.glob("*.nii.gz"))

        if not nii_files:
            raise FileNotFoundError(f"{run_dir} contains no .nii/.nii.gz files")

        volumes: List[np.ndarray] = []
        volume_source_files: List[Path] = []
        first_img: Optional[nib.Nifti1Image] = None

        for file in nii_files:
            img = nib.load(str(file))

            if first_img is None:
                first_img = img

            data = np.asarray(img.dataobj)

            if data.ndim == 3:
                volumes.append(data)
                volume_source_files.append(file)

            elif data.ndim == 4:
                for t in range(data.shape[-1]):
                    volumes.append(data[..., t])
                    volume_source_files.append(file)

            else:
                raise ValueError(f"Unexpected NIfTI dimensionality {file}: {data.shape}")

        if first_img is None:
            raise RuntimeError(f"Could not load images from {run_dir}")

        run_data = np.stack(volumes, axis=0).astype(np.float32)

        return RunData(
            name=run_dir.name,
            data=run_data,
            affine=first_img.affine,
            header=first_img.header.copy(),
            source_files=volume_source_files,
        )

    def load_patient_logs(self, patient_logs_path: Path) -> Dict[str, pd.DataFrame]:
        if not patient_logs_path.exists():
            logger.warning("Log folder not found: %s", patient_logs_path)
            return {}

        logs: Dict[str, pd.DataFrame] = {}

        log_files: List[Path] = []
        for pattern in ("*.txt", "*.tsv", "*.csv"):
            log_files.extend(sorted(patient_logs_path.glob(pattern)))

        for file in log_files:
            try:
                df = self._read_log_file(file)
            except Exception as exc:
                logger.warning("Could not read log %s: %s", file, exc)
                continue

            block_col = self.config.data.log_block_column

            if block_col in df.columns:
                block_nums = [x for x in df[block_col].dropna().unique() if int(x) != 0]

                for block_num in block_nums:
                    run_log = df[df[block_col] == block_num].copy().reset_index(drop=True)
                    run_name = self._run_name_from_log_file(file, int(block_num))
                    logs[run_name] = run_log
            else:
                logs[file.stem] = df.reset_index(drop=True)

        return logs

    def apply_common_stimulus_filter(
        self,
        logs: Dict[str, pd.DataFrame],
        runs: Dict[str, RunData],
    ) -> tuple[Dict[str, pd.DataFrame], Dict[str, RunData]]:
        """
        Implementation detail.

        Implementation detail.
        Implementation detail.
        Implementation detail.
        Implementation detail.
        Implementation detail.
        Implementation detail.
        Implementation detail.
        """
        filtered_logs: Dict[str, pd.DataFrame] = {}
        filtered_runs: Dict[str, RunData] = {}

        common_run_names = sorted([name for name in runs.keys() if name in logs])

        if not common_run_names:
            raise RuntimeError("No matching run names between data and logs")

        for run_name in common_run_names:
            raw_log = logs[run_name]
            run_data = runs[run_name]

            prepared_log = self._prepare_log_for_common_stimuli(raw_log)
            scan_indices = self._scan_indices_from_log(prepared_log, run_name=run_name)

            new_run_data = self._filter_run_scans(
                run_data=run_data,
                scan_indices=scan_indices,
            )

            filtered_logs[run_name] = prepared_log
            filtered_runs[run_name] = new_run_data

            logger.info(
                "Run %s: after log filtering %d rows, scans %d",
                run_name,
                len(prepared_log),
                new_run_data.data.shape[0],
            )

        return filtered_logs, filtered_runs

    def _prepare_log_for_common_stimuli(self, log_df: pd.DataFrame) -> pd.DataFrame:
        df = log_df.copy().reset_index(drop=True)

        response_col = self.config.data.log_response_column
        trial_col = self.config.data.log_trial_column
        stimulus_col = self.config.data.log_stimulus_column

        if response_col in df.columns:
            for ind in range(0, len(df) - 1, 2):
                df.loc[ind, response_col] = df.loc[ind + 1, response_col]

        if self.config.data.keep_even_log_rows_only:
            df = df[df.index % 2 == 0].copy()

        if self.config.data.remove_zero_trial_rows:
            if trial_col not in df.columns:
                available = ", ".join(map(str, df.columns.tolist()))
                raise KeyError(
                    f"Filtering requires column '{trial_col}', but it is missing. "
                    f"Available columns: {available}"
                )

            df = df[df[trial_col] != 0].copy()

        df = df.reset_index(drop=True)

        if stimulus_col not in df.columns:
            available = ", ".join(map(str, df.columns.tolist()))
            raise KeyError(
                f"Filtering requires column '{stimulus_col}', but it is missing. "
                f"Available columns: {available}"
            )

        df.loc[:, stimulus_col] = df[stimulus_col].astype(str)

        rename_map = self.config.data.stimulus_rename_map
        if rename_map:
            df.loc[:, stimulus_col] = df[stimulus_col].replace(rename_map)

        if self.config.data.append_trial_group_to_stimulus_name:
            df.loc[:, stimulus_col] = self._append_trial_group_to_stimulus_names(df)

        patterns = self.config.data.common_stimulus_patterns
        if patterns:
            keep_mask = np.zeros(len(df), dtype=bool)

            for pattern in patterns:
                keep_mask |= df[stimulus_col].str.contains(pattern, na=False, regex=False).to_numpy()

            df = df[keep_mask].copy().reset_index(drop=True)

        return df

    def _append_trial_group_to_stimulus_names(self, df: pd.DataFrame) -> List[str]:
        stimulus_col = self.config.data.log_stimulus_column
        trial_col = self.config.data.log_trial_column
        group_size = self.config.data.trial_group_size

        result: List[str] = []

        for _, row in df.iterrows():
            image = str(row[stimulus_col])
            trial_num = int(row[trial_col])
            n_trial = (trial_num - 1) // group_size + 1

            if "." in image:
                base, ext = image.rsplit(".", 1)
                new_image = f"{base}_{n_trial}.{ext}"
            else:
                new_image = f"{image}_{n_trial}"

            result.append(new_image)

        return result

    def _scan_indices_from_log(self, log_df: pd.DataFrame, run_name: str) -> List[int]:
        trial_col = self.config.data.log_trial_column

        if trial_col not in log_df.columns:
            available = ", ".join(map(str, log_df.columns.tolist()))
            raise KeyError(
                f"Run log {run_name} does not contain column '{trial_col}'. "
                f"Available columns: {available}"
            )

        scans_per_trial = self.config.data.scans_per_trial
        offset = self.config.data.scan_index_offset

        scan_indices: List[int] = []

        for trial_num_raw in log_df[trial_col].values:
            trial_num = int(trial_num_raw)

            start = trial_num * scans_per_trial + offset
            stop = start + scans_per_trial

            scan_indices.extend(list(range(start, stop)))

        return scan_indices

    def _filter_run_scans(self, run_data: RunData, scan_indices: List[int]) -> RunData:
        n_available = run_data.data.shape[0]

        invalid = [i for i in scan_indices if i < 0 or i >= n_available]

        if invalid:
            msg = (
                f"Run {run_data.name}: some scan_indices are out of bounds. "
                f"available scans: {n_available}. "
                f"invalid indices, first 10: {invalid[:10]}"
            )

            if self.config.data.strict_scan_filtering:
                raise IndexError(msg)

            logger.warning(msg)

        valid_indices = [i for i in scan_indices if 0 <= i < n_available]

        if len(valid_indices) == 0:
            raise RuntimeError(f"Run {run_data.name}: no scans remain after filtering")

        filtered_data = run_data.data[valid_indices, :, :, :]

        expected_nT = self.config.scanner.nT
        if filtered_data.shape[0] != expected_nT:
            logger.warning(
                "Run %s: after filtering got %d scans, but scanner.nT=%d",
                run_data.name,
                filtered_data.shape[0],
                expected_nT,
            )

        filtered_source_files: List[Path] = []

        if len(run_data.source_files) == n_available:
            filtered_source_files = [run_data.source_files[i] for i in valid_indices]
        else:
            filtered_source_files = run_data.source_files

        return RunData(
            name=run_data.name,
            data=filtered_data,
            affine=run_data.affine,
            header=run_data.header,
            source_files=filtered_source_files,
        )

    def load_or_compute_mask(self, patient: PatientData) -> BrainMask:
        """
        Load the brain mask.

        Order:
        1. paths.mask_path, if specified and passes sanity checks.
        2. Ready-made mask from preprocessed/<patient>, if it passes sanity checks.
        3. Fallback mask construction from functional data or SPM tissue files.

        The sanity checks are important for SPM-resliced fMRI: a naive
        nonzero mask can cover almost the whole FOV because interpolation leaves
        tiny nonzero values in the background. Such masks make ROC/Kappa and
        reproducibility maps look artificially strong everywhere.
        """
        first_run = next(iter(patient.runs.values()))
        expected_shape = first_run.spatial_shape

        explicit_mask_failed = False

        if self.config.paths.mask_path is not None:
            try:
                return self._load_mask_from_path(
                    mask_path=Path(self.config.paths.mask_path),
                    expected_shape=expected_shape,
                    label="external mask_path",
                )
            except ValueError as exc:
                if self._is_rejectable_mask_error(exc):
                    explicit_mask_failed = True
                    logger.warning(
                        "Rejecting external mask_path and falling back to computed mask: %s",
                        exc,
                    )
                else:
                    raise

        precomputed_mask_path = self._find_precomputed_patient_mask_path(patient)
        if precomputed_mask_path is not None:
            # If this is exactly the rejected explicit mask, do not try it again.
            try_again = True
            if explicit_mask_failed and self.config.paths.mask_path is not None:
                try:
                    try_again = Path(precomputed_mask_path).resolve() != Path(self.config.paths.mask_path).resolve()
                except Exception:
                    try_again = True

            if try_again:
                try:
                    return self._load_mask_from_path(
                        mask_path=precomputed_mask_path,
                        expected_shape=expected_shape,
                        label="ready-made SPM brain mask",
                    )
                except ValueError as exc:
                    if self._is_rejectable_mask_error(exc):
                        logger.warning(
                            "Rejecting ready-made SPM brain mask and falling back to computed mask: %s",
                            exc,
                        )
                    else:
                        raise

        method = str(self.config.preprocessing.mask_method).lower().strip()

        if method == "functional_nonzero":
            mask = self._compute_functional_nonzero_brain_mask(patient)

        elif method == "spm_tissue":
            mask = self._compute_spm_tissue_brain_mask(patient)

        else:
            mask = self._compute_functional_brain_mask(patient)

        self._validate_mask_volume_fraction(mask, label=f"computed {method} mask")

        logger.info(
            "Brain mask was built with method %s: %d voxels",
            method,
            int(mask.sum()),
        )

        return BrainMask(
            mask=mask,
            affine=first_run.affine,
            header=first_run.header,
        )

    def _load_mask_from_path(
        self,
        mask_path: Path,
        expected_shape: tuple[int, int, int],
        label: str,
    ) -> BrainMask:
        mask_path = Path(mask_path)
        mask_img = nib.load(str(mask_path))
        mask = mask_img.get_fdata().astype(bool)

        if mask.shape != expected_shape:
            raise ValueError(
                f"{label} has shape={mask.shape}, "
                f"but functional data have shape={expected_shape}. "
                f"File: {mask_path}"
            )

        raw_mask = np.asarray(mask, dtype=bool)
        cleaned_mask = self._clean_brain_mask(raw_mask)

        # A ready-made external/SPM mask is authoritative. Morphological hole
        # filling or closing must never add voxels that were absent from that
        # file, otherwise the statistical maps can extend outside the tissue
        # mask used by SPM. Cleaning is therefore allowed to remove disconnected
        # voxels, but not to enlarge the supplied mask.
        mask = np.asarray(cleaned_mask & raw_mask, dtype=bool)
        added = int(np.count_nonzero(mask & ~raw_mask))
        removed = int(np.count_nonzero(raw_mask & ~mask))
        self._validate_mask_volume_fraction(mask, label=f"{label} ({mask_path})")

        logger.info(
            "Loaded authoritative %s without mask expansion: %s, "
            "voxels=%d, removed=%d, added=%d",
            label,
            mask_path,
            int(mask.sum()),
            removed,
            added,
        )

        return BrainMask(
            mask=mask,
            affine=mask_img.affine,
            header=mask_img.header.copy(),
        )

    def _find_precomputed_patient_mask_path(self, patient: PatientData) -> Optional[Path]:
        """
        Find a ready-made mask from the new SPM export.

        Implementation detail.
        is in the native T1 grid and may not match the functional shape.
        """
        try:
            patient_data_path = self._resolve_patient_path(self.config.paths.data_path)
        except Exception:
            return None

        structural_dir = patient_data_path / "structural"

        candidates = [
            patient_data_path / "brain_mask.nii.gz",
            patient_data_path / "brain_mask.nii",
            structural_dir / "brain_mask_fmri.nii.gz",
            structural_dir / "brain_mask_fmri.nii",
            structural_dir / "brain_mask_anat_funcgrid.nii.gz",
            structural_dir / "brain_mask_anat_funcgrid.nii",
        ]

        for path in candidates:
            if path.exists():
                logger.info("Found ready-made SPM brain mask: %s", path)
                return path.resolve()

        return None

    @staticmethod
    def _mask_volume_fraction(mask: np.ndarray) -> tuple[int, int, float]:
        mask = np.asarray(mask, dtype=bool)
        n_mask = int(np.sum(mask))
        n_total = int(mask.size)
        fraction = n_mask / n_total if n_total > 0 else 0.0
        return n_mask, n_total, fraction

    def _log_mask_volume(self, label: str, mask: np.ndarray) -> None:
        n_mask, n_total, fraction = self._mask_volume_fraction(mask)
        logger.info(
            "%s: voxels=%d / %d, volume_fraction=%.6f, volume_percent=%.3f%%",
            label,
            n_mask,
            n_total,
            fraction,
            100.0 * fraction,
        )

    def _validate_mask_volume_fraction(self, mask: np.ndarray, label: str) -> None:
        n_mask, n_total, fraction = self._mask_volume_fraction(mask)
        min_fraction = float(getattr(self.config.preprocessing, "mask_min_volume_fraction", 0.01))
        max_fraction = float(getattr(self.config.preprocessing, "mask_max_volume_fraction", 0.80))
        reject = bool(getattr(self.config.preprocessing, "mask_reject_too_large", True))

        logger.info(
            "Mask sanity check for %s: voxels=%d / %d, volume_fraction=%.6f",
            label,
            n_mask,
            n_total,
            fraction,
        )

        if not reject:
            return

        if fraction < min_fraction:
            raise ValueError(
                f"Mask volume fraction is too small for {label}: "
                f"{fraction:.6f} < {min_fraction:.6f}"
            )

        if fraction > max_fraction:
            raise ValueError(
                f"Mask volume fraction is too large for {label}: "
                f"{fraction:.6f} > {max_fraction:.6f}. "
                "This usually indicates a whole-FOV/nonzero interpolation mask."
            )

    @staticmethod
    def _is_rejectable_mask_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "mask volume fraction" in text
        
    def _compute_spm_tissue_brain_mask(self, patient: PatientData) -> np.ndarray:
        """
        Brain mask from SPM T1 segmentation.

        Steps:
        1. build structural probability map:
               brain_probability = c1 + c2 + c3

        2. build structural brain mask:
               structural_mask = brain_probability >= spm_mask_threshold

        3. transfer structural_mask to the functional grid.

           Main mode:
               fractional_occupancy

           For each fMRI voxel compute:
               fraction = N(structural voxel centers inside fMRI voxel and inside brain mask)
                          / N(structural voxel centers inside fMRI voxel)

           fMRI-voxel is included in the brain mask if:
               fraction >= spm_mask_fraction_threshold
        """
        first_run = next(iter(patient.runs.values()))
        target_shape = first_run.spatial_shape
        target_affine = first_run.affine

        tissue_classes = list(self.config.preprocessing.spm_mask_tissue_classes)
        mode = str(getattr(self.config.preprocessing, "spm_mask_mode", "probability_sum")).lower().strip()
        threshold = float(self.config.preprocessing.spm_mask_threshold)

        tissue_paths = self._find_spm_tissue_files(
            patient_id=patient.patient_id,
            tissue_classes=tissue_classes,
        )

        if mode != "probability_sum":
            logger.warning(
                "Recommended SPM mask mode is spm_mask_mode='probability_sum'. "
                "current mode: %s",
                mode,
            )

        prob_data = None
        first_img = None
        source_shape = None

        for tissue_class in tissue_classes:
            path = tissue_paths[tissue_class]
            img = nib.load(str(path))
            data = np.asarray(img.get_fdata(), dtype=float)
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            if first_img is None:
                first_img = img
                source_shape = data.shape
                prob_data = np.zeros_like(data, dtype=float)
            else:
                if data.shape != source_shape:
                    raise ValueError(
                        f"SPM tissue file {path} has shape={data.shape}, "
                        f"expected {source_shape}"
                    )

            prob_data += data

            logger.info(
                "SPM tissue c%d loaded: %s, min=%.4f, max=%.4f",
                tissue_class,
                path,
                float(np.nanmin(data)),
                float(np.nanmax(data)),
            )

        if first_img is None or prob_data is None:
            raise RuntimeError("Could not load SPM tissue probability maps")

        prob_data = np.clip(prob_data, 0.0, 1.0)

        structural_raw_mask = np.isfinite(prob_data) & (prob_data >= threshold)

        logger.info(
            "Raw structural SPM tissue mask: classes=%s, threshold=%.3f",
            tissue_classes,
            threshold,
        )
        self._log_mask_volume("Raw structural SPM tissue mask", structural_raw_mask)

        structural_mask = self._clean_brain_mask(structural_raw_mask)

        self._log_mask_volume("Clean structural SPM tissue mask", structural_mask)

        if self.config.preprocessing.spm_mask_resample_to_functional:
            transfer_method = str(
                getattr(
                    self.config.preprocessing,
                    "spm_mask_transfer_method",
                    "fractional_occupancy",
                )
            ).lower().strip()

            if transfer_method == "fractional_occupancy":
                fraction_threshold = float(
                    getattr(
                        self.config.preprocessing,
                        "spm_mask_fraction_threshold",
                        0.5,
                    )
                )

                raw_mask = self._transfer_structural_mask_to_functional_fractional(
                    structural_mask=structural_mask,
                    structural_affine=first_img.affine,
                    target_shape=target_shape,
                    target_affine=target_affine,
                    fraction_threshold=fraction_threshold,
                )

            elif transfer_method == "resample_probability":
                logger.info(
                    "Using legacy mask transfer: resample_probability"
                )

                prob_img = nib.Nifti1Image(
                    prob_data.astype(np.float32),
                    affine=first_img.affine,
                    header=first_img.header.copy(),
                )

                resampled_img = resample_from_to(
                    prob_img,
                    (target_shape, target_affine),
                    order=1,
                )
                prob_func = np.asarray(resampled_img.get_fdata(), dtype=float)

                raw_mask = np.isfinite(prob_func) & (prob_func >= threshold)

            else:
                raise ValueError(
                    "Unknown preprocessing.spm_mask_transfer_method. "
                    "Allowed values: 'fractional_occupancy', 'resample_probability'. "
                    f"Got: {transfer_method}"
                )

        else:
            if structural_mask.shape != target_shape:
                raise ValueError(
                    "Structural SPM mask does not match the functional grid: "
                    f"{structural_mask.shape} != {target_shape}. "
                    "Set spm_mask_resample_to_functional: true."
                )

            raw_mask = structural_mask

        self._log_mask_volume("Raw functional brain mask from structural mask", raw_mask)

        mask = self._clean_brain_mask(raw_mask)

        if self.config.preprocessing.spm_mask_intersect_with_functional_nonzero:
            functional_nonzero = self._compute_functional_nonzero_brain_mask(patient)
            before = int(mask.sum())

            mask = mask & functional_nonzero
            mask = self._clean_brain_mask(mask)

            logger.info(
                "SPM-derived functional mask intersected with functional nonzero mask: "
                "before=%d, after=%d, removed=%d",
                before,
                int(mask.sum()),
                before - int(mask.sum()),
            )

        min_voxels = int(self.config.preprocessing.mask_min_voxels)
        if int(mask.sum()) < min_voxels:
            raise RuntimeError(
                f"SPM-derived functional brain mask is too small: {int(mask.sum())} voxels. "
                f"Minimum: {min_voxels}. "
                "Check spm_mask_threshold, spm_mask_fraction_threshold and tissue files c1/c2/c3."
            )

        self._log_mask_volume("Final functional brain mask", mask)

        return mask.astype(bool)
    
    def _transfer_structural_mask_to_functional_fractional(
        self,
        structural_mask: np.ndarray,
        structural_affine: np.ndarray,
        target_shape: tuple[int, int, int],
        target_affine: np.ndarray,
        fraction_threshold: float,
    ) -> np.ndarray:
        """
        Implementation detail.
        of structural voxels falling into each fMRI voxel.

        For each structural voxel center:
        1. convert structural voxel index -> world coordinates;
        2. convert world coordinates -> functional voxel coordinates;
        Implementation detail.
        4. count:
              total_counts[fMRI voxel] += 1
              Implementation detail.

        Then:
              fraction = brain_counts / total_counts
              functional_mask = fraction >= fraction_threshold
        """
        structural_mask = np.asarray(structural_mask, dtype=bool)

        sx, sy, sz = structural_mask.shape
        tx, ty, tz = target_shape

        total_counts = np.zeros(target_shape, dtype=np.int32)
        brain_counts = np.zeros(target_shape, dtype=np.int32)

        total_flat = total_counts.ravel()
        brain_flat = brain_counts.ravel()

        inv_target_affine = np.linalg.inv(target_affine)

        x_grid, y_grid = np.meshgrid(
            np.arange(sx, dtype=float),
            np.arange(sy, dtype=float),
            indexing="ij",
        )

        xy = np.column_stack(
            [
                x_grid.ravel(),
                y_grid.ravel(),
            ]
        )

        n_xy = xy.shape[0]

        logger.info(
            "Transfer structural mask to the functional grid using fractional occupancy: "
            "structural_shape=%s, functional_shape=%s, fraction_threshold=%.3f",
            structural_mask.shape,
            target_shape,
            fraction_threshold,
        )

        for z in range(sz):
            structural_vox = np.empty((n_xy, 3), dtype=float)
            structural_vox[:, 0] = xy[:, 0]
            structural_vox[:, 1] = xy[:, 1]
            structural_vox[:, 2] = float(z)

            world = apply_affine(structural_affine, structural_vox)
            functional_vox = apply_affine(inv_target_affine, world)

            functional_idx = np.floor(functional_vox + 0.5).astype(np.int64)

            valid = (
                (functional_idx[:, 0] >= 0)
                & (functional_idx[:, 0] < tx)
                & (functional_idx[:, 1] >= 0)
                & (functional_idx[:, 1] < ty)
                & (functional_idx[:, 2] >= 0)
                & (functional_idx[:, 2] < tz)
            )

            if not np.any(valid):
                continue

            ix = functional_idx[valid, 0]
            iy = functional_idx[valid, 1]
            iz = functional_idx[valid, 2]

            lin = np.ravel_multi_index(
                (ix, iy, iz),
                dims=target_shape,
            )

            total_flat += np.bincount(
                lin,
                minlength=total_flat.size,
            ).astype(np.int32)

            structural_slice_mask = structural_mask[:, :, z].ravel()
            valid_structural_brain = structural_slice_mask[valid]

            if np.any(valid_structural_brain):
                brain_lin = lin[valid_structural_brain]

                brain_flat += np.bincount(
                    brain_lin,
                    minlength=brain_flat.size,
                ).astype(np.int32)

        has_coverage = total_counts > 0

        fraction = np.zeros(target_shape, dtype=np.float32)
        fraction[has_coverage] = (
            brain_counts[has_coverage].astype(np.float32)
            / total_counts[has_coverage].astype(np.float32)
        )

        functional_mask = has_coverage & (fraction >= fraction_threshold)

        logger.info(
            "Fractional occupancy transfer: covered_fMRI_voxels=%d / %d",
            int(np.sum(has_coverage)),
            int(has_coverage.size),
        )

        logger.info(
            "Fractional occupancy stats inside covered voxels: min=%.4f, median=%.4f, max=%.4f",
            float(np.min(fraction[has_coverage])) if np.any(has_coverage) else 0.0,
            float(np.median(fraction[has_coverage])) if np.any(has_coverage) else 0.0,
            float(np.max(fraction[has_coverage])) if np.any(has_coverage) else 0.0,
        )

        self._log_mask_volume(
            "Functional mask after fractional occupancy threshold",
            functional_mask,
        )

        return functional_mask.astype(bool)

    def _find_spm_tissue_files(
        self,
        patient_id: str,
        tissue_classes: List[int],
    ) -> Dict[int, Path]:
        patient_data_path = self._resolve_patient_path(self.config.paths.data_path)
        structural_dir = patient_data_path / "structural"

        if not structural_dir.exists():
            raise FileNotFoundError(f"Structural folder not found: {structural_dir}")

        result: Dict[int, Path] = {}

        for tissue_class in tissue_classes:
            exact_candidates = [
                structural_dir / f"c{tissue_class}{patient_id}_T1.nii",
                structural_dir / f"c{tissue_class}{patient_id}_T1.nii.gz",
                structural_dir / f"wc{tissue_class}{patient_id}_T1.nii",
                structural_dir / f"wc{tissue_class}{patient_id}_T1.nii.gz",
            ]

            found = None

            for candidate in exact_candidates:
                if candidate.exists():
                    found = candidate
                    break

            if found is None:
                glob_candidates = sorted(structural_dir.glob(f"c{tissue_class}*T1*.nii*"))
                if not glob_candidates:
                    glob_candidates = sorted(structural_dir.glob(f"wc{tissue_class}*T1*.nii*"))
                if glob_candidates:
                    found = glob_candidates[0]

            if found is None:
                raise FileNotFoundError(
                    f"Not found SPM tissue file c{tissue_class} for patient {patient_id} "
                    f"in folder {structural_dir}"
                )

            result[tissue_class] = found.resolve()

        logger.info(
            "SPM tissue files for brain mask: %s",
            {f'c{k}': str(v) for k, v in result.items()},
        )

        return result

    def _compute_functional_nonzero_brain_mask(self, patient: PatientData) -> np.ndarray:
        """
        Brain mask from nonzero fMRI signal.

        Suitable for preprocessed SPM scans, where the signal outside the brain is already zero.
        Implementation detail.
            mask = mean_abs > eps
        """
        eps = float(self.config.preprocessing.spm_mask_functional_nonzero_eps)

        run_names = sorted(patient.runs.keys())
        if not self.config.preprocessing.mask_use_all_runs:
            run_names = run_names[:1]

        mean_images: List[np.ndarray] = []

        for run_name in run_names:
            data = np.asarray(patient.runs[run_name].data, dtype=np.float32)
            mean_img = np.nanmean(np.abs(data), axis=0)
            mean_images.append(mean_img)

        mean_abs = np.nanmean(np.stack(mean_images, axis=0), axis=0)
        raw_mask = np.isfinite(mean_abs) & (mean_abs > eps)

        logger.info(
            "Raw functional nonzero mask: eps=%.6g, voxels=%d",
            eps,
            int(raw_mask.sum()),
        )

        mask = self._clean_brain_mask(raw_mask)

        min_voxels = int(self.config.preprocessing.mask_min_voxels)
        if int(mask.sum()) < min_voxels:
            raise RuntimeError(
                f"After cleaning functional nonzero mask is too small: {int(mask.sum())} voxels. "
                f"Minimum: {min_voxels}."
            )

        logger.info(
            "Clean functional nonzero mask: voxels=%d, removed=%d",
            int(mask.sum()),
            int(raw_mask.sum()) - int(mask.sum()),
        )

        return mask.astype(bool)
        
    def _compute_functional_brain_mask(self, patient: PatientData) -> np.ndarray:
        method = str(self.config.preprocessing.mask_method).lower().strip()

        run_names = sorted(patient.runs.keys())

        if not self.config.preprocessing.mask_use_all_runs:
            run_names = run_names[:1]

        mean_images: List[np.ndarray] = []

        for run_name in run_names:
            data = np.asarray(patient.runs[run_name].data, dtype=np.float32)

            # mean(abs(signal)) is more robust than mean(signal)
            # when data are centered or contain negative values.
            mean_img = np.nanmean(np.abs(data), axis=0)
            mean_images.append(mean_img)

        mean_signal = np.nanmean(np.stack(mean_images, axis=0), axis=0)

        finite = np.isfinite(mean_signal)
        positive = finite & (mean_signal > 0)

        if not np.any(positive):
            raise RuntimeError("Could not build mask: there are no positive finite signal values")

        values = mean_signal[positive]

        if method == "functional_otsu":
            threshold = self._otsu_threshold(
                values=values,
                n_bins=int(self.config.preprocessing.mask_otsu_n_bins),
            )
        elif method == "functional_percentile":
            threshold = float(
                np.percentile(
                    values,
                    float(self.config.preprocessing.mask_intensity_percentile),
                )
            )
        else:
            raise ValueError(
                "Unknown preprocessing.mask_method. "
                "Allowed values: 'spm_tissue', 'functional_otsu', 'functional_percentile'. "
                f"Got: {self.config.preprocessing.mask_method}"
            )

        raw_mask = positive & (mean_signal >= threshold)

        logger.info(
            "Raw functional mask: method=%s, threshold=%.6g, voxels=%d",
            method,
            threshold,
            int(raw_mask.sum()),
        )

        mask = self._clean_brain_mask(raw_mask)

        min_voxels = int(self.config.preprocessing.mask_min_voxels)
        if int(mask.sum()) < min_voxels:
            raise RuntimeError(
                f"After cleaning brain mask is too small: {int(mask.sum())} voxels. "
                f"Minimum: {min_voxels}. "
                "Check preprocessing.mask_method or mask_intensity_percentile."
            )

        logger.info(
            "Clean functional mask: voxels=%d, removed=%d",
            int(mask.sum()),
            int(raw_mask.sum()) - int(mask.sum()),
        )

        return mask

    def _clean_brain_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)

        structure = ndi.generate_binary_structure(rank=3, connectivity=1)

        opening_iter = int(self.config.preprocessing.mask_opening_iterations)
        closing_iter = int(self.config.preprocessing.mask_closing_iterations)

        if opening_iter > 0:
            mask = ndi.binary_opening(mask, structure=structure, iterations=opening_iter)

        if self.config.preprocessing.mask_keep_largest_component:
            mask = self._keep_largest_connected_component(mask)

        if self.config.preprocessing.mask_fill_holes:
            mask = ndi.binary_fill_holes(mask)

        if closing_iter > 0:
            mask = ndi.binary_closing(mask, structure=structure, iterations=closing_iter)

        if self.config.preprocessing.mask_keep_largest_component:
            mask = self._keep_largest_connected_component(mask)

        return mask.astype(bool)

    def _keep_largest_connected_component(self, mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)

        if not np.any(mask):
            return mask

        structure = ndi.generate_binary_structure(rank=3, connectivity=1)
        labeled, n_components = ndi.label(mask, structure=structure)

        if n_components <= 1:
            return mask

        component_sizes = np.bincount(labeled.ravel())
        component_sizes[0] = 0

        largest_label = int(np.argmax(component_sizes))
        largest = labeled == largest_label

        logger.info(
            "Brain mask connected components: n=%d, largest=%d voxels, removed=%d voxels",
            int(n_components),
            int(component_sizes[largest_label]),
            int(np.sum(mask)) - int(component_sizes[largest_label]),
        )

        return largest

    @staticmethod
    def _otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
        """
        Simple Otsu threshold without a scikit-image dependency.

        Find the threshold that maximizes between-class variance
        for a one-dimensional intensity distribution.
        """
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            raise ValueError("Otsu threshold: empty value array")

        v_min = float(np.min(values))
        v_max = float(np.max(values))

        if v_max <= v_min:
            return v_min

        hist, bin_edges = np.histogram(values, bins=int(n_bins), range=(v_min, v_max))
        hist = hist.astype(float)

        total = np.sum(hist)
        if total <= 0:
            return float(np.percentile(values, 35.0))

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        weight_bg = np.cumsum(hist)
        weight_fg = total - weight_bg

        sum_bg = np.cumsum(hist * bin_centers)
        sum_total = sum_bg[-1]
        sum_fg = sum_total - sum_bg

        valid = (weight_bg > 0) & (weight_fg > 0)

        mean_bg = np.zeros_like(bin_centers)
        mean_fg = np.zeros_like(bin_centers)

        mean_bg[valid] = sum_bg[valid] / weight_bg[valid]
        mean_fg[valid] = sum_fg[valid] / weight_fg[valid]

        between = np.zeros_like(bin_centers)
        between[valid] = weight_bg[valid] * weight_fg[valid] * (mean_bg[valid] - mean_fg[valid]) ** 2

        idx = int(np.argmax(between))
        return float(bin_centers[idx])

    def _read_log_file(self, file: Path) -> pd.DataFrame:
        sep = self.config.data.log_delimiter
        return pd.read_csv(file, sep=sep)

    def _run_name_from_log_file(self, file: Path, block_num: int) -> str:
        """
        Basic legacy-code logic:
            file.split('_')[-1].split('.')[0] + '_' + str(block_num)

        Additionally handle names robustly, where fr/me are not the last name component.
        """
        stem = file.stem.lower()

        if "fr" in stem:
            prefix = "fr"
        elif "me" in stem:
            prefix = "me"
        else:
            prefix = file.name.split("_")[-1].split(".")[0]

        return f"{prefix}_{block_num}"

    def _is_allowed_run_name(self, run_name: str) -> bool:
        if "." in run_name:
            return False

        parts = run_name.split("_")
        if len(parts) < 2:
            return False

        if not parts[1].isnumeric():
            return False

        if self.config.data.run_name_prefixes:
            return parts[0] in self.config.data.run_name_prefixes

        return True

    def _passes_run_mode(self, run_name: str) -> bool:
        mode = self.config.data.run_mode
        prefix = run_name.split("_")[0]

        if mode == "+":
            return True

        return prefix == mode


def sorted_common_run_names(patient: PatientData, expected_n_runs: Optional[int] = None) -> List[str]:
    if patient.run_order:
        run_names = [name for name in patient.run_order if name in patient.runs]
    else:
        run_names = sorted(patient.runs.keys())

    if patient.logs:
        common = [name for name in run_names if name in patient.logs]
        if common:
            run_names = common
        else:
            logger.warning(
                "Run names in data and logs did not match. "
                "Data runs will be used without strict log matching."
            )

    if expected_n_runs is not None and expected_n_runs > 0:
        run_names = run_names[:expected_n_runs]

    return run_names
