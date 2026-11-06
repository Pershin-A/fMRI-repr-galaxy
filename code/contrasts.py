# file: contrasts.py

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from config import AppConfig
from data_types import EBResult

logger = logging.getLogger(__name__)


class ContrastCalculator:
    def __init__(self, config: AppConfig):
        self.config = config

    def first_level_contrasts(self, k: int) -> Dict[str, np.ndarray]:
        """
        Build configured first-level contrast vectors.

        Two forms are supported:

        1. Legacy positional form:
               values: [1, -1, 0, ...]
               auto_pad_to_nS: true

        2. Manifest condition-column form:
               columns: ["Self", "Malleable", ...]
               values:  [1, -1, ...]
               auto_pad_nuisance: true

           In this form columns are mapped onto config.data.condition_order,
           and nuisance regressors are kept at zero.
        """
        result: Dict[str, np.ndarray] = {}
        raw = self.config.analysis.first_level_contrasts

        if not raw:
            c = np.zeros(k)
            c[0] = 1.0
            result["regressor_1"] = c
            return result

        selected_names = getattr(self.config.analysis, "first_level_contrast_names", None)
        if selected_names:
            selected_set = {str(name) for name in selected_names}
            missing = sorted(selected_set - set(raw.keys()))
            if missing:
                raise ValueError(
                    f"Requested first-level contrasts are absent from the configuration: {missing}. "
                    f"Available: {sorted(raw.keys())}"
                )
            raw_items = [(name, spec) for name, spec in raw.items() if name in selected_set]
        else:
            raw_items = list(raw.items())

        condition_order = list(getattr(self.config.data, "condition_order", []))

        for name, spec in raw_items:
            values = np.asarray(spec["values"], dtype=float)

            if "columns" in spec:
                columns = [str(col) for col in spec["columns"]]

                if len(columns) != len(values):
                    raise ValueError(
                        f"Contrast {name}: columns length {len(columns)} does not match "
                        f"values length {len(values)}"
                    )

                if not condition_order:
                    if len(values) > k:
                        raise ValueError(
                            f"Contrast {name} has {len(values)} values, but k={k}"
                        )
                    c = np.zeros(k, dtype=float)
                    c[: len(values)] = values
                else:
                    c = np.zeros(k, dtype=float)

                    for column, value in zip(columns, values):
                        if column not in condition_order:
                            raise ValueError(
                                f"Contrast {name} refers to column '{column}', "
                                f"but condition_order is {condition_order}"
                            )

                        idx = condition_order.index(column)
                        if idx >= k:
                            raise ValueError(
                                f"Contrast {name} column '{column}' maps to index {idx}, "
                                f"but design has only k={k} columns"
                            )

                        c[idx] = float(value)

                result[name] = c
                continue

            auto_pad = bool(
                spec.get("auto_pad_to_nS", False)
                or spec.get("auto_pad_nuisance", False)
            )

            c = np.atleast_1d(np.asarray(values, dtype=float)).astype(float).ravel()

            if auto_pad and len(c) < k:
                c = np.pad(c, (0, k - len(c)))

            if len(c) != k:
                raise ValueError(f"Contrast {name} has length {len(c)}, expected {k}")

            result[name] = c

        return result

    def build_me_vs_fr_contrast(
        self,
        k_design: int,
        images_order: List[str],
        logs_by_run: Dict[str, pd.DataFrame],
        run_names: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Implementation detail.

        Implementation detail.
            coefficient = MeMorphShare / 100 - 0.5

        Implementation detail.
        Implementation detail.
        Implementation detail.

        Implementation detail.
        Implementation detail.
        """
        nS = self.config.scanner.nS

        if len(images_order) == 0:
            raise RuntimeError(
                "images_order is empty. Cannot build the me-vs-fr contrast. "
                "Call set_images_order_from_logs() first."
            )

        if k_design < nS:
            raise ValueError(
                f"k_design={k_design}, but scanner.nS={nS}. "
                "The design matrix has fewer columns than stimulus regressors."
            )

        stimulus_coeffs = np.zeros(nS, dtype=float)

        stimulus_col = self.config.data.log_stimulus_column
        me_col = self.config.data.log_me_morph_column
        fr_col = self.config.data.log_fr_morph_column

        collected_me: Dict[str, List[float]] = {image: [] for image in images_order[:nS]}
        collected_fr: Dict[str, List[float]] = {image: [] for image in images_order[:nS]}

        for run_name in run_names:
            if run_name not in logs_by_run:
                continue

            log_df = logs_by_run[run_name]

            for required_col in (stimulus_col, me_col, fr_col):
                if required_col not in log_df.columns:
                    available = ", ".join(map(str, log_df.columns.tolist()))
                    raise KeyError(
                        f"Run {run_name} log does not contain column '{required_col}'. "
                        f"Available columns: {available}"
                    )

            for image in images_order[:nS]:
                image_slice = log_df[log_df[stimulus_col].astype(str) == str(image)]

                if len(image_slice) == 0:
                    continue

                me_share = float(image_slice[me_col].values[0]) / 100.0
                fr_share = float(image_slice[fr_col].values[0]) / 100.0

                collected_me[image].append(me_share)
                collected_fr[image].append(fr_share)

        for i, image in enumerate(images_order[:nS]):
            if not (("MeFr" in image) or ("FrMe" in image)):
                stimulus_coeffs[i] = 0.0
                continue

            me_values = collected_me.get(image, [])

            if len(me_values) == 0:
                logger.warning(
                    "No MeMorphShare value was found for stimulus %s in any run. "
                    "The contrast coefficient will be set to 0.",
                    image,
                )
                stimulus_coeffs[i] = 0.0
                continue

            me_share_mean = float(np.mean(me_values))

            stimulus_coeffs[i] = me_share_mean - 0.5

        c = np.zeros(k_design, dtype=float)
        c[:nS] = stimulus_coeffs

        n_nuisance = k_design - nS

        logger.info(
            "me_vs_fr contrast built: full_length=%d, stimulus_regressors=%d, nuisance_regressors=%d",
            len(c),
            nS,
            n_nuisance,
        )

        logger.info(
            "me_vs_fr stimulus coefficients: %s",
            np.round(stimulus_coeffs, 4).tolist(),
        )

        logger.info(
            "me_vs_fr full contrast: %s",
            np.round(c, 4).tolist(),
        )

        return {"me_vs_fr": c}

    def second_level_contrasts(self, k_star: int) -> Dict[str, np.ndarray]:
        result: Dict[str, np.ndarray] = {}

        raw = self.config.analysis.second_level_contrasts

        if not raw:
            d = np.zeros(k_star)
            d[0] = 1.0
            result["average"] = d
            return result

        for name, spec in raw.items():
            values = spec["values"]
            auto_pad = bool(spec.get("auto_pad_to_k_star", False))

            d = np.atleast_1d(np.asarray(values, dtype=float)).astype(float).ravel()

            if auto_pad and len(d) < k_star:
                d = np.pad(d, (0, k_star - len(d)))

            if len(d) != k_star:
                raise ValueError(f"Second-level contrast {name} has length {len(d)}, expected {k_star}")

            result[name] = d

        return result

    def compute_runwise_t(self, eb: EBResult, c: np.ndarray) -> np.ndarray:
        M = eb.h.shape[0]
        t = np.zeros(M, dtype=float)

        for i in range(M):
            effect = float(c.T @ eb.h[i])
            var = float(c.T @ eb.H[i] @ c)

            if var <= 0 or not np.isfinite(var):
                t[i] = np.nan
            else:
                t[i] = effect / np.sqrt(var)

        return t

    def compute_second_level_T(self, eb: EBResult, X_star: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
        effect = float(d.T @ eb.B @ c)

        XtX_inv = np.linalg.pinv(X_star.T @ X_star)
        factor = float(d.T @ XtX_inv @ d)
        variance = factor * float(c.T @ eb.Omega @ c)

        if variance <= 0 or not np.isfinite(variance):
            return np.nan

        return effect / np.sqrt(variance)