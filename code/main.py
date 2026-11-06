
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config import load_config
from pipeline import LiouReproducibilityPipeline


def silence_noisy_loggers() -> None:
    """
    Suppress extra DEBUG messages from matplotlib/PIL,
    especially when logging.level is: DEBUG.
    """
    noisy_loggers = [
        "matplotlib",
        "matplotlib.pyplot",
        "matplotlib.font_manager",
        "PIL",
        "PIL.PngImagePlugin",
    ]

    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)

def setup_logging(config) -> None:
    level = getattr(logging, config.logging.level.upper(), logging.INFO)

    handlers = [logging.StreamHandler(sys.stdout)]

    if config.logging.log_to_file:
        handlers.append(logging.FileHandler(config.logging.log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )
    
    silence_noisy_loggers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Liou fMRI reproducibility analysis")

    parser.add_argument(
        "--config",
        type=str,
        default="settings.yaml",
        help="Path to the YAML config. Default: settings.yaml next to main.py",
    )

    parser.add_argument("--data-path", type=str, default=None, help="Override paths.data_path")
    parser.add_argument("--logs-path", type=str, default=None, help="Override paths.logs_path")
    parser.add_argument("--output-path", type=str, default=None, help="Override paths.output_path")
    parser.add_argument("--patient-id", type=str, default=None, help="Patient ID if data_path/logs_path are root folders")
    parser.add_argument("--mode", type=str, default=None, choices=["+", "fr", "me"], help="Which runs to use: +, fr or me")
    parser.add_argument("--max-voxels", type=int, default=None, help="Limit the number of voxels for a test run")
    parser.add_argument("--n-jobs", type=int, default=None, help="Number of CPU workers for Python voxelwise computation")
    parser.add_argument(
        "--glm-method",
        type=str,
        default=None,
        choices=["bayes", "ols", "choose_best_kappa"],
        help="GLM method: bayes, ols or choose_best_kappa",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)

    if args.data_path is not None:
        config.paths.data_path = Path(args.data_path).resolve()

    if args.logs_path is not None:
        config.paths.logs_path = Path(args.logs_path).resolve()

    if args.output_path is not None:
        config.paths.output_path = Path(args.output_path).resolve()
        config.paths.output_path.mkdir(parents=True, exist_ok=True)

    if args.patient_id is not None:
        config.paths.patient_id = args.patient_id

    if args.mode is not None:
        config.data.run_mode = args.mode

    if args.max_voxels is not None:
        config.analysis.max_voxels = args.max_voxels

    if args.n_jobs is not None:
        config.analysis.n_jobs = args.n_jobs

    if args.glm_method is not None:
        config.analysis.glm_method = args.glm_method

    setup_logging(config)

    logging.getLogger(__name__).info("Starting Liou reproducibility pipeline")
    logging.getLogger(__name__).info("data_path=%s", config.paths.data_path)
    logging.getLogger(__name__).info("logs_path=%s", config.paths.logs_path)
    logging.getLogger(__name__).info("output_path=%s", config.paths.output_path)
    logging.getLogger(__name__).info("glm_method=%s", getattr(config.analysis, "glm_method", "bayes"))
    logging.getLogger(__name__).info("n_jobs=%s", getattr(config.analysis, "n_jobs", 1))

    pipeline = LiouReproducibilityPipeline(config)
    summary = pipeline.run()

    logging.getLogger(__name__).info("Done")
    logging.getLogger(__name__).info("Final summary saved to summary.json; the full dictionary is not printed to the log")


if __name__ == "__main__":
    main()