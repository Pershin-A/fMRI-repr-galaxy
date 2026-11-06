"""Run the fMRI reproducibility pipeline as a batch Galaxy job."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile

import yaml


SUBJECT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def safe_member_name(name: str, subject_id: str) -> PurePosixPath:
    name = name.replace("\\", "/")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or ":" in name
        or ".." in path.parts
        or not path.parts
        or path.parts[0] != subject_id
    ):
        raise ValueError(f"Unsafe or unexpected ZIP member: {name}")
    return path


def safe_manifest_path(name: str, subject_id: str) -> PurePosixPath:
    relative = PurePosixPath(name.replace("\\", "/"))
    if relative.is_absolute() or ":" in name or ".." in relative.parts:
        raise ValueError(f"Unsafe manifest path: {name}")
    return safe_member_name(f"{subject_id}/{relative}", subject_id)


def extract_subject(archive: Path, destination: Path, subject_id: str) -> Path:
    with ZipFile(archive) as source:
        names = [item.filename for item in source.infolist()]
        if len(names) != len(set(names)):
            raise ValueError("ZIP contains duplicate member names")
        for item in source.infolist():
            relative = safe_member_name(item.filename, subject_id)
            # Symlinks in a ZIP must not be followed or materialized.
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"ZIP contains a symlink: {item.filename}")
            target = destination.joinpath(*relative.parts)
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(item) as inp, target.open("wb") as out:
                    shutil.copyfileobj(inp, out)
    subject = destination / subject_id
    manifest_path = subject / "subject.json"
    if not manifest_path.is_file():
        raise ValueError("ZIP must contain <subject_id>/subject.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("subject_id") != subject_id:
        raise ValueError("subject.json subject_id differs from the selected subject ID")
    if not (subject / "brain_mask.nii.gz").is_file() and not (subject / "brain_mask.nii").is_file():
        raise ValueError("Subject ZIP must include brain_mask.nii.gz or brain_mask.nii")
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("subject.json must list at least one run")
    for run in runs:
        if not isinstance(run, dict) or not SUBJECT_ID.fullmatch(str(run.get("run_id", ""))):
            raise ValueError("Each run must have a safe run_id")
        for key in ("bold", "log"):
            relative = safe_manifest_path(str(run[key]), subject_id)
            if not destination.joinpath(*relative.parts).is_file():
                raise ValueError(f"Missing {key} file for run {run.get('run_id')}")
        preprocessing = run.get("preprocessing") or {}
        if isinstance(preprocessing, dict) and preprocessing.get("motion_parameters"):
            safe_manifest_path(str(preprocessing["motion_parameters"]), subject_id)
    return subject


def run(args: argparse.Namespace) -> None:
    if not SUBJECT_ID.fullmatch(args.subject_id) or args.subject_id in (".", ".."):
        raise ValueError("Subject ID must contain Latin letters, digits, _, . or -")
    tool_dir = Path(__file__).resolve().parent
    code_dir = tool_dir / "code"
    if not (code_dir / "main.py").is_file():
        raise FileNotFoundError("Tool package is incomplete: code/main.py is missing")
    with tempfile.TemporaryDirectory(prefix="fmri_galaxy_") as scratch:
        work = Path(scratch)
        data_root = work / "preprocessed"
        data_root.mkdir()
        extract_subject(args.input, data_root, args.subject_id)

        settings = yaml.safe_load((code_dir / "settings.yaml").read_text(encoding="utf-8"))
        settings["spm_preprocessing"]["enabled"] = False
        settings["logging"]["log_to_file"] = False
        settings["analysis"]["glm_method"] = args.glm_method
        settings["analysis"]["n_jobs"] = max(1, int(os.environ.get("GALAXY_SLOTS", "1")))
        settings["analysis"]["roc_n_restarts"] = args.roc_restarts
        settings["analysis"]["roc_restart_jobs"] = 1
        settings["analysis"]["roc_parallel_restarts"] = False
        if args.max_voxels:
            settings["analysis"]["max_voxels"] = args.max_voxels
        config_path = work / "settings.yaml"
        config_path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")

        result_root = work / "results"
        result_root.mkdir()
        command = [
            sys.executable, str(code_dir / "main.py"),
            "--config", str(config_path),
            "--patient-id", args.subject_id,
            "--data-path", str(data_root),
            "--logs-path", str(data_root),
            "--output-path", str(result_root),
        ]
        subprocess.run(command, cwd=work, check=True)
        summaries = sorted(result_root.rglob("summary.json"), key=lambda path: len(path.parts))
        if not summaries:
            raise RuntimeError("Analysis completed without summary.json")
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(summaries[0], args.summary)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(args.output, "w", ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(result_root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(result_root))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--max-voxels", type=int, default=0)
    parser.add_argument("--glm-method", choices=("bayes", "ols", "choose_best_kappa"), default="bayes")
    parser.add_argument("--roc-restarts", type=int, default=50)
    args = parser.parse_args()
    if args.max_voxels < 0 or args.roc_restarts < 1:
        parser.error("max-voxels must be >= 0 and roc-restarts must be >= 1")
    run(args)


if __name__ == "__main__":
    main()
