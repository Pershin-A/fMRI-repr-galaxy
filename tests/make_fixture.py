"""Create a deterministic synthetic subject; no human data are used."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test-data" / "synthetic_subject.zip"
SUBJECT = "synthetic_01"


def main() -> None:
    rng = np.random.default_rng(20261106)
    with tempfile.TemporaryDirectory() as tmp:
        subject = Path(tmp) / SUBJECT
        (subject / "runs").mkdir(parents=True)
        (subject / "logs").mkdir()
        shape = (12, 12, 12, 60)
        mask = np.zeros(shape[:3], dtype=np.uint8)
        mask[:11, :11, :11] = 1
        nib.save(nib.Nifti1Image(mask, np.eye(4)), subject / "brain_mask.nii.gz")
        runs = []
        x_star = []
        for index, kind in enumerate(("fr", "fr", "fr", "me", "me", "me"), start=1):
            run_id = f"{kind}_{(index - 1) % 3 + 1}"
            data = rng.normal(100, 1.0, size=shape).astype(np.float32)
            event_lines = ["onset\tduration\ttrial_type\tstimulus_name"]
            for event in range(12):
                condition = "self" if event % 2 else "other"
                onset = 4 + 9 * event
                event_lines.append(f"{onset}\t3\t{condition}\tsynthetic_{event:02d}")
                frame = min(shape[-1] - 1, round(onset / 2))
                data[2:10, 2:10, 2:10, frame:frame + 2] += (0.5 if condition == "self" else -0.5)
            bold_dir = subject / "runs" / run_id
            bold_dir.mkdir()
            nib.save(nib.Nifti1Image(data, np.eye(4)), bold_dir / "bold.nii.gz")
            (subject / "logs" / f"{run_id}.tsv").write_text("\n".join(event_lines) + "\n", encoding="utf-8")
            runs.append({
                "run_id": run_id,
                "bold": f"runs/{run_id}/bold.nii.gz",
                "log": f"logs/{run_id}.tsv",
                "tr_sec": 2.0,
                "slice_timing": {"available": True, "values": [0.0] * shape[2]},
            })
            x_star.append([1, -1 if kind == "fr" else 1])
        manifest = {
            "schema_version": "1.2",
            "subject_id": SUBJECT,
            "runs": runs,
            "events": {"onset_column": "onset", "duration_column": "duration", "condition_column": "trial_type", "stimulus_column": "stimulus_name"},
            "first_level_design": {"type": "condition_level_glm", "conditions": ["other", "self"], "include_intercept": True, "include_drift_terms": False},
            "first_level_contrasts": {"self_vs_other": {"columns": ["other", "self"], "values": [-1, 1], "auto_pad_nuisance": True}},
            "x_star": {"columns": ["average", "run_type"], "values": x_star},
            "second_level_contrasts": {"average": {"columns": ["average", "run_type"], "values": [1, 0]}},
            "primary_second_level_contrast": "average",
        }
        (subject / "subject.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        OUT.parent.mkdir(exist_ok=True)
        with ZipFile(OUT, "w", ZIP_DEFLATED, compresslevel=6) as archive:
            for file in sorted(subject.rglob("*")):
                if file.is_file():
                    archive.write(file, file.relative_to(subject.parent))
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
