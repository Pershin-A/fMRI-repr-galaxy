"""Small checks for package inputs and a full synthetic analysis run."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_galaxy import extract_subject, safe_manifest_path  # noqa: E402


class GalaxyRunnerTests(unittest.TestCase):
    def test_rejects_archive_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "bad.zip"
            with ZipFile(archive, "w") as z:
                z.writestr("../escape.txt", "bad")
            with self.assertRaises(ValueError):
                extract_subject(archive, root / "input", "synthetic_01")

    def test_rejects_absolute_manifest_path(self) -> None:
        with self.assertRaises(ValueError):
            safe_manifest_path("/etc/passwd", "synthetic_01")
        with self.assertRaises(ValueError):
            safe_manifest_path("C:\\data\\bold.nii", "synthetic_01")

    def test_complete_synthetic_run(self) -> None:
        fixture = ROOT / "test-data" / "synthetic_subject.zip"
        self.assertTrue(fixture.is_file(), "Run tests/make_fixture.py first")
        with tempfile.TemporaryDirectory() as tmp:
            for method in ("ols", "bayes"):
                with self.subTest(method=method):
                    result = Path(tmp) / f"{method}_result.zip"
                    summary = Path(tmp) / f"{method}_summary.json"
                    subprocess.run([
                        sys.executable, str(ROOT / "run_galaxy.py"),
                        "--input", str(fixture), "--subject-id", "synthetic_01",
                        "--output", str(result), "--summary", str(summary),
                        "--max-voxels", "4", "--glm-method", method, "--roc-restarts", "2",
                    ], check=True, cwd=ROOT)
                    self.assertTrue(result.is_file())
                    report = json.loads(summary.read_text(encoding="utf-8"))
                    self.assertIn("synthetic_01", json.dumps(report))
                    with ZipFile(result) as z:
                        names = z.namelist()
                        self.assertTrue(any(name.endswith("summary.json") for name in names))
                        self.assertTrue(any(name.endswith(".nii.gz") for name in names))


if __name__ == "__main__":
    unittest.main()
