"""Make a self-contained archive for manual installation on Galaxy."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "fmri_reproducibility-galaxy-0.2.0.zip"
FILES = [ROOT / name for name in ("README.md", "fmri_reproducibility.xml", "run_galaxy.py", "requirements.txt", "tool_conf.xml.example")]
FILES += list((ROOT / "code").glob("*.py"))
FILES += list((ROOT / "tests").glob("*.py"))
FILES += [ROOT / "code" / "settings.yaml", ROOT / "test-data" / "synthetic_subject.zip"]


def main() -> None:
    missing = [str(path) for path in FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Package files missing: {missing}")
    with ZipFile(OUT, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(FILES):
            archive.write(path, Path("fmri_reproducibility") / path.relative_to(ROOT))
    print(f"Created {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
