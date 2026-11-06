"""Lint the Galaxy XML with Galaxy's own parser and linters."""

from pathlib import Path

from galaxy.tool_util.lint import lint_tool_source
from galaxy.tool_util.parser.factory import get_tool_source


ROOT = Path(__file__).resolve().parents[1]
source = get_tool_source(config_file=ROOT / "fmri_reproducibility.xml")
if not lint_tool_source(source):
    raise SystemExit(1)
