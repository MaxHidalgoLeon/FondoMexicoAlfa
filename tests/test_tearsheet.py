"""Smoke tests for scripts/render_tearsheet.py."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
HTML    = ROOT / "reports" / "FMIA_Tearsheet.html"
PDF     = ROOT / "reports" / "FMIA_Tearsheet.pdf"


def _run_script():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "render_tearsheet.py")],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert result.returncode == 0, f"render_tearsheet.py exited with code {result.returncode}:\n{result.stderr}"


def test_html_exists_after_render():
    """HTML output file must exist and be non-empty after render."""
    _run_script()
    assert HTML.exists(), f"HTML not found: {HTML}"
    assert HTML.stat().st_size > 1_000, f"HTML suspiciously small: {HTML.stat().st_size} bytes"


def test_pdf_exists_after_render():
    """PDF output file must exist and be at least 50 KB."""
    assert PDF.exists(), f"PDF not found: {PDF}"
    assert PDF.stat().st_size > 50_000, f"PDF too small: {PDF.stat().st_size} bytes"


def test_no_external_image_refs():
    """HTML must not contain local file-path <img src> references."""
    assert HTML.exists(), "HTML not found — run test_html_exists_after_render first"
    text = HTML.read_text(encoding="utf-8")
    # Any src that is not a data URI is a local/external ref
    bad = re.findall(r'<img[^>]+src=["\'](?!data:image)[^"\']+["\']', text)
    assert not bad, f"External/local image refs found in HTML:\n" + "\n".join(bad)
