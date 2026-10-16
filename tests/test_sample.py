"""The bundled sample: packaged, reproducible, and what `dsearch demo` indexes."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

from dsearch import sample
from dsearch.extract import extract

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_sample.py"


class TestPackagedSample:
    def test_sample_ships_inside_the_package(self):
        path = sample.sample_path()
        assert path.is_file()
        assert path.name == sample.SAMPLE_FILENAME
        assert "dsearch" in path.parts

    def test_copy_sample_is_idempotent(self, tmp_path: Path):
        first = sample.copy_sample(tmp_path / "demo")
        stamp = first.stat().st_mtime_ns
        second = sample.copy_sample(tmp_path / "demo")
        assert second == first
        assert second.stat().st_mtime_ns == stamp  # not rewritten

    def test_extracts_like_a_real_book(self):
        pages = extract(sample.sample_path())
        assert len(pages) > 20
        # Front matter has no arabic folio; the first chapter opens on folio 1.
        assert pages[0].printed_page is None
        first_body = next(p for p in pages if p.printed_page == 1)
        assert first_body.chapter is not None
        assert first_body.chapter.startswith("I.")
        assert {p.chapter for p in pages if p.chapter} >= {
            "I. Of Our Spiritual Strivings",
            "II. Of the Dawn of Freedom",
            "III. Of Mr. Booker T. Washington and Others",
        }
        joined = "\n".join(p.text for p in pages)
        assert "problem of the Twentieth Century is the problem of the color" in joined
        assert "—" in joined  # typographic characters survive the typesetting
        assert "·" not in joined


@pytest.mark.skipif(not SCRIPT.is_file(), reason="scripts/ is not in the installed package")
class TestGenerator:
    def test_committed_pdf_matches_the_script(self, tmp_path: Path):
        """Regenerating the sample must give the committed bytes, or the two have drifted."""
        spec = importlib.util.spec_from_file_location("make_sample", SCRIPT)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
        spec.loader.exec_module(module)
        out = tmp_path / "sample.pdf"
        pages = module.build(module.DEFAULT_TEXT, out)
        assert pages == len(extract(out))
        committed = hashlib.sha256(sample.sample_path().read_bytes()).hexdigest()
        regenerated = hashlib.sha256(out.read_bytes()).hexdigest()
        assert regenerated == committed, "run `uv run python scripts/make_sample.py` and commit"
