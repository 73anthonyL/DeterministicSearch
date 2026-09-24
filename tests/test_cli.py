"""Tests for the CLI: exit codes, output content, and error handling.

Embedding is stubbed so these run offline and fast.
"""

from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from dsearch import index, search
from dsearch.cli import app

BODY = (
    "The clinic dismissed his knee pain without an examination. "
    "He returned to the strawberry rows before dawn the next morning. "
    "A foreman watched from the edge of the field and said nothing at all."
)

runner = CliRunner()


def fake_embed(texts, tier=index.DEFAULT_TIER, *, pages=None, page_count=None, progress=None):
    """Deterministic hashed bag-of-words, so ranking is stable without a model."""
    dim = 16
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in text.lower().split():
            out[row, hash(token) % dim] += 1.0
    if progress and page_count:
        progress(page_count, page_count)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-9)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("DSEARCH_HOME", str(tmp_path / "lib"))
    monkeypatch.setattr(index, "embed_texts", fake_embed)
    monkeypatch.setattr(search, "embed_texts", fake_embed)
    # Fixed width so panel text does not wrap unpredictably in assertions.
    monkeypatch.setenv("COLUMNS", "200")


@pytest.fixture
def book(pdf_factory):
    return pdf_factory(
        name="ffbb.pdf",
        pages=[f"{BODY} Page {n} of the chapter." for n in range(1, 6)],
        folios=[str(n + 46) for n in range(1, 6)],
    )


@pytest.fixture
def added(book):
    result = runner.invoke(
        app, ["add", str(book), "--author", "Seth Holmes", "--title", "Fresh Fruit, Broken Bodies"]
    )
    assert result.exit_code == 0, result.output
    return book


class TestHelpAndVersion:
    def test_help_lists_every_command(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for command in ("add", "edit", "list", "remove", "search"):
            assert command in result.output

    def test_version(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "dsearch" in result.output

    def test_no_args_shows_help_not_a_traceback(self):
        result = runner.invoke(app, [])
        assert "Usage" in result.output


class TestAdd:
    def test_reports_what_it_indexed(self, book):
        result = runner.invoke(app, ["add", str(book), "--title", "Fresh Fruit"])
        assert result.exit_code == 0
        assert "Indexed" in result.output
        assert "Fresh Fruit" in result.output
        assert "5 pages" in result.output

    def test_second_add_says_already_indexed(self, added):
        result = runner.invoke(app, ["add", str(added)])
        assert result.exit_code == 0
        assert "Already indexed" in result.output

    def test_missing_file_exits_with_a_clean_error(self, tmp_path):
        result = runner.invoke(app, ["add", str(tmp_path / "nope.pdf")])
        assert result.exit_code != 0
        assert "Traceback" not in result.output

    def test_scanned_pdf_explains_the_problem(self, pdf_factory):
        path = pdf_factory(name="scan.pdf", pages=[""] * 5)
        result = runner.invoke(app, ["add", str(path)])
        assert result.exit_code == 1
        assert "scan" in result.output.lower()
        assert "Traceback" not in result.output

    def test_unknown_tier_is_rejected(self, book):
        result = runner.invoke(app, ["add", str(book), "--tier", "turbo"])
        assert result.exit_code == 1
        assert "Unknown tier" in result.output

    def test_accepts_chunking_options(self, book):
        result = runner.invoke(app, ["add", str(book), "--chunk-size", "2", "--overlap", "0"])
        assert result.exit_code == 0
        assert index.list_sources()[0].chunk_size == 2


class TestList:
    def test_empty_library(self):
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "empty" in result.output

    def test_shows_the_indexed_source(self, added):
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "Fresh Fruit" in result.output
        assert "Seth Holmes" in result.output
        assert "fast" in result.output

    def test_marks_an_unknown_author(self, book):
        runner.invoke(app, ["add", str(book)])
        assert "unknown" in runner.invoke(app, ["list"]).output


class TestEdit:
    def test_updates_the_citation(self, book):
        runner.invoke(app, ["add", str(book)])
        result = runner.invoke(
            app,
            ["edit", "ffbb", "--author", "Seth Holmes", "--title", "Fresh Fruit, Broken Bodies"],
        )
        assert result.exit_code == 0, result.output
        assert "Updated" in result.output
        out = runner.invoke(app, ["search", "knee pain clinic", "--k", "1"]).output
        assert "Holmes, Seth. *Fresh Fruit, Broken Bodies*. p. 4" in out

    def test_nothing_to_change_is_a_clean_error(self, added):
        result = runner.invoke(app, ["edit", "ffbb"])
        assert result.exit_code == 1
        assert "Nothing to change" in result.output

    def test_unknown_source_is_a_clean_error(self, added):
        result = runner.invoke(app, ["edit", "nope", "--title", "x"])
        assert result.exit_code == 1
        assert "Traceback" not in result.output


class TestRemove:
    def test_removes_and_confirms(self, added):
        source_id = index.list_sources()[0].short_id
        result = runner.invoke(app, ["remove", source_id])
        assert result.exit_code == 0
        assert "Removed" in result.output
        assert index.list_sources() == []

    def test_unknown_source_is_a_clean_error(self, added):
        result = runner.invoke(app, ["remove", "no-such-source"])
        assert result.exit_code == 1
        assert "Traceback" not in result.output


class TestSearch:
    def test_returns_panels_with_citation_and_page(self, added):
        result = runner.invoke(app, ["search", "knee pain clinic"])
        assert result.exit_code == 0
        assert "Fresh Fruit, Broken Bodies" in result.output
        assert "Holmes, Seth." in result.output
        assert "p. 4" in result.output  # Printed folios are 47..51.

    def test_honours_k(self, added):
        few = runner.invoke(app, ["search", "strawberry rows", "--k", "1"]).output
        many = runner.invoke(app, ["search", "strawberry rows", "--k", "4"]).output
        assert many.count("Holmes, Seth.") > few.count("Holmes, Seth.")

    def test_search_deeper_path(self, added):
        result = runner.invoke(app, ["search", "field", "--k", "20"])
        assert result.exit_code == 0

    def test_empty_library_says_so(self):
        result = runner.invoke(app, ["search", "anything"])
        assert result.exit_code == 0
        assert "No passages found" in result.output

    def test_filter_by_source(self, added):
        result = runner.invoke(app, ["search", "knee pain", "--source", "ffbb"])
        assert result.exit_code == 0
        assert "Fresh Fruit" in result.output

    def test_unknown_source_filter_is_a_clean_error(self, added):
        result = runner.invoke(app, ["search", "knee", "--source", "no-such-book"])
        assert result.exit_code == 1
        assert "Traceback" not in result.output

    def test_zero_k_is_rejected(self, added):
        result = runner.invoke(app, ["search", "knee", "--k", "0"])
        assert result.exit_code == 1

    def test_accepts_a_paragraph_query(self, added):
        paragraph = (
            "I am trying to work out how the book handles the relationship between "
            "bodily pain and the structure of the farm. " * 12
        )
        result = runner.invoke(app, ["search", paragraph, "--k", "2"])
        assert result.exit_code == 0
        assert "Holmes, Seth." in result.output

    def test_output_is_verbatim_source_text(self, added):
        # Nothing in a result may be text the tool wrote itself.
        result = runner.invoke(app, ["search", "foreman field", "--k", "1"])
        assert "foreman watched" in result.output.replace("\n", " ")


class TestStaleIndex:
    def test_search_rebuilds_an_old_index_and_says_why(self, added):
        meta = index.list_sources()[0]
        stored = meta.to_dict()
        del stored["index_version"]
        index._write_json(index.source_dir(meta.source_id) / "meta.json", stored)
        index.save_library([index.SourceMeta.from_dict(stored)])

        result = runner.invoke(app, ["search", "knee pain clinic"])
        assert result.exit_code == 0, result.output
        assert "re-extracted" in result.output
        assert index.list_sources()[0].index_version == index.INDEX_VERSION
        assert "Holmes, Seth." in result.output  # And the search still ran.
