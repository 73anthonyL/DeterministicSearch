"""Tests for the CLI: exit codes, output content, and error handling.

Embedding is stubbed so these run offline and fast.
"""

from __future__ import annotations

import zlib

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
            # Not the built-in hash(): that is salted per process, which made
            # the ranking, and so the asserted page, vary from run to run.
            out[row, zlib.crc32(token.encode()) % dim] += 1.0
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
        for command in ("add", "edit", "list", "remove", "search", "terms"):
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


class TestMultiAdd:
    """Several PDFs in one call, processed in order."""

    @pytest.fixture
    def shelf(self, pdf_factory):
        return [
            pdf_factory(name=f"book{n}.pdf", pages=[f"{BODY} Book number {n}."] * 3)
            for n in (1, 2, 3)
        ]

    def test_indexes_every_file(self, shelf):
        result = runner.invoke(app, ["add", *map(str, shelf)])
        assert result.exit_code == 0, result.output
        assert result.output.count("Indexed") == 3
        assert [s.filename for s in index.list_sources()] == ["book1.pdf", "book2.pdf", "book3.pdf"]

    def test_expands_a_quoted_glob(self, shelf):
        pattern = str(shelf[0].parent / "book*.pdf")
        result = runner.invoke(app, ["add", pattern])
        assert result.exit_code == 0, result.output
        assert len(index.list_sources()) == 3

    def test_glob_matching_nothing_is_a_clean_error(self, tmp_path):
        result = runner.invoke(app, ["add", str(tmp_path / "nothing*.pdf")])
        assert result.exit_code == 1
        assert "No files match" in result.output

    def test_already_indexed_files_are_skipped_with_a_note(self, shelf):
        runner.invoke(app, ["add", str(shelf[0])])
        result = runner.invoke(app, ["add", *map(str, shelf)])
        assert result.exit_code == 0, result.output
        assert result.output.count("Already indexed") == 1
        assert result.output.count("Indexed") == 2  # "Already indexed" does not contain "Indexed".

    def test_rejects_author_and_title_for_several_files(self, shelf):
        result = runner.invoke(app, ["add", *map(str, shelf), "--author", "Seth Holmes"])
        assert result.exit_code == 1
        assert "dsearch edit" in result.output
        assert index.list_sources() == []

    def test_one_bad_file_does_not_stop_the_rest(self, shelf, pdf_factory):
        scan = pdf_factory(name="scan.pdf", pages=[""] * 5)
        result = runner.invoke(app, ["add", str(shelf[0]), str(scan), str(shelf[1])])
        assert result.exit_code == 1
        assert "scan" in result.output.lower()
        assert len(index.list_sources()) == 2

    def test_metadata_comes_from_each_pdf(self, pdf_factory):
        one = pdf_factory(name="one.pdf", pages=[BODY] * 3, metadata={"author": "A. One"})
        two = pdf_factory(name="two.pdf", pages=[f"{BODY} Two."] * 3, metadata={"author": "B. Two"})
        runner.invoke(app, ["add", str(one), str(two)])
        assert [s.author for s in index.list_sources()] == ["A. One", "B. Two"]


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


class TestLazyTierEmbedding:
    """Searching at a tier the library lacks asks first, then embeds only that tier."""

    def test_prompt_names_sources_pages_estimate_and_alternative(self, added):
        result = runner.invoke(app, ["search", "knee pain", "--tier", "balanced"], input="n\n")
        assert result.exit_code == 0, result.output
        out = result.output
        assert "1 of 1 source(s) have no balanced vectors" in out
        assert "Fresh Fruit, Broken Bodies (5 pages)" in out
        assert "5 pages in all" in out
        assert "would take about" in out
        assert "--tier fast" in out
        assert "[y/N]" in out
        assert "Nothing embedded" in out
        assert not index.vectors_path(index.list_sources()[0].source_id, "balanced").is_file()

    def test_default_answer_is_no(self, added):
        result = runner.invoke(app, ["search", "knee pain", "--tier", "balanced"], input="\n")
        assert "Nothing embedded" in result.output
        assert "Holmes, Seth." not in result.output

    def test_non_interactive_stdin_counts_as_no(self, added):
        result = runner.invoke(app, ["search", "knee pain", "--tier", "balanced"], input="")
        assert result.exit_code == 0
        assert "Nothing embedded" in result.output

    def test_yes_embeds_only_the_missing_tier_and_searches(self, added):
        meta = index.list_sources()[0]
        fast = index.vectors_path(meta.source_id, "fast").read_bytes()
        result = runner.invoke(app, ["search", "knee pain", "--tier", "balanced"], input="y\n")
        assert result.exit_code == 0, result.output
        assert "Embedded" in result.output
        assert "Holmes, Seth." in result.output
        assert index.vectors_path(meta.source_id, "balanced").is_file()
        assert index.vectors_path(meta.source_id, "fast").read_bytes() == fast

    def test_yes_flag_skips_the_prompt(self, added):
        result = runner.invoke(app, ["search", "knee pain", "--tier", "balanced", "--yes"])
        assert result.exit_code == 0, result.output
        assert "[y/N]" not in result.output
        assert "Holmes, Seth." in result.output

    def test_no_prompt_once_the_tier_exists(self, added):
        runner.invoke(app, ["search", "knee pain", "--tier", "balanced", "--yes"])
        result = runner.invoke(app, ["search", "knee pain", "--tier", "balanced"])
        assert "[y/N]" not in result.output
        assert "Holmes, Seth." in result.output

    def test_no_alternative_named_when_none_is_shared(self, added, pdf_factory):
        other = pdf_factory(name="other.pdf", pages=[f"{BODY} Other book."] * 3)
        runner.invoke(app, ["add", str(other), "--tier", "best"])
        result = runner.invoke(app, ["search", "knee", "--tier", "balanced"], input="n\n")
        assert "2 of 2 source(s)" in result.output
        assert "already have" not in result.output


class TestEmbedCommand:
    def test_pre_warms_a_tier_across_the_library(self, added, pdf_factory):
        other = pdf_factory(name="other.pdf", pages=[f"{BODY} Other book."] * 3)
        runner.invoke(app, ["add", str(other)])
        result = runner.invoke(app, ["embed", "--tier", "balanced"])
        assert result.exit_code == 0, result.output
        assert result.output.count("Embedded") == 2
        assert all("balanced" in index.available_tiers(s.source_id) for s in index.list_sources())

    def test_limits_to_named_sources(self, added, pdf_factory):
        other = pdf_factory(name="other.pdf", pages=[f"{BODY} Other book."] * 3)
        runner.invoke(app, ["add", str(other)])
        result = runner.invoke(app, ["embed", "--tier", "balanced", "--source", "other"])
        assert result.exit_code == 0, result.output
        tiers = {s.filename: index.available_tiers(s.source_id) for s in index.list_sources()}
        assert tiers == {"ffbb.pdf": ["fast"], "other.pdf": ["fast", "balanced"]}

    def test_nothing_to_do_when_already_present(self, added):
        result = runner.invoke(app, ["embed", "--tier", "fast"])
        assert result.exit_code == 0
        assert "Nothing to do" in result.output

    def test_empty_library(self):
        result = runner.invoke(app, ["embed", "--tier", "fast"])
        assert result.exit_code == 0
        assert "empty" in result.output

    def test_list_shows_tiers_per_source(self, added):
        runner.invoke(app, ["embed", "--tier", "best"])
        out = runner.invoke(app, ["list"]).output
        assert "fast, best" in out


FARM_BODY = (
    "The crew boss watched the pickers. "
    "Pickers filled each strawberry flat. "
    "The crew boss weighed the strawberry flat. "
    "Workers rested at noon."
)
CLINIC_BODY = (
    "The physician examined the patient. "
    "A patient waited at the clinic. "
    "The physician left the clinic early. "
    "Workers rested at noon."
)


class TestTerms:
    @pytest.fixture
    def chapters(self, pdf_factory):
        """One book added as two single-chapter PDFs."""
        for name, body in (("farm.pdf", FARM_BODY), ("clinic.pdf", CLINIC_BODY)):
            path = pdf_factory(name=name, pages=[body] * 3, folios=["46", "47", "48"])
            result = runner.invoke(
                app, ["add", str(path), "--author", "Seth Holmes", "--title", "Fresh Fruit"]
            )
            assert result.exit_code == 0, result.output

    def test_lists_terms_under_each_chapter(self, chapters):
        result = runner.invoke(app, ["terms"])
        assert result.exit_code == 0, result.output
        assert "Fresh Fruit" in result.output
        assert "Seth Holmes" in result.output
        assert result.output.index("farm.pdf") < result.output.index("pickers")
        assert result.output.index("clinic.pdf") < result.output.index("physician")

    def test_shows_the_count_and_the_densest_page(self, chapters):
        result = runner.invoke(app, ["terms", "--source", "clinic.pdf"])
        row = next(line for line in result.output.splitlines() if "physician" in line)
        assert "6" in row
        assert "p. 46 (PDF 1)" in row

    def test_leaves_out_terms_shared_by_every_chapter(self, chapters):
        assert "Workers" not in runner.invoke(app, ["terms"]).output

    def test_says_which_tier_reranked(self, chapters):
        assert "re-ranked at tier fast" in runner.invoke(app, ["terms"]).output

    def test_source_limits_the_chapters_shown(self, chapters):
        result = runner.invoke(app, ["terms", "--source", "clinic.pdf"])
        assert result.exit_code == 0
        assert "physician" in result.output
        assert "pickers" not in result.output

    def test_top_limits_terms_per_chapter(self, chapters):
        result = runner.invoke(app, ["terms", "--source", "clinic.pdf", "--top", "1"])
        assert result.output.count("(PDF ") == 1

    def test_no_rerank_embeds_nothing(self, chapters, monkeypatch):
        def explode(*args, **kwargs):
            raise AssertionError("--no-rerank must not embed anything")

        monkeypatch.setattr(index, "embed_texts", explode)
        result = runner.invoke(app, ["terms", "--no-rerank"])
        assert result.exit_code == 0, result.output
        assert "ranked by word counts" in result.output

    def test_a_missing_tier_is_noted_not_embedded(self, chapters, monkeypatch):
        def explode(*args, **kwargs):
            raise AssertionError("a missing tier must not start an embedding run")

        monkeypatch.setattr(index, "embed_texts", explode)
        result = runner.invoke(app, ["terms", "--tier", "balanced"])
        assert result.exit_code == 0, result.output
        assert "dsearch embed --tier balanced" in result.output
        assert "physician" in result.output

    def test_a_single_chapter_book_says_it_is_ranked_by_frequency(self, added):
        result = runner.invoke(app, ["terms"])
        assert result.exit_code == 0, result.output
        assert "single chapter" in result.output

    def test_empty_library(self):
        result = runner.invoke(app, ["terms"])
        assert result.exit_code == 0
        assert "empty" in result.output

    def test_unknown_source_is_a_clean_error(self, chapters):
        result = runner.invoke(app, ["terms", "--source", "nope"])
        assert result.exit_code == 1
        assert "Traceback" not in result.output

    def test_unknown_tier_is_a_clean_error(self, chapters):
        result = runner.invoke(app, ["terms", "--tier", "turbo"])
        assert result.exit_code == 1
        assert "Unknown tier" in result.output

    def test_non_positive_top_is_a_clean_error(self, chapters):
        result = runner.invoke(app, ["terms", "--top", "0"])
        assert result.exit_code == 1
        assert "top must be at least 1" in result.output
