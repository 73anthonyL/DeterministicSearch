"""Tests for the CLI: exit codes, output content, and error handling.

Embedding is stubbed so these run offline and fast.
"""

from __future__ import annotations

import json
import sys
import types
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
        for command in ("add", "edit", "list", "remove", "search", "terms", "ui"):
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
        # Typer validates the choice before the command runs: a usage error, exit 2.
        result = runner.invoke(app, ["add", str(book), "--tier", "ultra"])
        assert result.exit_code == 2
        assert "fast" in result.output
        assert "Traceback" not in result.output

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

    @pytest.fixture
    def chapters(self, pdf_factory):
        """One book added as two PDFs that share a title, as the README suggests."""
        for n in (1, 2):
            path = pdf_factory(name=f"ch{n}.pdf", pages=[f"{BODY} Chapter {n} knee."] * 2)
            result = runner.invoke(app, ["add", str(path), "--title", "Shared Title"])
            assert result.exit_code == 0, result.output

    def test_filter_by_a_shared_title_searches_the_whole_book(self, chapters):
        result = runner.invoke(app, ["search", "knee pain", "--source", "Shared Title", "-k", "4"])
        assert result.exit_code == 0, result.output
        assert "matches several" not in result.output
        assert "ch1.pdf" not in result.output  # the panel names the title, not the file
        assert result.output.count("Shared Title") >= 2

    def test_filter_by_filename_within_a_shared_title(self, chapters):
        result = runner.invoke(app, ["search", "knee pain", "--source", "ch2.pdf"])
        assert result.exit_code == 0, result.output
        assert "Chapter 2" in result.output
        assert "Chapter 1" not in result.output

    def test_zero_k_is_rejected(self, added):
        # Validated by the option itself (min=1): a usage error, exit 2, before any work.
        result = runner.invoke(app, ["search", "knee", "--k", "0"])
        assert result.exit_code == 2
        assert "Traceback" not in result.output

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


class TestUi:
    def test_launches_streamlit_on_the_packaged_app(self, monkeypatch):
        calls: list[list[str]] = []
        monkeypatch.setattr("dsearch.cli.subprocess.call", lambda cmd: calls.append(cmd) or 0)
        result = runner.invoke(app, ["ui", "--port", "8600", "--headless"])
        assert result.exit_code == 0, result.output
        (command,) = calls
        assert command[1:4] == ["-m", "streamlit", "run"]
        assert command[4].endswith("app.py")
        assert command[5:] == ["--server.port", "8600", "--server.headless", "true"]

    def test_streamlit_exit_code_is_passed_through(self, monkeypatch):
        monkeypatch.setattr("dsearch.cli.subprocess.call", lambda cmd: 3)
        assert runner.invoke(app, ["ui"]).exit_code == 3

    def test_missing_streamlit_is_a_one_line_hint(self, monkeypatch):
        monkeypatch.setattr("dsearch.cli.importlib.util.find_spec", lambda name: None)
        result = runner.invoke(app, ["ui"])
        assert result.exit_code == 1
        assert "dsearch[ui]" in result.output
        assert "Traceback" not in result.output


class TestJsonOutput:
    def test_list_json(self, added):
        result = runner.invoke(app, ["list", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["sources"][0]["title"] == "Fresh Fruit, Broken Bodies"
        assert payload["sources"][0]["tiers_available"] == ["fast"]
        assert "library" in payload

    def test_list_json_of_an_empty_library(self):
        result = runner.invoke(app, ["list", "--json"])
        assert result.exit_code == 0
        assert json.loads(result.stdout)["sources"] == []

    def test_search_json_carries_passage_page_and_citations(self, added):
        result = runner.invoke(app, ["search", "knee pain", "--json", "-k", "2"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert len(payload) == 2
        first = payload[0]
        assert first["source"]["title"] == "Fresh Fruit, Broken Bodies"
        assert first["page_label"].startswith("p. ")
        assert first["best_sentence_text"] in first["chunk"]["text"]
        assert first["citation"]["mla"].startswith("Holmes, Seth.")
        assert first["citation"]["in_text"].startswith("(Holmes")

    def test_search_json_with_no_results_is_an_empty_list(self):
        result = runner.invoke(app, ["search", "knee", "--json"])  # empty library
        assert result.exit_code == 0
        assert json.loads(result.stdout) == []

    def test_terms_json(self, added):
        result = runner.invoke(app, ["terms", "--json", "--no-rerank"])
        assert result.exit_code == 0, result.output
        books = json.loads(result.stdout)
        assert books[0]["title"] == "Fresh Fruit, Broken Bodies"
        assert "page_label" in books[0]["chapters"][0]["terms"][0]

    def test_notes_do_not_pollute_json_stdout(self, added, monkeypatch):
        # A skipped source produces a note; it must go to stderr.
        monkeypatch.setattr(
            "dsearch.cli.load_corpus",
            lambda tier, source=None: _corpus_with_skip(tier, source),
        )
        result = runner.invoke(app, ["search", "knee", "--json"])
        assert result.exit_code == 0, result.output
        json.loads(result.stdout)  # parses despite the note
        assert "Skipped" in result.output


def _corpus_with_skip(tier, source):
    corpus = search.load_corpus(tier, source)
    corpus.skipped.append("Ghost Book")
    return corpus


class TestDoctor:
    def test_prints_versions_paths_and_models(self, book):
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0, result.output
        for word in ("dsearch", "python", "library", "model cache", "fast", "device", "pymupdf"):
            assert word in result.output

    def test_json(self):
        result = runner.invoke(app, ["doctor", "--json"])
        assert result.exit_code == 0
        info = json.loads(result.stdout)
        assert set(info["models"]["tiers"]) == {"fast", "balanced", "best"}


class TestVerbosity:
    def test_verbose_logs_pipeline_steps(self, added):
        result = runner.invoke(app, ["-v", "search", "knee"])
        assert result.exit_code == 0, result.output
        assert "corpus:" in result.output  # the DEBUG line from load_corpus

    def test_env_var_does_the_same(self, added, monkeypatch):
        monkeypatch.setenv("DSEARCH_LOG", "DEBUG")
        result = runner.invoke(app, ["search", "knee"])
        assert "corpus:" in result.output

    def test_silent_by_default(self, added):
        result = runner.invoke(app, ["search", "knee"])
        assert "corpus:" not in result.output

    def test_quiet_drops_notes_but_not_errors(self, added, monkeypatch):
        monkeypatch.setattr(
            "dsearch.cli.load_corpus",
            lambda tier, source=None: _corpus_with_skip(tier, source),
        )
        result = runner.invoke(app, ["-q", "search", "knee"])
        assert result.exit_code == 0
        assert "Skipped" not in result.output
        result = runner.invoke(app, ["-q", "search", "knee", "--source", "nope"])
        assert result.exit_code == 1
        assert "No source matches" in result.output


class TestModelUnavailable:
    @pytest.fixture
    def no_model(self, monkeypatch):
        """A hub that cannot be reached, behind the real `load_model` path."""

        class Boom:
            def __init__(self, name):
                raise OSError("We couldn't connect to 'https://huggingface.co'")

        monkeypatch.setitem(
            sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=Boom)
        )
        index._MODEL_CACHE.clear()

        def embed_via_model(texts, tier=index.DEFAULT_TIER, **kwargs):
            index.load_model(tier)
            raise AssertionError("load_model should have failed")

        monkeypatch.setattr(index, "embed_texts", embed_via_model)

    def test_add_fails_with_one_line(self, book, no_model):
        result = runner.invoke(app, ["add", str(book)])
        assert result.exit_code == 1
        assert "Could not load the fast model" in result.output
        assert "Traceback" not in result.output

    def test_offline_flag_is_named(self, book, no_model, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        result = runner.invoke(app, ["add", str(book)])
        assert "HF_HUB_OFFLINE=1" in result.output


class TestErrorText:
    """Error messages carry filenames and user input, which must print as typed."""

    def test_brackets_in_a_missing_pattern_are_not_markup(self, tmp_path):
        result = runner.invoke(app, ["add", str(tmp_path / "[draft].pdf")])
        assert result.exit_code == 1
        assert "[draft].pdf" in result.output

    def test_brackets_in_an_unknown_source_are_not_markup(self, added):
        result = runner.invoke(app, ["edit", "[bold]x[/bold]", "--author", "A"])
        assert result.exit_code == 1
        assert "[bold]x[/bold]" in result.output

    def test_brackets_in_a_per_file_add_error_are_not_markup(self, book, monkeypatch):
        def boom(*args, **kwargs):
            raise ValueError("bad [red]glyph[/red] table")

        monkeypatch.setattr("dsearch.cli.index_add", boom)
        result = runner.invoke(app, ["add", str(book)])
        assert result.exit_code == 1
        assert "bad [red]glyph[/red] table" in result.output


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
        assert result.exit_code == 2
        assert "Traceback" not in result.output

    def test_non_positive_top_is_a_clean_error(self, chapters):
        result = runner.invoke(app, ["terms", "--top", "0"])
        assert result.exit_code == 2
        assert "Traceback" not in result.output
