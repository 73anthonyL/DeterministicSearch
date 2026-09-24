"""Tests for MLA citation formatting."""

from __future__ import annotations

import pytest

from dsearch.chunk import Chunk
from dsearch.cite import (
    citation_block,
    format_author,
    mla,
    page_reference,
    parenthetical,
    quotation,
)
from dsearch.index import SourceMeta
from dsearch.search import Result

SENTENCE = "The clinic dismissed his knee pain."


def make_result(
    *,
    author="Seth Holmes",
    title="Fresh Fruit, Broken Bodies",
    printed=47,
    pdf_page=59,
    sentences=None,
    best=0,
):
    sentences = sentences or [SENTENCE, "No one examined the knee."]
    chunk = Chunk(
        chunk_id=0,
        source_id="abc",
        text=" ".join(sentences),
        sentences=sentences,
        pdf_page=pdf_page,
        printed_page=printed,
        chapter="Chapter 3",
    )
    meta = SourceMeta(
        source_id="abc",
        filename="ffbb.pdf",
        title=title,
        author=author,
        added="2026-09-24",
        page_count=100,
    )
    return Result(chunk=chunk, score=0.5, best_sentence=best, meta=meta)


class TestFormatAuthor:
    def test_inverts_first_and_last(self):
        assert format_author("Seth Holmes") == "Holmes, Seth"

    def test_keeps_middle_names_with_the_first(self):
        assert format_author("Seth M. Holmes") == "Holmes, Seth M."

    def test_leaves_an_already_inverted_name(self):
        assert format_author("Holmes, Seth") == "Holmes, Seth"

    def test_strips_a_trailing_period_from_an_inverted_name(self):
        assert format_author("Holmes, Seth.") == "Holmes, Seth"

    def test_single_name(self):
        assert format_author("Aristotle") == "Aristotle"

    def test_two_authors_invert_only_the_first(self):
        assert format_author("Seth Holmes and Philippe Bourgois") == (
            "Holmes, Seth, and Philippe Bourgois"
        )

    def test_three_or_more_authors_use_et_al(self):
        assert format_author("Seth Holmes and Philippe Bourgois and Nancy Scheper-Hughes") == (
            "Holmes, Seth, et al"
        )

    @pytest.mark.parametrize("raw", ["", "   ", None])
    def test_missing_author(self, raw):
        assert format_author(raw) == ""


class TestPageReference:
    def test_prefers_the_printed_folio(self):
        assert page_reference(make_result(printed=47, pdf_page=59).chunk) == "p. 47"

    def test_falls_back_to_the_pdf_page_and_labels_it(self):
        assert page_reference(make_result(printed=None, pdf_page=59).chunk) == "PDF p. 59"


class TestMla:
    def test_matches_the_specified_format(self):
        assert mla(make_result()) == "Holmes, Seth. *Fresh Fruit, Broken Bodies*. p. 47."

    def test_drops_a_missing_author_rather_than_faking_one(self):
        assert mla(make_result(author="")) == "*Fresh Fruit, Broken Bodies*. p. 47."

    def test_uses_the_pdf_page_when_no_folio_was_found(self):
        cite = mla(make_result(printed=None, pdf_page=59))
        assert cite.endswith("PDF p. 59.")

    def test_accepts_an_explicit_meta_override(self):
        result = make_result()
        other = SourceMeta(
            source_id="x",
            filename="x.pdf",
            title="Another Book",
            author="Jane Doe",
            added="2026-09-24",
            page_count=1,
        )
        assert mla(result, other) == "Doe, Jane. *Another Book*. p. 47."

    def test_title_is_italicised_for_markdown_renderers(self):
        assert "*Fresh Fruit, Broken Bodies*" in mla(make_result())


class TestQuotation:
    def test_wraps_the_best_sentence_in_typographic_quotes(self):
        assert quotation(make_result()) == f"“{SENTENCE}”"

    def test_quotes_the_sentence_that_actually_matched(self):
        assert quotation(make_result(best=1)) == "“No one examined the knee.”"

    def test_is_verbatim(self):
        # The guarantee of the whole tool: the quote is the source's own words.
        result = make_result()
        assert result.chunk.sentences[0] in quotation(result)


class TestParenthetical:
    def test_surname_and_page(self):
        assert parenthetical(make_result()) == "(Holmes 47)"

    def test_omits_page_when_no_folio_is_known(self):
        assert parenthetical(make_result(printed=None)) == "(Holmes)"

    def test_page_only_when_the_author_is_unknown(self):
        assert parenthetical(make_result(author="")) == "(47)"

    def test_empty_when_nothing_is_known(self):
        assert parenthetical(make_result(author="", printed=None)) == ""


class TestCitationBlock:
    def test_places_the_page_reference_outside_the_quotation(self):
        # MLA: closing quote, then the parenthetical, then the period.
        block = citation_block(make_result())
        assert block.splitlines()[0] == "“The clinic dismissed his knee pain” (Holmes 47)."

    def test_full_reference_on_the_second_line(self):
        assert citation_block(make_result()).splitlines()[1] == (
            "Holmes, Seth. *Fresh Fruit, Broken Bodies*. p. 47."
        )

    def test_handles_a_sentence_without_terminal_punctuation(self):
        block = citation_block(make_result(sentences=["A fragment with no period"]))
        assert block.splitlines()[0] == "“A fragment with no period” (Holmes 47)."

    def test_keeps_a_question_mark_inside_the_quotation(self):
        # MLA 9: a question mark belongs to the quoted words, so it stays inside
        # and a period follows the parenthetical. Only a period is moved out.
        block = citation_block(make_result(sentences=["Is it worth risking your life?"]))
        assert block.splitlines()[0] == "“Is it worth risking your life?” (Holmes 47)."

    def test_keeps_an_exclamation_inside_the_quotation(self):
        block = citation_block(make_result(sentences=["We are field workers!"]))
        assert block.splitlines()[0] == "“We are field workers!” (Holmes 47)."

    def test_falls_back_to_a_plain_quotation_when_nothing_is_known(self):
        block = citation_block(make_result(author="", printed=None))
        assert block.splitlines()[0] == f"“{SENTENCE}”"
