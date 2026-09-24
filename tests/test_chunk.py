"""Tests for sentence splitting, window math, and metadata propagation."""

from __future__ import annotations

import pytest

from dsearch.chunk import (
    Chunk,
    _clean_page_text,
    _dehyphenate,
    _running_heads,
    chunk,
    split_sentences,
)
from dsearch.extract import Page


def make_pages(*specs: tuple[str, ...]) -> list[Page]:
    """Build Pages from (text, [printed_page], [chapter]) tuples."""
    pages = []
    for index, spec in enumerate(specs, start=1):
        text = spec[0]
        printed = spec[1] if len(spec) > 1 else None
        chapter = spec[2] if len(spec) > 2 else None
        pages.append(Page(pdf_page=index, text=text, printed_page=printed, chapter=chapter))
    return pages


def sentences_page(count: int, prefix: str = "S") -> Page:
    """A page holding `count` trivially distinguishable sentences."""
    text = " ".join(f"{prefix}{n} is a sentence." for n in range(1, count + 1))
    return Page(pdf_page=1, text=text)


class TestSplitSentences:
    def test_splits_on_terminators(self):
        assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]

    def test_empty_and_whitespace(self):
        assert split_sentences("") == []
        assert split_sentences("   \n  ") == []

    def test_single_sentence_without_terminator(self):
        assert split_sentences("A fragment with no period") == ["A fragment with no period"]

    def test_collapses_internal_whitespace(self):
        assert split_sentences("One   sentence\n  here.") == ["One sentence here."]

    @pytest.mark.parametrize(
        "text",
        [
            "Dr. Holmes rode north.",
            "Mr. Field spoke.",
            "See Fig. 3 for the map.",
            "They arrived at approx. 4 in the morning.",
            "The vol. was missing.",
        ],
    )
    def test_does_not_split_on_abbreviations(self, text):
        assert split_sentences(text) == [text]

    def test_does_not_split_on_initials(self):
        assert split_sentences("J. R. Smith wrote it.") == ["J. R. Smith wrote it."]

    def test_splits_after_an_abbreviation_ends_a_sentence(self):
        # Known limitation of a regex splitter: a sentence genuinely ending in
        # an abbreviation is joined to the next. Documented, not silently wrong.
        result = split_sentences("He worked for the co. They left.")
        assert result == ["He worked for the co. They left."]

    def test_keeps_closing_quote_with_its_sentence(self):
        result = split_sentences('She said, "We are field workers." Then she bent down.')
        assert result == ['She said, "We are field workers."', "Then she bent down."]

    def test_does_not_split_mid_decimal(self):
        assert split_sentences("It fell 3.5 degrees overnight.") == [
            "It fell 3.5 degrees overnight."
        ]

    def test_ellipsis_stays_with_sentence(self):
        assert split_sentences("He paused... Then he spoke.") == [
            "He paused...",
            "Then he spoke.",
        ]

    def test_splits_before_a_number_starting_a_sentence(self):
        assert split_sentences("They waited. 1994 changed everything.") == [
            "They waited.",
            "1994 changed everything.",
        ]

    def test_does_not_split_on_lowercase_continuation(self):
        # A period followed by a lowercase word is not a sentence boundary.
        assert split_sentences("the U.S. border was closed.") == ["the U.S. border was closed."]


class TestDehyphenate:
    def test_rejoins_word_split_across_lines(self):
        assert _dehyphenate("compan-\nions") == "companions"

    def test_rejoins_with_surrounding_space(self):
        assert _dehyphenate("compan- \n ions") == "companions"

    def test_leaves_inline_hyphen_alone(self):
        assert _dehyphenate("well-being of workers") == "well-being of workers"

    def test_leaves_dash_at_line_end_without_word(self):
        assert _dehyphenate("a dash -\n") == "a dash -\n"


class TestRunningHeads:
    def test_detects_a_repeated_short_line(self):
        pages = make_pages(*[(f"introduction\nBody text {n} here.",) for n in range(1, 9)])
        assert "introduction" in _running_heads(pages)

    def test_ignores_lines_unique_to_one_page(self):
        pages = make_pages(*[(f"Unique line {n}\nBody text here.",) for n in range(1, 9)])
        assert _running_heads(pages) == {"Body text here."}

    def test_ignores_long_repeated_lines(self):
        long_line = "A repeated line that is far too long to be a running head in any book."
        pages = make_pages(*[(f"{long_line}\nBody {n}.",) for n in range(1, 9)])
        assert long_line not in _running_heads(pages)

    def test_too_few_pages_to_judge(self):
        pages = make_pages(("introduction\nBody.",), ("introduction\nBody.",))
        assert _running_heads(pages) == set()


class TestCleanPageText:
    def test_strips_folio_and_running_head(self):
        text = "introduction\n3\nThe workers rose before dawn."
        assert _clean_page_text(text, {"introduction"}) == "The workers rose before dawn."

    def test_reflows_hard_wrapped_lines(self):
        text = "The workers rose\nbefore dawn and\nwalked to the field."
        assert (
            _clean_page_text(text, set()) == "The workers rose before dawn and walked to the field."
        )

    def test_rejoins_hyphenated_word_across_lines(self):
        assert "companions" in _clean_page_text("my compan-\nions ate", set())


class TestChunkWindowing:
    def test_default_window_and_stride(self):
        # 5 sentences, size 3, overlap 1 -> stride 2 -> windows [0:3], [2:5].
        chunks = chunk([sentences_page(5)])
        assert len(chunks) == 2
        assert chunks[0].sentences == [
            "S1 is a sentence.",
            "S2 is a sentence.",
            "S3 is a sentence.",
        ]
        assert chunks[1].sentences == [
            "S3 is a sentence.",
            "S4 is a sentence.",
            "S5 is a sentence.",
        ]

    def test_overlap_shares_sentences_between_neighbours(self):
        chunks = chunk([sentences_page(5)], size=3, overlap=1)
        assert chunks[0].sentences[-1] == chunks[1].sentences[0]

    def test_zero_overlap_partitions_without_repeats(self):
        chunks = chunk([sentences_page(6)], size=3, overlap=0)
        assert len(chunks) == 2
        seen = [s for c in chunks for s in c.sentences]
        assert len(seen) == len(set(seen)) == 6

    def test_larger_overlap_produces_more_chunks(self):
        few = chunk([sentences_page(9)], size=3, overlap=1)
        many = chunk([sentences_page(9)], size=3, overlap=2)
        assert len(many) > len(few)

    def test_page_shorter_than_window_yields_one_chunk(self):
        chunks = chunk([sentences_page(2)], size=3, overlap=1)
        assert len(chunks) == 1
        assert len(chunks[0].sentences) == 2

    def test_single_sentence_page(self):
        chunks = chunk([sentences_page(1)])
        assert len(chunks) == 1
        assert chunks[0].text == "S1 is a sentence."

    def test_no_duplicate_tail_window(self):
        # 4 sentences, size 3, stride 2 -> [0:3] then [2:4]; no third window
        # that would merely repeat the tail.
        chunks = chunk([sentences_page(4)], size=3, overlap=1)
        assert len(chunks) == 2
        assert chunks[-1].sentences == ["S3 is a sentence.", "S4 is a sentence."]

    def test_size_one_no_overlap(self):
        chunks = chunk([sentences_page(3)], size=1, overlap=0)
        assert len(chunks) == 3
        assert all(len(c.sentences) == 1 for c in chunks)

    def test_text_is_the_joined_sentences(self):
        chunks = chunk([sentences_page(3)])
        assert chunks[0].text == " ".join(chunks[0].sentences)

    def test_empty_pages_produce_no_chunks(self):
        assert chunk(make_pages(("",), ("   ",))) == []

    def test_empty_page_list(self):
        assert chunk([]) == []


class TestChunkValidation:
    @pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-1, 0)])
    def test_rejects_bad_size(self, size, overlap):
        with pytest.raises(ValueError, match="size must be at least 1"):
            chunk([sentences_page(3)], size=size, overlap=overlap)

    def test_rejects_negative_overlap(self):
        with pytest.raises(ValueError, match="overlap must not be negative"):
            chunk([sentences_page(3)], size=3, overlap=-1)

    @pytest.mark.parametrize("overlap", [3, 4])
    def test_rejects_overlap_at_or_above_size(self, overlap):
        # An overlap equal to size would make the stride zero and loop forever.
        with pytest.raises(ValueError, match="must be smaller than size"):
            chunk([sentences_page(3)], size=3, overlap=overlap)


class TestChunkMetadata:
    def test_ids_are_sequential_from_zero(self):
        chunks = chunk([sentences_page(9)])
        assert [c.chunk_id for c in chunks] == list(range(len(chunks)))

    def test_ids_continue_across_pages(self):
        pages = make_pages(("A one. A two. A three.",), ("B one. B two. B three.",))
        chunks = chunk(pages, size=3, overlap=0)
        assert [c.chunk_id for c in chunks] == [0, 1]

    def test_source_id_is_propagated(self):
        chunks = chunk([sentences_page(3)], source_id="abc123")
        assert all(c.source_id == "abc123" for c in chunks)

    def test_page_metadata_is_propagated(self):
        pages = make_pages(
            ("A one. A two. A three.", 47, "Chapter 1"),
            ("B one. B two. B three.", 48, "Chapter 1"),
        )
        chunks = chunk(pages, size=3, overlap=0)
        assert [(c.pdf_page, c.printed_page, c.chapter) for c in chunks] == [
            (1, 47, "Chapter 1"),
            (2, 48, "Chapter 1"),
        ]

    def test_missing_page_metadata_stays_none(self):
        chunks = chunk(make_pages(("One. Two. Three.",)))
        assert chunks[0].printed_page is None
        assert chunks[0].chapter is None

    def test_chunks_never_span_a_page_break(self):
        # Every chunk must cite exactly one page, so no window may mix pages.
        pages = make_pages(("A one. A two.", 1), ("B one. B two.", 2))
        chunks = chunk(pages, size=3, overlap=1)
        for c in chunks:
            assert all(s.startswith("A") for s in c.sentences) or all(
                s.startswith("B") for s in c.sentences
            )


class TestChunkRoundTrip:
    def test_to_dict_and_back(self):
        original = chunk([sentences_page(3)], source_id="deadbeef")[0]
        assert Chunk.from_dict(original.to_dict()) == original

    def test_to_dict_is_json_serialisable(self):
        import json

        data = chunk([sentences_page(3)], source_id="deadbeef")[0].to_dict()
        assert json.loads(json.dumps(data))["source_id"] == "deadbeef"


class TestAlternatingRunningHeads:
    """Books alternate heads between verso and recto, halving each one's count."""

    def test_detects_both_sides_of_an_alternating_head(self):
        pages = make_pages(
            *[
                (("c h a p t e r 1" if n % 2 == 0 else "i n t r o d u c t i o n") + f"\nBody {n}.",)
                for n in range(12)
            ]
        )
        heads = _running_heads(pages)
        assert "c h a p t e r 1" in heads
        assert "i n t r o d u c t i o n" in heads

    def test_alternating_heads_are_stripped_from_chunks(self):
        pages = make_pages(
            *[
                (
                    ("c h a p t e r 1" if n % 2 == 0 else "i n t r o d u c t i o n")
                    + f"\n{n}\nThe workers rose before dawn number {n}.",
                )
                for n in range(12)
            ]
        )
        for c in chunk(pages):
            assert "c h a p t e r" not in c.text
            assert "i n t r o d u c t i o n" not in c.text
