"""Tests for page extraction, printed-folio detection, and scan detection."""

from __future__ import annotations

import pymupdf
import pytest

from dsearch.extract import (
    NoTextLayerError,
    Page,
    _parse_printed_page,
    _printed_page_for,
    extract,
    normalize_ligatures,
)

# A page of prose long enough to clear the text-layer threshold, so these
# fixtures exercise folio and chapter logic rather than scan detection.
BODY = (
    "The workers moved down the rows before dawn, backs bent to the plants, "
    "filling box after box in the gray light of the field."
)


class TestParsePrintedPage:
    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ("47", 47),
            ("  47  ", 47),
            ("- 47 -", 47),
            ("[47]", 47),
            ("1", 1),
            ("9999", 9999),
        ],
    )
    def test_accepts_bare_folios(self, line, expected):
        assert _parse_printed_page(line) == expected

    @pytest.mark.parametrize(
        "line",
        [
            "",
            "   ",
            "Chapter 3",
            "47 migrant workers crossed",
            "page 47",
            "10000",  # Too large to be a folio.
            "0",  # Pages are 1-based.
            "3.14",
            "1987-1990",
        ],
    )
    def test_rejects_non_folios(self, line):
        assert _parse_printed_page(line) is None

    def test_roman_numerals_are_discarded(self):
        # Front-matter folios are real, but mixing them with arabic numbers
        # would make "p. 12" ambiguous, so they resolve to None.
        assert _parse_printed_page("xiv") is None
        assert _parse_printed_page("ii") is None


class TestPrintedPageFor:
    def test_reads_folio_from_last_line(self):
        assert _printed_page_for("Body text here.\nMore body.\n47") == 47

    def test_reads_folio_from_first_line(self):
        assert _printed_page_for("47\nBody text here.\nMore body.") == 47

    def test_footer_wins_over_header(self):
        # A running head can carry a number too; the footer is the likelier folio.
        assert _printed_page_for("12\nBody text.\n47") == 47

    def test_ignores_folio_in_the_middle_of_a_page(self):
        assert _printed_page_for("Body text.\n47\nMore body.") is None

    def test_returns_none_for_blank_page(self):
        assert _printed_page_for("   \n\n  ") is None


class TestExtract:
    def test_returns_one_page_per_pdf_page(self, pdf_factory):
        path = pdf_factory(pages=[f"{BODY} Page {n} of the manuscript." for n in (1, 2, 3)])
        pages = extract(path)
        assert len(pages) == 3
        assert [p.pdf_page for p in pages] == [1, 2, 3]
        assert isinstance(pages[0], Page)

    def test_extracts_text(self, pdf_factory):
        path = pdf_factory(pages=[f"Migrant workers pick strawberries. {BODY}"])
        assert "Migrant workers pick strawberries." in extract(path)[0].text

    def test_reads_printed_pages_offset_from_pdf_pages(self, pdf_factory):
        # Two pages of front matter, then the book's own numbering starts at 1.
        path = pdf_factory(
            pages=[
                f"Front matter. {BODY}",
                f"Front matter. {BODY}",
                f"Body. {BODY}",
                f"Body. {BODY}",
            ],
            folios=[None, None, "1", "2"],
        )
        pages = extract(path)
        assert [p.printed_page for p in pages] == [None, None, 1, 2]
        assert pages[2].pdf_page == 3 and pages[2].printed_page == 1

    def test_missing_folio_is_none(self, pdf_factory):
        path = pdf_factory(pages=[f"No number on this page. {BODY}"])
        assert extract(path)[0].printed_page is None

    def test_chapter_from_pdf_outline(self, pdf_factory):
        path = pdf_factory(
            pages=[
                f"Intro body. {BODY}",
                f"Chapter one body. {BODY}",
                f"More of one. {BODY}",
                f"Chapter two body. {BODY}",
            ],
            toc=[(1, "Introduction", 1), (1, "Chapter 1: The Body", 2), (1, "Chapter 2", 4)],
        )
        chapters = [p.chapter for p in extract(path)]
        assert chapters == [
            "Introduction",
            "Chapter 1: The Body",
            "Chapter 1: The Body",
            "Chapter 2",
        ]

    def test_chapter_from_font_size_when_no_outline(self, pdf_factory):
        path = pdf_factory(
            pages=[
                f"Body text on the opening page. {BODY}",
                f"Continued body text. {BODY}",
                f"New section body. {BODY}",
            ],
            headings={1: "Introduction", 3: "Symbolic Violence"},
        )
        chapters = [p.chapter for p in extract(path)]
        assert chapters == ["Introduction", "Introduction", "Symbolic Violence"]

    def test_chapter_is_none_when_undetectable(self, pdf_factory):
        path = pdf_factory(pages=[f"Uniform body text. {BODY}", f"More uniform body text. {BODY}"])
        assert [p.chapter for p in extract(path)] == [None, None]

    def test_outline_wins_over_font_heuristic(self, pdf_factory):
        path = pdf_factory(
            pages=[f"Body. {BODY}", f"Body. {BODY}"],
            headings={1: "A Large Decorative Line"},
            toc=[(1, "Real Chapter Title", 1)],
        )
        assert extract(path)[0].chapter == "Real Chapter Title"

    def test_scanned_pdf_raises(self, tmp_path):
        # An image-only PDF: pages exist, but there is no text layer to search.
        doc = pymupdf.open()
        for _ in range(5):
            doc.new_page()
        path = tmp_path / "scan.pdf"
        doc.save(path)
        doc.close()
        with pytest.raises(NoTextLayerError, match="looks like a scan"):
            extract(path)

    def test_mostly_scanned_pdf_raises(self, pdf_factory):
        # One good page out of ten is still a scan, not a searchable book.
        path = pdf_factory(pages=[f"Real text on this page only. {BODY}"] + [""] * 9)
        with pytest.raises(NoTextLayerError):
            extract(path)

    def test_partially_scanned_pdf_is_accepted(self, pdf_factory):
        # Plates and blank versos are normal; they must not fail the whole book.
        path = pdf_factory(pages=[f"Real text here. {BODY}"] * 8 + ["", ""])
        assert len(extract(path)) == 10

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract(tmp_path / "nope.pdf")


class TestHeaderFooterBand:
    """Folios that sit beside a running head, as in the real sample chapters."""

    def test_finds_folio_beside_a_running_head(self):
        page = "introduction\n7\n" + "\n".join(["Body line."] * 10)
        assert _printed_page_for(page) == 7

    def test_finds_folio_above_a_running_foot(self):
        page = "\n".join(["Body line."] * 10) + "\n7\nintroduction"
        assert _printed_page_for(page) == 7

    def test_short_page_does_not_scan_a_band(self):
        # With only three lines there is no header/footer band, so a bare
        # integer in the middle stays body text.
        assert _printed_page_for("Body text.\n47\nMore body.") is None


class TestLigatures:
    """Ligature glyphs must become letters, or quoted evidence is corrupted."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("ﬁve", "five"),
            ("inﬂuence", "influence"),
            ("oﬀer", "offer"),
            ("oﬃce", "office"),
            ("baﬄe", "baffle"),
        ],
    )
    def test_expands_the_unicode_ligature_block(self, raw, expected):
        assert normalize_ligatures(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # The glyph is extracted as its own run, leaving a spurious space.
            ("through  ve army checkpoints", "through five army checkpoints"),
            ("of ce", "office"),
            ("in uence", "influence"),
            ("ri es over their shoulders", "rifles over their shoulders"),
            ("I will pay the  ne", "I will pay the fine"),
        ],
    )
    def test_expands_the_private_use_ligatures_and_their_spurious_space(self, raw, expected):
        assert normalize_ligatures(raw) == expected

    def test_keeps_a_space_that_precedes_a_non_letter(self):
        # Only a space that would otherwise split a word is removed.
        assert normalize_ligatures("the moti . Next") == "the motifi . Next"

    def test_leaves_ordinary_text_untouched(self):
        text = "The workers rose before dawn and walked to the field."
        assert normalize_ligatures(text) == text

    def test_leaves_an_unmapped_private_use_character_visible(self):
        # Guessing at an unknown glyph would silently invent text.
        assert "" in normalize_ligatures("unknown  glyph")

    def test_extract_normalises_every_page(self, pdf_factory, monkeypatch):
        # A ligature glyph cannot be round-tripped through a base-14 PDF font,
        # so this asserts the wiring rather than the glyph: every page's text
        # must pass through normalisation on its way out of extract().
        seen: list[str] = []

        def spy(text: str) -> str:
            seen.append(text)
            return text.replace("CANARY", "normalised")

        monkeypatch.setattr("dsearch.extract.normalize_ligatures", spy)
        path = pdf_factory(pages=[f"CANARY {BODY}", f"CANARY {BODY}"])
        pages = extract(path)
        assert len(seen) >= 2
        assert all("normalised" in page.text for page in pages)
