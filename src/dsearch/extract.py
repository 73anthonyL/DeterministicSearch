"""Per-page text extraction from PDFs, with best-effort page numbers and chapter titles."""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)

# Bit set in a span's `flags` when MuPDF judged it a superscript — an endnote
# marker, in a book. Dropping the span keeps "competency.35" from reaching the
# quoted evidence as if it were part of the sentence.
SUPERSCRIPT_FLAG = 1

# Fewer letter-spaced tokens than this could be a real word ("A I" in a title);
# "F I V E" and longer are typographic spacing and are collapsed.
MIN_LETTER_SPACED_RUN = 3

# A page counts as "empty" below this many characters of extracted text. Running
# heads and stray artifacts mean a scanned page is rarely exactly zero-length.
MIN_CHARS_FOR_TEXT_LAYER = 20

# If more than this share of pages have no usable text layer, the PDF is a scan.
SCANNED_PAGE_RATIO = 0.8

# A printed folio is a short bare integer. Real page numbers stay under this.
MAX_PRINTED_PAGE = 9999

# How many lines at the top and bottom of a page count as the header/footer
# band. Books commonly set a running head and the folio on the same band, so
# the number lands on the second extracted line rather than the first.
HEADER_FOOTER_BAND = 3

# A heading must be this much larger than body text to count as a chapter title.
HEADING_SIZE_RATIO = 1.25

# Headings are short. Longer lines are body text that merely happens to be large.
MAX_HEADING_CHARS = 90

# Ligatures that arrive as a single glyph and must be expanded back into
# letters, or quoted evidence reads "the  ne" instead of "the fine".
#
# The U+FBxx entries are the real Unicode ligature block and are always correct.
# The U+F0DE/U+F0DF entries are private-use codepoints: they carry no Unicode
# meaning, and their values here were read off the sample typesetting, where
# replacing them yields "five", "office", "influence", "rifles" and so on across
# all 400+ occurrences. Another publisher's font may use these slots
# differently, so only these two observed slots are mapped; any other
# private-use character is left visible rather than silently guessed at.
LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "ft",
    "\ufb06": "st",
    "\uf0de": "fi",
    "\uf0df": "fl",
}

# A ligature glyph is extracted as its own run, so a spurious space follows it
# ("of" + fi + " ce"). The space is dropped only when a letter follows, which is
# the case that would otherwise split a word in two.
_LIGATURE_RE = re.compile(
    "(" + "|".join(re.escape(glyph) for glyph in LIGATURES) + r")(\s(?=[A-Za-z]))?"
)

# Roman numerals used for front matter folios (i, ii, ... xlviii).
_ROMAN_RE = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)
_BARE_INT_RE = re.compile(r"^\d{1,4}$")

# An endnote marker that survived the superscript check: one to three digits
# glued to the end of a word or its closing punctuation, then whitespace or the
# end of the line. The letter before it must be lowercase so that codes such as
# "B12", "F16" and "COVID19" are left alone; a real number in prose ("in 1984
# the") is always preceded by a space and so never matches.
_ENDNOTE_MARKER_RE = re.compile(r"([a-z][.,;:!?)\]\"'”’]{0,2})(\d{1,3})(?=\s|$)")

# Three or more single characters separated by single spaces: a letter-spaced
# chapter number such as "F I V E".
_LETTER_SPACED_RE = re.compile(
    r"(?<!\S)(\w(?: \w){" + str(MIN_LETTER_SPACED_RUN - 1) + r",})(?!\S)"
)


def normalize_ligatures(text: str) -> str:
    """Expand ligature glyphs back into the letters they stand for.

    Without this, a passage the PDF renders as "five army checkpoints" is
    retrieved and quoted as "  ve army checkpoints" — the tool would be showing
    the reader something the source does not say.
    """
    return _LIGATURE_RE.sub(lambda m: LIGATURES[m.group(1)], text)


def strip_endnote_markers(text: str) -> tuple[str, int]:
    """Remove endnote digits attached to a word ("competency.35 In" -> "competency. In").

    This is the fallback for markers the PDF did not flag as superscript. It is
    deliberately narrow (see `_ENDNOTE_MARKER_RE`) and the count is returned so
    the caller can log it — a nonzero count on a page with no notes is the
    signal that a real number was eaten.
    """
    return _ENDNOTE_MARKER_RE.subn(r"\1", text)


def normalize_title(title: str) -> str:
    """Tidy a chapter title for storage: "F I V E  “Doctors”" -> "FIVE “Doctors”".

    Publishers letter-space chapter numbers for effect; the spaces are
    typography, not text, and would otherwise show in every result header.
    """
    collapsed = _LETTER_SPACED_RE.sub(lambda m: m.group(1).replace(" ", ""), title)
    return re.sub(r"\s+", " ", collapsed).strip()


def _page_text(page: pymupdf.Page) -> str:
    """Page text rebuilt from spans, with superscript spans dropped.

    Block and line boundaries are preserved exactly as `get_text("text")` lays
    them out, so folio and running-head detection downstream see the same
    lines. A page with no span data at all (some generators emit only raw text)
    falls back to the plain text layer.
    """
    blocks: list[str] = []
    for block in page.get_text("dict").get("blocks", []):
        lines = []
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            kept = [s.get("text", "") for s in spans if not s.get("flags", 0) & SUPERSCRIPT_FLAG]
            if spans:
                lines.append("".join(kept))
        if lines:
            blocks.append("\n".join(lines))
    if not blocks:
        return page.get_text("text")
    return "\n".join(blocks) + "\n"


def _clean_page_text(page: pymupdf.Page) -> str:
    """Extract one page's text with ligatures expanded and endnote markers removed."""
    text, stripped = strip_endnote_markers(normalize_ligatures(_page_text(page)))
    if stripped:
        logger.debug("page %d: stripped %d endnote marker(s)", page.number + 1, stripped)
    return text


class NoTextLayerError(Exception):
    """Raised when a PDF is a scan: it has images but no extractable text."""


@dataclass(frozen=True)
class Page:
    """One page of a source PDF.

    `pdf_page` is the 1-based physical page index and is always known.
    `printed_page` is the folio printed on the paper, which is what a reader
    cites; it is often offset from `pdf_page` by the front matter, and is None
    when no number could be found.
    """

    pdf_page: int
    text: str
    printed_page: int | None = None
    chapter: str | None = None


def _parse_printed_page(line: str) -> int | None:
    """Return the page number if `line` is a bare folio, else None.

    Accepts a plain integer, or an integer flanked by decoration a designer
    might add ("- 47 -", "[47]"). Roman numerals are recognised but discarded:
    they are real folios, yet mixing them with arabic numbers would make
    citations ambiguous.
    """
    stripped = line.strip().strip("[]()-–—•. \t")
    if not stripped:
        return None
    if _BARE_INT_RE.match(stripped):
        value = int(stripped)
        return value if 0 < value <= MAX_PRINTED_PAGE else None
    if _ROMAN_RE.match(stripped):
        return None
    return None


def _printed_page_for(text: str) -> int | None:
    """Find a printed folio in the header or footer band of a page.

    On a page long enough to have a real header and footer, the folio may sit
    beside a running head and so land a line or two in. On a very short page
    there is no meaningful band, so only the outermost lines are trusted —
    otherwise a stray integer in the body would be read as a page number.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None

    if len(lines) > 2 * HEADER_FOOTER_BAND:
        footer = list(reversed(lines[-HEADER_FOOTER_BAND:]))
        header = lines[:HEADER_FOOTER_BAND]
    else:
        footer, header = [lines[-1]], [lines[0]]

    # Footer first: the bottom of the page is the more common home for a folio.
    for line in (*footer, *header):
        found = _parse_printed_page(line)
        if found is not None:
            return found
    return None


def _body_font_size(doc: pymupdf.Document) -> float:
    """Most common span font size in the document, rounded to the nearest point.

    Body text dominates any book by character count, so the modal size is a
    reliable baseline to measure headings against.
    """
    sizes: Counter[float] = Counter()
    for page in doc:
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    chars = len(span.get("text", "").strip())
                    if chars:
                        sizes[round(span["size"], 1)] += chars
    if not sizes:
        return 0.0
    return sizes.most_common(1)[0][0]


def _heading_on_page(page: pymupdf.Page, body_size: float) -> str | None:
    """Return the first line on `page` that looks like a chapter heading."""
    if body_size <= 0:
        return None
    threshold = body_size * HEADING_SIZE_RATIO
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = normalize_ligatures("".join(span.get("text", "") for span in spans)).strip()
            if not text or len(text) > MAX_HEADING_CHARS:
                continue
            if _parse_printed_page(text) is not None:
                continue  # A large page number is not a heading.
            if max((span["size"] for span in spans), default=0.0) >= threshold:
                return normalize_title(text)
    return None


def _chapters_from_toc(doc: pymupdf.Document) -> dict[int, str]:
    """Map 1-based pdf page -> chapter title using the PDF's own outline."""
    try:
        toc = doc.get_toc()
    except Exception:
        return {}
    starts: dict[int, str] = {}
    for entry in toc:
        # get_toc rows are [level, title, page]; deeper levels are sections.
        if len(entry) < 3:
            continue
        level, title, page_no = entry[0], entry[1], entry[2]
        if level != 1 or page_no < 1:
            continue
        title = normalize_title(title)
        if title:
            starts.setdefault(page_no, title)
    return starts


def _carry_forward(starts: dict[int, str], page_count: int) -> list[str | None]:
    """Expand chapter start pages into a per-page chapter title."""
    chapters: list[str | None] = []
    current: str | None = None
    for pdf_page in range(1, page_count + 1):
        if pdf_page in starts:
            current = starts[pdf_page]
        chapters.append(current)
    return chapters


def extract(pdf_path: str | Path) -> list[Page]:
    """Extract every page of `pdf_path` with its page number and chapter title.

    Raises FileNotFoundError if the path is missing, and NoTextLayerError if the
    PDF is a scan — in that case the user needs to OCR it before dsearch can
    retrieve anything from it.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"No such PDF: {path}")

    with pymupdf.open(path) as doc:
        page_count = doc.page_count
        if page_count == 0:
            raise NoTextLayerError(f"{path.name} has no pages.")

        texts = [_clean_page_text(page) for page in doc]
        empty = sum(1 for text in texts if len(text.strip()) < MIN_CHARS_FOR_TEXT_LAYER)
        if empty / page_count > SCANNED_PAGE_RATIO:
            raise NoTextLayerError(
                f"{path.name} has no text layer on {empty} of {page_count} pages. "
                "It looks like a scan — run it through OCR (e.g. ocrmypdf) and add it again."
            )

        starts = _chapters_from_toc(doc)
        if not starts:
            body_size = _body_font_size(doc)
            for index, page in enumerate(doc, start=1):
                heading = _heading_on_page(page, body_size)
                if heading:
                    starts.setdefault(index, heading)
        chapters = _carry_forward(starts, page_count)

    return [
        Page(
            pdf_page=index,
            text=text,
            printed_page=_printed_page_for(text),
            chapter=chapters[index - 1],
        )
        for index, text in enumerate(texts, start=1)
    ]
