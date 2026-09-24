"""MLA citations built from stored source metadata and a retrieved result.

The quoted sentence is copied verbatim out of the PDF. Nothing here rewrites,
summarises, or completes it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # Annotations only: `index` imports this module at add time.
    from dsearch.chunk import Chunk
    from dsearch.index import SourceMeta
    from dsearch.search import Result

# MLA abbreviates a list of three or more authors to the first plus "et al."
MAX_NAMED_AUTHORS = 2

# Filename separators that stand in for spaces, and the letter/digit seam in
# names such as "Chapter5" or "Holmes2013".
_FILENAME_SEPARATOR_RE = re.compile(r"[_\-]+")
_LETTER_DIGIT_SEAM_RE = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")


def title_from_filename(filename: str) -> str:
    """A readable title from a filename.

    "fresh_fruit-broken_bodies.pdf" -> "Fresh Fruit Broken Bodies", and
    "Chapter5.pdf" -> "Chapter 5". Only the first letter of each word is
    raised, so "don't" does not become "Don'T" and an acronym such as "FFBB"
    keeps its capitals.
    """
    stem = Path(filename).stem
    words = _LETTER_DIGIT_SEAM_RE.sub(" ", _FILENAME_SEPARATOR_RE.sub(" ", stem))
    return " ".join(word[:1].upper() + word[1:] for word in words.split())


def resolve_author(explicit: str | None, pdf_author: str | None) -> str:
    """The author to cite: the `--author` flag, else the PDF's metadata, else nothing.

    There is no filename fallback for an author — a filename is a title at
    best, and a citation with an invented author is worse than one with none.
    """
    return (explicit or "").strip() or (pdf_author or "").strip()


def resolve_title(explicit: str | None, pdf_title: str | None, filename: str) -> str:
    """The title to cite: the `--title` flag, else the PDF's metadata, else the filename.

    The chapter title is never a candidate: it names a part of the work, and
    MLA cites the work.
    """
    return (explicit or "").strip() or (pdf_title or "").strip() or title_from_filename(filename)


# Straight quotes in, typographic quotes out — this text is pasted into essays.
OPEN_QUOTE = "“"
CLOSE_QUOTE = "”"

# MLA 9 treats the two kinds of terminal punctuation differently. A final
# period is dropped from the quotation and reappears after the parenthetical.
# A question mark or exclamation point is part of the quoted words, so it stays
# inside the quotation marks and a period is added after the parenthetical.
_DROPPED_TERMINAL = "."
_KEPT_TERMINALS = "?!"


def format_author(raw: str) -> str:
    """Render an author in MLA inverted order: "Seth Holmes" -> "Holmes, Seth".

    A name already inverted (containing a comma) is left alone. Two authors are
    joined with "and" and only the first is inverted; three or more collapse to
    "et al.", as MLA 9 requires.
    """
    name = (raw or "").strip()
    if not name:
        return ""
    if "," in name:
        return name.rstrip(".")

    authors = [part.strip() for part in name.split(" and ") if part.strip()]
    if len(authors) > MAX_NAMED_AUTHORS:
        return f"{_invert(authors[0])}, et al"
    if len(authors) == MAX_NAMED_AUTHORS:
        return f"{_invert(authors[0])}, and {authors[1]}"
    return _invert(authors[0])


def _invert(name: str) -> str:
    """Put the last word of a name first: "Seth Holmes" -> "Holmes, Seth"."""
    parts = name.split()
    if len(parts) < 2:
        return name
    return f"{parts[-1]}, {' '.join(parts[:-1])}"


def page_reference(chunk: Chunk) -> str:
    """The page element of a citation.

    Prefers the folio printed on the page, which is what a reader checking the
    citation will look for. Falls back to the PDF's own page number, labelled so
    nobody mistakes it for a printed page.
    """
    if chunk.printed_page is not None:
        return f"p. {chunk.printed_page}"
    return f"PDF p. {chunk.pdf_page}"


def mla(result: Result, meta: SourceMeta | None = None) -> str:
    """One MLA citation line for a retrieved passage.

    Example: `Holmes, Seth. *Fresh Fruit, Broken Bodies*. p. 47.`

    Title is italicised with markdown asterisks, which both `rich` and Streamlit
    render. Elements that are unknown are dropped rather than faked.
    """
    source = meta or result.meta
    parts = []

    author = format_author(source.author)
    if author:
        parts.append(f"{author}.")

    title = (source.title or "").strip()
    if title:
        parts.append(f"*{title}*.")

    parts.append(f"{page_reference(result.chunk)}.")
    return " ".join(parts)


def quotation(result: Result) -> str:
    """The best-matching sentence, in typographic quotes, verbatim."""
    sentence = result.best_sentence_text.strip()
    return f"{OPEN_QUOTE}{sentence}{CLOSE_QUOTE}"


def parenthetical(result: Result, meta: SourceMeta | None = None) -> str:
    """An MLA in-text citation: `(Holmes 47)`.

    This is what actually goes in the body of an essay, beside the quotation.
    """
    source = meta or result.meta
    surname = format_author(source.author).split(",")[0].strip()
    page = result.chunk.printed_page
    inner = " ".join(part for part in (surname, str(page) if page is not None else "") if part)
    return f"({inner})" if inner else ""


def quoted_line(result: Result, meta: SourceMeta | None = None) -> str:
    """The best sentence in quotation marks, followed by its in-text citation.

    A final period moves outside the closing quotation mark, after the
    parenthetical, as MLA 9 requires; a question mark or exclamation point
    stays inside and a period is added after the parenthetical.
    """
    source = meta or result.meta
    sentence = result.best_sentence_text.strip()
    inline = parenthetical(result, source)

    if not inline:
        return quotation(result)
    if sentence.endswith(_DROPPED_TERMINAL):
        return f"{OPEN_QUOTE}{sentence[:-1]}{CLOSE_QUOTE} {inline}."
    # Covers a kept "?"/"!" and a sentence with no terminal punctuation at all
    # (a fragment split by a page break), which read the same way.
    return f"{OPEN_QUOTE}{sentence}{CLOSE_QUOTE} {inline}."


def citation_block(result: Result, meta: SourceMeta | None = None) -> str:
    """The citation and the quotation, ready to paste into an essay.

    Line one is the full reference (`Author. *Title*. p. N.`); line two is the
    best-matching sentence in quotation marks.
    """
    source = meta or result.meta
    return f"{mla(result, source)}\n{quoted_line(result, source)}"
