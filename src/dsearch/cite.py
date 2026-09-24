"""MLA citations built from stored source metadata and a retrieved result.

The quoted sentence is copied verbatim out of the PDF. Nothing here rewrites,
summarises, or completes it.
"""

from __future__ import annotations

from dsearch.chunk import Chunk
from dsearch.index import SourceMeta
from dsearch.search import Result

# MLA abbreviates a list of three or more authors to the first plus "et al."
MAX_NAMED_AUTHORS = 2

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


def citation_block(result: Result, meta: SourceMeta | None = None) -> str:
    """The quotation and its citation, ready to paste into an essay.

    The quotation keeps its own terminal punctuation inside the quotation marks;
    the in-text citation follows, then the full reference on the next line.
    """
    source = meta or result.meta
    sentence = result.best_sentence_text.strip()
    inline = parenthetical(result, source)

    if not inline:
        quoted = quotation(result)
    elif sentence.endswith(_DROPPED_TERMINAL):
        quoted = f"{OPEN_QUOTE}{sentence[:-1]}{CLOSE_QUOTE} {inline}."
    else:
        # Covers a kept "?"/"!" and a sentence with no terminal punctuation at
        # all (a fragment split by a page break), which read the same way.
        quoted = f"{OPEN_QUOTE}{sentence}{CLOSE_QUOTE} {inline}."

    return f"{quoted}\n{mla(result, source)}"
