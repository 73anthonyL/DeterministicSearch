"""Sentence splitting and overlapping-window chunking.

Pure functions: given pages, produce the `Chunk` records that everything
downstream — embeddings, BM25, citations, and the planned key-term and
embedding-map features — reads from.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field

from dsearch.extract import Page

# Window defaults. Three sentences is long enough to carry an argument and
# short enough that the best-matching sentence still dominates the embedding.
DEFAULT_SIZE = 3
DEFAULT_OVERLAP = 1

# A running head repeats on at least this share of a source's pages. Books
# alternate heads between verso and recto (the chapter number on one side, the
# chapter title on the other), so each one appears on only about half the pages
# -- measured at 48% in the sample chapters. The threshold sits below that, and
# still far above the ~5% noise floor of lines that merely recur.
RUNNING_HEAD_RATIO = 0.3

# Running heads are short; a repeated line longer than this is body text
# (a refrain, a table row) and is kept.
MAX_RUNNING_HEAD_CHARS = 60

# A source must have at least this many pages before repeated-line detection is
# trustworthy. Below it, coincidence is as likely as a real running head.
MIN_PAGES_FOR_RUNNING_HEAD = 4

# Words that end in a period without ending a sentence.
_ABBREVIATIONS = frozenset(
    [
        # Titles and names
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "st",
        "jr",
        "sr",
        # Scholarly apparatus
        "vs",
        "etc",
        "cf",
        "al",
        "fig",
        "figs",
        "no",
        "nos",
        "pp",
        "ch",
        "chs",
        "ed",
        "eds",
        "vol",
        "vols",
        "trans",
        "rev",
        "ca",
        "circa",
        "e.g",
        "i.e",
        # Organisations
        "inc",
        "ltd",
        "co",
        "corp",
        "dept",
        "univ",
        "approx",
        # Months
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sept",
        "sep",
        "oct",
        "nov",
        "dec",
    ]
)

# A sentence ends at .!? (plus any closing quote or bracket) when the next
# non-space character opens a new sentence.
_BOUNDARY_RE = re.compile(r"""([.!?]+)(["'”’)\]]*)(\s+)(?=["'“‘(\[]*[A-Z0-9])""")

# The token immediately before a candidate boundary.
_LAST_WORD_RE = re.compile(r"([A-Za-z][A-Za-z.]*)$")


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage: a window of consecutive sentences from one page.

    `chunk_id` is the 0-based position of this chunk within its source, and is
    also its row index in `vectors_<tier>.npy` — that alignment is what lets the
    planned embedding map join vectors back to chapters without extra state.
    A chunk is identified globally by the pair (`source_id`, `chunk_id`).

    Chunks never span a page break, so `printed_page` cites the passage exactly.
    """

    chunk_id: int
    source_id: str
    text: str
    sentences: list[str] = field(default_factory=list)
    pdf_page: int = 0
    printed_page: int | None = None
    chapter: str | None = None

    def to_dict(self) -> dict:
        """Serialise for `chunks.jsonl`."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        """Rebuild from a `chunks.jsonl` row."""
        return cls(**data)


def _dehyphenate(text: str) -> str:
    """Rejoin words broken across a line break ("compan-\\nions" -> "companions").

    Typesetters hyphenate at the right margin, and pymupdf preserves those
    breaks. Left alone they split words in the middle of quoted evidence. A
    genuine compound that happens to break at its hyphen loses the hyphen; that
    is rarer than mid-word breaks and costs only a character.
    """
    return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)


def _running_heads(pages: list[Page]) -> set[str]:
    """Find short lines repeated across most pages: running heads and feet.

    These are navigation furniture, not prose. Left in, they are spliced into
    the first sentence of a page and end up inside a quotation.
    """
    if len(pages) < MIN_PAGES_FOR_RUNNING_HEAD:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        lines = {
            line.strip()
            for line in page.text.splitlines()
            if line.strip() and len(line.strip()) <= MAX_RUNNING_HEAD_CHARS
        }
        counts.update(lines)
    threshold = len(pages) * RUNNING_HEAD_RATIO
    return {line for line, count in counts.items() if count >= threshold}


def _clean_page_text(text: str, running_heads: set[str]) -> str:
    """Strip folios and running heads, rejoin hyphenated words, reflow lines."""
    kept = []
    for line in _dehyphenate(text).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in running_heads:
            continue
        if stripped.isdigit():  # A bare folio.
            continue
        kept.append(stripped)
    return re.sub(r"\s+", " ", " ".join(kept)).strip()


def _is_abbreviation(text_before: str) -> bool:
    """True if the word ending at a candidate boundary is a known abbreviation."""
    match = _LAST_WORD_RE.search(text_before)
    if not match:
        return False
    word = match.group(1).rstrip(".").lower()
    if not word:
        return False
    if word in _ABBREVIATIONS:
        return True
    # A single letter is an initial ("J. Smith"); "U.S." collapses to "us".
    return len(word) == 1


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences with a regex — no NLTK download required.

    Handles the abbreviations and initials that a naive split on "." would
    break. Ellipses and multi-character terminators ("?!") stay with their
    sentence.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY_RE.finditer(text):
        end = match.end(2)  # Keep terminal punctuation and closing quotes.
        if _is_abbreviation(text[start : match.start(1)]):
            continue
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = match.end(3)

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def chunk(
    pages: list[Page],
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    *,
    source_id: str = "",
) -> list[Chunk]:
    """Window each page's sentences into overlapping chunks.

    Windows never cross a page break so that every result can cite one exact
    page. The cost is a short chunk at the foot of each page, and a sentence
    split by a page break being chunked as two fragments.

    Raises ValueError if `size` is below 1 or `overlap` is not a valid stride.
    """
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}")
    if overlap < 0:
        raise ValueError(f"overlap must not be negative, got {overlap}")
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")

    heads = _running_heads(pages)
    step = size - overlap
    chunks: list[Chunk] = []
    next_id = 0

    for page in pages:
        sentences = split_sentences(_clean_page_text(page.text, heads))
        if not sentences:
            continue
        for start in range(0, len(sentences), step):
            window = sentences[start : start + size]
            if not window:
                break
            chunks.append(
                Chunk(
                    chunk_id=next_id,
                    source_id=source_id,
                    text=" ".join(window),
                    sentences=window,
                    pdf_page=page.pdf_page,
                    printed_page=page.printed_page,
                    chapter=page.chapter,
                )
            )
            next_id += 1
            if start + size >= len(sentences):
                break  # This window reached the end; another would repeat its tail.

    return chunks
