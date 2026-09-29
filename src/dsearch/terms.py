"""Per-chapter key terms: the words and phrases that set one chapter apart.

Nothing here generates text. A term is a word or two-word phrase that occurs
in the source, reported with how often it occurs and the page where it is
densest, so every term can be traced back to the book.

Terms are found in two passes. The lexical pass scores each candidate by how
much more often it occurs in the chapter than in the rest of the same book.
The optional re-rank pass embeds the best candidates and orders them by
similarity to the chapter's stored vectors. The two orderings are fused by
reciprocal rank, the same way `search` fuses dense and BM25 results.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np

from dsearch import index
from dsearch.chunk import Chunk
from dsearch.index import DEFAULT_TIER, SourceMeta, SourceNotFoundError
from dsearch.search import rrf

DEFAULT_TOP = 10

# How many lexical candidates per chapter go forward to the re-rank. Wide
# enough that a term the lexical pass ranked 30th can still surface, small
# enough that embedding them takes about a second.
CANDIDATE_POOL = 50

# A term must occur this often in a chapter to be a candidate. Below it, a
# word that happens to appear only in one chapter looks maximally distinctive
# on the strength of a single sentence.
MIN_TERM_COUNT = 3

# Shorter words are almost all function words or stray initials.
MIN_WORD_CHARS = 3

# Added to a term's count outside the chapter, so a term that occurs nowhere
# else has a finite score instead of a division by zero.
REST_SMOOTHING = 1.0

# A single word is folded into a two-word phrase when at least this share of
# its occurrences are inside that phrase: "Miguel" is dropped for "San Miguel",
# but "border" survives alongside "border patrol".
ABSORB_RATIO = 0.6

# One word may appear in at most this many of a chapter's terms. Without a cap
# a chapter's list can read "embodied", "embodied anthropology", "embodied
# experiences": true, but three places spent on one idea.
MAX_TERMS_PER_WORD = 2

# Below this length a centroid difference is numerical noise, not a direction.
MIN_CONTRAST_NORM = 1e-6

# Words too common to characterise anything. Deliberately a short, readable
# list rather than a dependency: it only has to keep function words out of the
# candidates, since the scoring already demotes words common to every chapter.
# Kept as a wrapped string because 170 one-word lines would bury the module.
STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be
    because been before being below between both but by can could did do does
    doing down during each either else even ever few for from further had has
    have having he her here hers him his how however i if in into is it its
    itself just less many may me might more most much must my no nor not now of
    off often on once one only or other our out over own per rather same she
    should since so some such than that the their them then there these they
    this those though through thus to too two under until up upon us very was
    we were what when where which while who whom whose why will with within
    without would yet you your
    said says say like get got well back still new first way make made see use
    used three don't doesn't didn't can't won't isn't aren't wasn't weren't
    i'm i've it's that's
    """.split()  # noqa: SIM905
)

# Punctuation that ends a phrase. A two-word term must not straddle a comma or
# a dash: "workers, farm" is an accident of word order, not a phrase.
_SEGMENT_RE = re.compile(r"[^\w\s'’-]+|\s[-–—]+\s|[–—]")

# A word: letters (any script), with internal apostrophes or hyphens.
_WORD_RE = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*")

# Identifies a page across a book that is split over several sources.
PageKey = tuple[str, int]


@dataclass(frozen=True)
class Term:
    """One key term, with the evidence of where it lives in the source."""

    text: str
    count: int
    score: float
    source_id: str
    pdf_page: int
    printed_page: int | None = None

    @property
    def page_label(self) -> str:
        """The densest page, cited the same way a search result is."""
        if self.printed_page is not None:
            return f"p. {self.printed_page} (PDF {self.pdf_page})"
        return f"PDF p. {self.pdf_page}"


@dataclass
class ChapterTerms:
    """The key terms of one chapter."""

    label: str
    # False when the PDF gave no chapter title and `label` is the filename.
    titled: bool
    source_ids: list[str] = field(default_factory=list)
    terms: list[Term] = field(default_factory=list)


@dataclass
class BookTerms:
    """Key terms for every chapter of one book.

    A book is every source sharing a title and author, because a book is often
    added as one PDF per chapter.
    """

    title: str
    author: str
    chapters: list[ChapterTerms] = field(default_factory=list)
    # The tier used for the re-rank; None when the ranking is lexical only.
    tier: str | None = None
    # False when the book has a single chapter, so terms are ranked by plain
    # frequency: there is nothing to be distinctive from.
    contrasted: bool = True
    note: str | None = None


@dataclass
class _Chapter:
    """Working state for one chapter while terms are counted and scored."""

    label: str
    titled: bool
    source_ids: list[str] = field(default_factory=list)
    # One entry per de-duplicated sentence: its page, and its phrase segments.
    sentences: list[tuple[PageKey, list[list[str]]]] = field(default_factory=list)
    # (source_id, chunk_id) of every chunk, to find the chapter's vectors.
    rows: list[tuple[str, int]] = field(default_factory=list)
    counts: Counter[str] = field(default_factory=Counter)
    pages: dict[str, Counter[PageKey]] = field(default_factory=dict)
    # Non-stopword words in the chapter: the denominator of a term's rate.
    size: int = 0


def unique_sentences(chunks: Iterable[Chunk], overlap: int) -> Iterator[tuple[Chunk, str]]:
    """Yield each sentence of a source once, with the chunk it came from.

    Chunks overlap, so counting words from `chunk.text` would count every
    shared sentence twice. Within a page each chunk after the first repeats
    the last `overlap` sentences of the one before it; those are skipped.
    """
    previous: PageKey | None = None
    for item in chunks:
        page = (item.source_id, item.pdf_page)
        fresh = item.sentences[overlap:] if page == previous else item.sentences
        previous = page
        for sentence in fresh:
            yield item, sentence


def segment_words(sentence: str) -> list[list[str]]:
    """Split a sentence into phrase segments of words, original case kept."""
    segments = [_WORD_RE.findall(part) for part in _SEGMENT_RE.split(sentence)]
    return [words for words in segments if words]


def _is_content_word(word: str) -> bool:
    return len(word) >= MIN_WORD_CHARS and word.replace("’", "'") not in STOPWORDS


def _chapter_key(item: Chunk) -> tuple[str, str]:
    """Chunks with no chapter title are grouped by source, never pooled."""
    if item.chapter:
        return ("chapter", item.chapter)
    return ("source", item.source_id)


def _collect(
    chunks_by_source: dict[str, list[Chunk]], metas: dict[str, SourceMeta]
) -> list[_Chapter]:
    """Group a book's de-duplicated sentences into chapters, in reading order."""
    chapters: dict[tuple[str, str], _Chapter] = {}
    for source_id, chunks in chunks_by_source.items():
        meta = metas[source_id]
        for item in chunks:
            chapter = _chapter_for(chapters, item, meta)
            chapter.rows.append((source_id, item.chunk_id))
        for item, sentence in unique_sentences(chunks, meta.overlap):
            chapter = _chapter_for(chapters, item, meta)
            chapter.sentences.append(((source_id, item.pdf_page), segment_words(sentence)))
    return list(chapters.values())


def _chapter_for(
    chapters: dict[tuple[str, str], _Chapter], item: Chunk, meta: SourceMeta
) -> _Chapter:
    key = _chapter_key(item)
    if key not in chapters:
        chapters[key] = _Chapter(label=item.chapter or meta.filename, titled=bool(item.chapter))
    chapter = chapters[key]
    if item.source_id not in chapter.source_ids:
        chapter.source_ids.append(item.source_id)
    return chapter


def _vocabulary(chapters: list[_Chapter]) -> set[str]:
    """Every lowercased word in the book."""
    return {
        word.lower()
        for chapter in chapters
        for _, segments in chapter.sentences
        for words in segments
        for word in words
    }


def _singular(word: str) -> str | None:
    """The singular a regular English plural would have, or None if not plural."""
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return None


def _fold_plurals(vocabulary: set[str]) -> dict[str, str]:
    """Map each word to the form it is counted under.

    A regular plural is counted with its singular when the book uses both, so
    "patient" and "patients" are one term rather than two competing ones. Only
    these two endings are handled: a stemmer would merge more, but would also
    merge words that are not the same ("united" and "unit").
    """
    canonical = {}
    for word in vocabulary:
        singular = _singular(word)
        folds = singular in vocabulary and len(singular or "") >= MIN_WORD_CHARS
        canonical[word] = singular if folds and singular else word
    return canonical


def _count(
    chapter: _Chapter, canonical: dict[str, str], spellings: dict[str, Counter[str]]
) -> None:
    """Count every word and two-word phrase in the chapter, and where it falls.

    `spellings` collects, across the whole book, how each term was actually
    written, so that a name can be shown with its capitals.
    """
    for page, segments in chapter.sentences:
        for words in segments:
            keys = [canonical[word.lower()] for word in words]
            content = [_is_content_word(key) for key in keys]
            for position, key in enumerate(keys):
                if not content[position]:
                    continue
                chapter.size += 1
                _tally(chapter, key, page)
                spellings.setdefault(key, Counter())[words[position]] += 1
                if position + 1 < len(keys) and content[position + 1]:
                    phrase = f"{key} {keys[position + 1]}"
                    _tally(chapter, phrase, page)
                    written = " ".join(words[position : position + 2])
                    spellings.setdefault(phrase, Counter())[written] += 1


def _tally(chapter: _Chapter, term: str, page: PageKey) -> None:
    chapter.counts[term] += 1
    chapter.pages.setdefault(term, Counter())[page] += 1


def _display_forms(spellings: dict[str, Counter[str]]) -> dict[str, str]:
    """The most common spelling of each term; the earliest seen on a tie."""
    return {term: written.most_common(1)[0][0] for term, written in spellings.items()}


def distinctiveness(count: int, size: int, rest_count: int, rest_size: int) -> float:
    """How strongly a term marks a chapter out from the rest of its book.

    The term's rate in the chapter, weighted by the log of how many times
    higher that is than its rate elsewhere (its contribution to the KL
    divergence between the chapter and the rest). A textbook TF-IDF was tried
    first and rejected: with a handful of chapters nearly every word occurs in
    all of them, so IDF is almost constant and frequent words win.

    With no other chapters there is nothing to contrast with, and the score
    falls back to the plain rate.
    """
    if size <= 0 or count <= 0:
        return 0.0
    rate = count / size
    if rest_size <= 0:
        return rate
    rest_rate = (rest_count + REST_SMOOTHING) / rest_size
    if rate <= rest_rate:
        return 0.0
    return rate * math.log(rate / rest_rate)


def _candidates(chapter: _Chapter, book: Counter[str], book_size: int, pool: int) -> list[str]:
    """The chapter's `pool` most distinctive terms, best first."""
    scores = {
        term: distinctiveness(count, chapter.size, book[term] - count, book_size - chapter.size)
        for term, count in chapter.counts.items()
        if count >= MIN_TERM_COUNT
    }
    ranked = sorted((t for t in scores if scores[t] > 0), key=lambda t: (-scores[t], t))
    return ranked[:pool]


def chapter_direction(chapter_vectors: np.ndarray, book_vectors: np.ndarray) -> np.ndarray:
    """The unit vector pointing from the book's average towards this chapter.

    Chapters of one book share most of their meaning, so a candidate's
    similarity to the chapter's own centroid mostly measures how typical it is
    of the *book*. Subtracting the book's centroid leaves what is particular to
    the chapter. When the two coincide (a one-chapter book) the plain centroid
    is used.
    """
    centroid = chapter_vectors.mean(axis=0)
    contrast = centroid - book_vectors.mean(axis=0)
    direction = contrast if np.linalg.norm(contrast) > MIN_CONTRAST_NORM else centroid
    return direction / max(float(np.linalg.norm(direction)), MIN_CONTRAST_NORM)


def _fuse(candidates: list[str], shown: list[str], direction: np.ndarray, tier: str) -> list[str]:
    """Re-order lexical candidates by fusing in their embedding similarity."""
    similarity = index.embed_texts(shown, tier) @ direction
    lexical = list(range(len(candidates)))
    dense = [int(i) for i in np.argsort(-similarity, kind="stable")]
    fused = rrf([lexical, dense])
    return [candidates[i] for i in sorted(fused, key=lambda i: (-fused[i], i))]


def prune(ranked: list[str], counts: Counter[str], top: int) -> list[str]:
    """Drop terms that only repeat a better one, and keep the best `top`.

    A word is dropped when most of its occurrences are inside a candidate
    phrase, or when a phrase ranked above it already contains it. Otherwise
    "San Miguel", "San", and "Miguel" would take three of ten places. Any term
    is dropped once its words are already in `MAX_TERMS_PER_WORD` better terms.
    """
    phrases = [term for term in ranked if " " in term]
    kept: list[str] = []
    uses: Counter[str] = Counter()
    for term in ranked:
        if len(kept) >= top:
            break
        if any(uses[word] >= MAX_TERMS_PER_WORD for word in term.split()):
            continue
        if " " not in term:
            absorbed = any(
                term in phrase.split() and counts[phrase] >= ABSORB_RATIO * counts[term]
                for phrase in phrases
            )
            repeated = any(term in other.split() for other in kept if " " in other)
            if absorbed or repeated:
                continue
        kept.append(term)
        uses.update(term.split())
    return kept


def _densest_page(pages: Counter[PageKey], order: dict[PageKey, int]) -> PageKey:
    """The page with the most occurrences; the earliest one on a tie."""
    return min(pages, key=lambda page: (-pages[page], order[page]))


def _build_terms(
    chapter: _Chapter,
    ranked: list[str],
    display: dict[str, str],
    printed: dict[PageKey, int | None],
    order: dict[PageKey, int],
) -> list[Term]:
    terms = []
    for position, term in enumerate(ranked, start=1):
        source_id, pdf_page = _densest_page(chapter.pages[term], order)
        terms.append(
            Term(
                text=display[term],
                count=chapter.counts[term],
                # Rank-based, so lexical-only and re-ranked runs are comparable.
                score=1.0 / position,
                source_id=source_id,
                pdf_page=pdf_page,
                printed_page=printed[(source_id, pdf_page)],
            )
        )
    return terms


def _chapter_vectors(chapter: _Chapter, vectors: dict[str, np.ndarray]) -> np.ndarray:
    return np.vstack([vectors[source_id][chunk_id] for source_id, chunk_id in chapter.rows])


def extract_terms(
    chunks_by_source: dict[str, list[Chunk]],
    metas: dict[str, SourceMeta],
    *,
    vectors: dict[str, np.ndarray] | None = None,
    tier: str = DEFAULT_TIER,
    top: int = DEFAULT_TOP,
    pool: int = CANDIDATE_POOL,
) -> BookTerms:
    """Key terms for each chapter of one book.

    `chunks_by_source` holds every source of the book, in reading order.
    Pass `vectors` (row i of a source's matrix is its chunk i) to re-rank the
    lexical candidates by embedding; leave it None for a lexical-only ranking
    that loads no model.

    Raises ValueError if `top` is below 1.
    """
    if top < 1:
        raise ValueError(f"top must be at least 1, got {top}")

    first = next(iter(metas.values()), None)
    book = BookTerms(title=first.title if first else "", author=first.author if first else "")
    chapters = _collect(chunks_by_source, metas)
    if not chapters:
        return book

    canonical = _fold_plurals(_vocabulary(chapters))
    spellings: dict[str, Counter[str]] = {}
    for chapter in chapters:
        _count(chapter, canonical, spellings)
    display = _display_forms(spellings)

    totals: Counter[str] = Counter()
    for chapter in chapters:
        totals.update(chapter.counts)
    book_size = sum(chapter.size for chapter in chapters)
    book.contrasted = len(chapters) > 1

    all_chunks = [item for chunks in chunks_by_source.values() for item in chunks]
    printed = {(c.source_id, c.pdf_page): c.printed_page for c in all_chunks}
    order = {page: position for position, page in enumerate(printed)}
    book_vectors = np.vstack(list(vectors.values())) if vectors else None

    for chapter in chapters:
        ranked = _candidates(chapter, totals, book_size, pool)
        if ranked and vectors is not None and book_vectors is not None:
            direction = chapter_direction(_chapter_vectors(chapter, vectors), book_vectors)
            ranked = _fuse(ranked, [display[t] for t in ranked], direction, tier)
        ranked = prune(ranked, chapter.counts, top)
        book.chapters.append(
            ChapterTerms(
                label=chapter.label,
                titled=chapter.titled,
                source_ids=chapter.source_ids,
                terms=_build_terms(chapter, ranked, display, printed, order),
            )
        )
    if vectors is not None:
        book.tier = tier
    return book


def group_books(sources: list[SourceMeta]) -> list[list[SourceMeta]]:
    """Group sources into books: same title and author, compared caselessly.

    Matching is exact apart from case and surrounding space. A typo in one
    source's title splits it into its own book; `dsearch edit` fixes that.
    """
    books: dict[tuple[str, str], list[SourceMeta]] = {}
    for meta in sources:
        key = (meta.title.strip().casefold(), meta.author.strip().casefold())
        books.setdefault(key, []).append(meta)
    return list(books.values())


def _load_vectors(group: list[SourceMeta], tier: str) -> dict[str, np.ndarray] | None:
    """Every source's vectors at `tier`, or None if any source lacks them."""
    vectors = {}
    for meta in group:
        try:
            matrix = index.load_vectors(meta.source_id, tier)
        except SourceNotFoundError:
            return None
        if matrix.shape[0] != meta.chunk_count:
            return None
        vectors[meta.source_id] = matrix
    return vectors


def key_terms(
    *,
    source: str | None = None,
    tier: str = DEFAULT_TIER,
    top: int = DEFAULT_TOP,
    rerank: bool = True,
) -> list[BookTerms]:
    """Key terms for every book in the library, or the book holding `source`.

    `source` narrows what is *reported* to that source's chapters. The rest of
    its book is still read, because a chapter's terms are only distinctive
    relative to the chapters around it.

    A book that lacks `tier` vectors is ranked lexically and says so in its
    `note`, rather than starting an embedding run to list some words.

    Raises SourceNotFoundError or AmbiguousSourceError for a bad `source`, and
    ValueError for an unknown tier.
    """
    index._validate_tier(tier)
    only = index.resolve_source(source) if source else None

    books = []
    for group in group_books(index.list_sources()):
        if only and only.source_id not in {meta.source_id for meta in group}:
            continue
        metas = {meta.source_id: meta for meta in group}
        chunks = {meta.source_id: index.load_chunks(meta.source_id) for meta in group}
        vectors = _load_vectors(group, tier) if rerank else None
        book = extract_terms(chunks, metas, vectors=vectors, tier=tier, top=top)
        if rerank and vectors is None:
            book.note = (
                f"No {tier} vectors for every source of this book, so terms are ranked by "
                f"word counts alone. Run `dsearch embed --tier {tier}` to re-rank them."
            )
        if only:
            book.chapters = [c for c in book.chapters if only.source_id in c.source_ids]
        books.append(book)
    return books
