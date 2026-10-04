"""Hybrid retrieval: dense embeddings and BM25, fused by reciprocal rank.

Nothing here generates text. A result is always a verbatim span of the source
PDF, addressed by source, page, and sentence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from dsearch.chunk import Chunk, split_sentences
from dsearch.index import (
    DEFAULT_TIER,
    SourceMeta,
    SourceNotFoundError,
    embed_texts,
    list_sources,
    load_chunks,
    load_vectors,
    resolve_source,
)

# Above this many words a query is treated as a paragraph of thinking rather
# than a single question, and is scored sentence by sentence.
LONG_QUERY_WORDS = 200

# Reciprocal rank fusion constant. 60 is the value from the original RRF paper;
# it damps the influence of the very top ranks so one ranker cannot dominate.
RRF_K = 60

# How many candidates each ranker contributes to the fusion pool. Wider than k
# so that a chunk ranked well by only one of the two still surfaces.
CANDIDATE_POOL = 200

DEFAULT_K = 5

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class Result:
    """One retrieved passage, with the context needed to read and cite it."""

    chunk: Chunk
    score: float
    best_sentence: int
    meta: SourceMeta
    before: Chunk | None = None
    after: Chunk | None = None

    @property
    def best_sentence_text(self) -> str:
        """The sentence that actually matched — the line worth quoting."""
        if 0 <= self.best_sentence < len(self.chunk.sentences):
            return self.chunk.sentences[self.best_sentence]
        return self.chunk.text

    @property
    def page_label(self) -> str:
        """How the page is cited: printed folio when known, PDF page otherwise."""
        if self.chunk.printed_page is not None:
            return f"p. {self.chunk.printed_page} (PDF {self.chunk.pdf_page})"
        return f"PDF p. {self.chunk.pdf_page}"


@dataclass
class Corpus:
    """Every chunk and vector loaded for one search, plus its source metadata."""

    chunks: list[Chunk] = field(default_factory=list)
    vectors: np.ndarray | None = None
    metas: dict[str, SourceMeta] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.chunks)

    def neighbours(self, position: int) -> tuple[Chunk | None, Chunk | None]:
        """The chunks on either side of `position`, within the same source.

        Neighbours give a reader enough context to judge a passage without
        opening the PDF. They stop at a source boundary so context never bleeds
        from one book into another.
        """
        target = self.chunks[position]
        before = after = None
        if position > 0 and self.chunks[position - 1].source_id == target.source_id:
            before = self.chunks[position - 1]
        if (
            position + 1 < len(self.chunks)
            and self.chunks[position + 1].source_id == target.source_id
        ):
            after = self.chunks[position + 1]
        return before, after


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, for BM25."""
    return _TOKEN_RE.findall(text.lower())


def load_corpus(tier: str = DEFAULT_TIER, source: str | None = None) -> Corpus:
    """Load chunks and vectors for the whole library, or one source.

    A source that has no vectors at this tier is skipped and named in
    `Corpus.skipped` rather than failing the search: one un-upgraded book
    should not block the rest of the library.
    """
    metas = [resolve_source(source)] if source else list_sources()

    corpus = Corpus()
    blocks: list[np.ndarray] = []
    for meta in metas:
        try:
            vectors = load_vectors(meta.source_id, tier)
            chunks = load_chunks(meta.source_id)
        except SourceNotFoundError:
            corpus.skipped.append(meta.title or meta.filename)
            continue
        if len(chunks) != vectors.shape[0]:
            corpus.skipped.append(f"{meta.title or meta.filename} (index out of sync)")
            continue
        corpus.chunks.extend(chunks)
        blocks.append(vectors)
        corpus.metas[meta.source_id] = meta

    corpus.vectors = np.vstack(blocks) if blocks else None
    return corpus


def _query_matrix(query: str, tier: str) -> np.ndarray:
    """Embed the query as one or more unit vectors.

    A short query is one vector. A long one — a paragraph of the user's own
    thinking — is split into sentences and embedded per sentence, so a single
    relevant thought is not averaged away by the rest of the paragraph.
    """
    if len(query.split()) > LONG_QUERY_WORDS:
        parts = [s for s in split_sentences(query) if s.strip()] or [query]
    else:
        parts = [query]
    return embed_texts(parts, tier)


def _dense_scores(query_vectors: np.ndarray, corpus_vectors: np.ndarray) -> np.ndarray:
    """Max cosine similarity of each chunk against any query vector.

    Both sides are L2-normalised, so the dot product is the cosine. Taking the
    max (not the mean) is what makes a long query behave like a set of separate
    questions.
    """
    return (corpus_vectors @ query_vectors.T).max(axis=1)


def _bm25_scores(query: str, corpus: Corpus) -> np.ndarray:
    """BM25 score of every chunk against the query's tokens."""
    from rank_bm25 import BM25Okapi

    tokens = tokenize(query)
    if not tokens:
        return np.zeros(len(corpus), dtype=np.float32)
    bm25 = BM25Okapi([tokenize(c.text) for c in corpus.chunks])
    return np.asarray(bm25.get_scores(tokens), dtype=np.float32)


def _rank(scores: np.ndarray, pool: int, *, positive_only: bool) -> list[int]:
    """Best `pool` chunk positions under `scores`, best first.

    `positive_only` distinguishes the two rankers. A BM25 score of zero means
    not one query term appears in the chunk — no evidence at all, so it earns no
    rank. A low cosine is still a meaningful ordering, so the dense ranker ranks
    every chunk.
    """
    order = np.argsort(-scores)[:pool]
    if positive_only:
        return [int(i) for i in order if scores[i] > 0]
    return [int(i) for i in order]


def rrf(rankings: list[list[int]]) -> dict[int, float]:
    """Fuse ranked candidate lists by reciprocal rank.

    RRF combines rankings rather than raw scores, which matters because a cosine
    in [-1, 1] and an unbounded BM25 score cannot be added meaningfully.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, position in enumerate(ranking, start=1):
            fused[position] = fused.get(position, 0.0) + 1.0 / (RRF_K + rank)
    return fused


def _best_sentence(chunk: Chunk, query_vectors: np.ndarray, tier: str) -> int:
    """Index of the chunk sentence closest to any part of the query."""
    if len(chunk.sentences) <= 1:
        return 0
    sentence_vectors = embed_texts(list(chunk.sentences), tier)
    return int((sentence_vectors @ query_vectors.T).max(axis=1).argmax())


def search(
    query: str,
    k: int = DEFAULT_K,
    *,
    tier: str = DEFAULT_TIER,
    source: str | None = None,
    corpus: Corpus | None = None,
) -> list[Result]:
    """Retrieve the `k` passages that best answer `query`.

    Pass a pre-loaded `corpus` to run several searches without re-reading the
    library from disk. Raises SourceNotFoundError if `source` matches nothing.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")

    query = query.strip()
    if not query:
        return []

    if corpus is None:
        corpus = load_corpus(tier, source)
    if not len(corpus) or corpus.vectors is None:
        return []

    query_vectors = _query_matrix(query, tier)
    dense = _dense_scores(query_vectors, corpus.vectors)
    keyword = _bm25_scores(query, corpus)
    fused = rrf(
        [
            _rank(dense, CANDIDATE_POOL, positive_only=False),
            _rank(keyword, CANDIDATE_POOL, positive_only=True),
        ]
    )
    if not fused:
        return []

    top = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:k]

    results = []
    for position, score in top:
        chunk_at = corpus.chunks[position]
        before, after = corpus.neighbours(position)
        results.append(
            Result(
                chunk=chunk_at,
                score=float(score),
                best_sentence=_best_sentence(chunk_at, query_vectors, tier),
                meta=corpus.metas[chunk_at.source_id],
                before=before,
                after=after,
            )
        )
    return results
