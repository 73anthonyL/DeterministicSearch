"""Tests for hybrid retrieval: fusion, long queries, filtering, and context.

Embeddings are stubbed with a deterministic bag-of-words vector so ranking is
predictable and no model download is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from dsearch import index, search
from dsearch.chunk import Chunk
from dsearch.index import SourceMeta

VOCAB = [
    "strawberry",
    "border",
    "clinic",
    "pain",
    "foreman",
    "housing",
    "wage",
    "knee",
]


def bag_embed(texts, tier=index.DEFAULT_TIER, *, pages=None, page_count=None, progress=None):
    """A tiny deterministic embedder: one dimension per vocabulary word."""
    out = np.zeros((len(texts), len(VOCAB)), dtype=np.float32)
    for row, text in enumerate(texts):
        lowered = text.lower()
        for column, word in enumerate(VOCAB):
            out[row, column] = float(lowered.count(word))
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-9)


@pytest.fixture(autouse=True)
def stub_embeddings(monkeypatch):
    monkeypatch.setattr(search, "embed_texts", bag_embed)
    monkeypatch.setattr(index, "embed_texts", bag_embed)


def make_chunk(chunk_id: int, sentences: list[str], source_id="src1", page=1, printed=None):
    return Chunk(
        chunk_id=chunk_id,
        source_id=source_id,
        text=" ".join(sentences),
        sentences=sentences,
        pdf_page=page,
        printed_page=printed,
        chapter="Chapter 1",
    )


def make_corpus(chunks: list[Chunk], titles: dict[str, str] | None = None) -> search.Corpus:
    titles = titles or {"src1": "Fresh Fruit, Broken Bodies"}
    metas = {
        sid: SourceMeta(
            source_id=sid,
            filename=f"{sid}.pdf",
            title=title,
            author="Seth Holmes",
            added="2026-09-24",
            page_count=10,
        )
        for sid, title in titles.items()
    }
    return search.Corpus(
        chunks=chunks,
        vectors=bag_embed([c.text for c in chunks]),
        metas=metas,
    )


@pytest.fixture
def corpus():
    return make_corpus(
        [
            make_chunk(0, ["The strawberry rows stretched on.", "Pickers bent low."], printed=10),
            make_chunk(
                1, ["Crossing the border took three nights.", "The desert was cold."], printed=11
            ),
            make_chunk(
                2, ["The clinic dismissed his knee pain.", "No one examined the knee."], printed=12
            ),
            make_chunk(
                3, ["Housing for pickers was a shed.", "The wage barely covered rent."], printed=13
            ),
            make_chunk(
                4, ["A foreman watched the rows.", "He said nothing about the wage."], printed=14
            ),
        ]
    )


class TestTokenize:
    def test_lowercases_and_drops_punctuation(self):
        assert search.tokenize("The Clinic's knee-pain!") == ["the", "clinic", "s", "knee", "pain"]

    def test_empty(self):
        assert search.tokenize("   ") == []


class TestSearchBasics:
    def test_finds_the_relevant_chunk(self, corpus):
        results = search.search("knee pain at the clinic", k=1, corpus=corpus)
        assert len(results) == 1
        assert "clinic" in results[0].chunk.text.lower()

    def test_respects_k(self, corpus):
        assert len(search.search("strawberry", k=3, corpus=corpus)) == 3

    def test_k_larger_than_corpus_returns_everything(self, corpus):
        assert len(search.search("strawberry border clinic", k=50, corpus=corpus)) == len(corpus)

    def test_results_are_sorted_by_descending_score(self, corpus):
        results = search.search("wage housing", k=5, corpus=corpus)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_query_returns_nothing(self, corpus):
        assert search.search("   ", corpus=corpus) == []

    def test_empty_corpus_returns_nothing(self):
        assert search.search("anything", corpus=search.Corpus()) == []

    def test_rejects_non_positive_k(self, corpus):
        with pytest.raises(ValueError, match="k must be at least 1"):
            search.search("strawberry", k=0, corpus=corpus)

    def test_results_are_verbatim_spans_of_the_source(self, corpus):
        # The core guarantee: nothing is generated.
        texts = {c.text for c in corpus.chunks}
        for result in search.search("strawberry wage", k=5, corpus=corpus):
            assert result.chunk.text in texts


class TestHybridFusion:
    def test_keyword_only_match_still_surfaces(self, corpus):
        # "shed" is outside the embedding vocabulary, so only BM25 can find it.
        results = search.search("shed", k=1, corpus=corpus)
        assert "shed" in results[0].chunk.text.lower()

    def test_semantic_match_survives_when_bm25_is_silent(self, corpus):
        # A proper noun absent from the corpus leaves BM25 with nothing; the
        # dense ranker still returns its best guess rather than an empty list.
        assert search.search("Oaxaca", k=2, corpus=corpus)

    def test_fusion_beats_either_ranker_alone(self, corpus):
        # A chunk matching on both keyword and vector should outrank one that
        # matches on only a single ranker.
        results = search.search("clinic knee pain", k=5, corpus=corpus)
        assert results[0].chunk.chunk_id == 2


class TestBestSentence:
    def test_picks_the_matching_sentence(self, corpus):
        result = search.search("border crossing", k=1, corpus=corpus)[0]
        assert result.chunk.chunk_id == 1
        assert result.best_sentence == 0
        assert "border" in result.best_sentence_text

    def test_picks_a_later_sentence_when_that_is_the_match(self, corpus):
        # Chunk 2's second sentence is purely about the knee, so it beats the
        # first, which splits its weight between "knee" and "pain".
        result = search.search("knee", k=1, corpus=corpus)[0]
        assert result.chunk.chunk_id == 2
        assert result.best_sentence == 1
        assert result.best_sentence_text == "No one examined the knee."

    def test_index_is_always_in_range(self, corpus):
        for result in search.search("strawberry border clinic wage", k=5, corpus=corpus):
            assert 0 <= result.best_sentence < len(result.chunk.sentences)

    def test_single_sentence_chunk(self):
        corpus = make_corpus([make_chunk(0, ["Only one sentence about the clinic."])])
        result = search.search("clinic", k=1, corpus=corpus)[0]
        assert result.best_sentence == 0
        assert result.best_sentence_text == "Only one sentence about the clinic."


class TestLongQuery:
    def test_long_query_is_scored_per_sentence(self, corpus, monkeypatch):
        seen: list[list[str]] = []
        original = search.embed_texts

        def spy(texts, *args, **kwargs):
            seen.append(list(texts))
            return original(texts, *args, **kwargs)

        monkeypatch.setattr(search, "embed_texts", spy)

        filler = "This is padding about nothing in particular. " * 40
        long_query = filler + "What did the clinic say about his knee pain?"
        assert len(long_query.split()) > search.LONG_QUERY_WORDS
        search.search(long_query, k=1, corpus=corpus)
        assert len(seen[0]) > 1  # Split into sentences, not embedded whole.

    def test_short_query_is_embedded_whole(self, corpus, monkeypatch):
        seen: list[list[str]] = []
        original = search.embed_texts
        monkeypatch.setattr(
            search,
            "embed_texts",
            lambda texts, *a, **kw: (seen.append(list(texts)), original(texts, *a, **kw))[1],
        )
        search.search("knee pain", k=1, corpus=corpus)
        assert len(seen[0]) == 1

    def test_one_relevant_sentence_in_a_long_paragraph_still_retrieves(self, corpus):
        # Max-over-sentences is the point: the signal must not be averaged away.
        filler = "I have been thinking about method and about writing generally. " * 30
        query = filler + " The clinic dismissed his knee pain entirely."
        assert len(query.split()) > search.LONG_QUERY_WORDS
        result = search.search(query, k=1, corpus=corpus)[0]
        assert result.chunk.chunk_id == 2


class TestNeighbours:
    def test_middle_chunk_has_both_neighbours(self, corpus):
        before, after = corpus.neighbours(2)
        assert before.chunk_id == 1
        assert after.chunk_id == 3

    def test_first_chunk_has_no_before(self, corpus):
        before, after = corpus.neighbours(0)
        assert before is None
        assert after.chunk_id == 1

    def test_last_chunk_has_no_after(self, corpus):
        before, after = corpus.neighbours(len(corpus) - 1)
        assert before.chunk_id == len(corpus) - 2
        assert after is None

    def test_context_does_not_cross_a_source_boundary(self):
        corpus = make_corpus(
            [
                make_chunk(0, ["Book one ends here about the wage."], source_id="src1"),
                make_chunk(0, ["Book two begins here about the clinic."], source_id="src2"),
            ],
            titles={"src1": "Book One", "src2": "Book Two"},
        )
        assert corpus.neighbours(0)[1] is None
        assert corpus.neighbours(1)[0] is None

    def test_results_carry_context(self, corpus):
        result = search.search("clinic knee", k=1, corpus=corpus)[0]
        assert result.before is not None and result.after is not None


class TestPageLabel:
    def test_uses_printed_page_when_known(self):
        corpus = make_corpus([make_chunk(0, ["About the clinic."], page=3, printed=47)])
        assert search.search("clinic", k=1, corpus=corpus)[0].page_label == "p. 47 (PDF 3)"

    def test_falls_back_to_pdf_page(self):
        corpus = make_corpus([make_chunk(0, ["About the clinic."], page=3, printed=None)])
        assert search.search("clinic", k=1, corpus=corpus)[0].page_label == "PDF p. 3"


class TestCorpusLoading:
    """Loading from a real on-disk library."""

    @pytest.fixture(autouse=True)
    def library_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DSEARCH_HOME", str(tmp_path / "lib"))

    @pytest.fixture
    def two_books(self, pdf_factory):
        body_a = (
            "The clinic dismissed his knee pain. No one examined the knee at all. "
            "He returned to the strawberry rows the next morning before dawn."
        )
        body_b = (
            "Crossing the border took three nights on foot. The desert was cold and open. "
            "A foreman later set the wage for every picker in the shed."
        )
        a = index.add(pdf_factory(name="one.pdf", pages=[body_a] * 3), title="Book One")
        b = index.add(pdf_factory(name="two.pdf", pages=[body_b] * 3), title="Book Two")
        return a.meta, b.meta

    def test_loads_every_source(self, two_books):
        corpus = search.load_corpus("fast")
        assert len(corpus.metas) == 2
        assert len(corpus) > 0
        assert corpus.vectors.shape[0] == len(corpus)

    def test_empty_library(self):
        corpus = search.load_corpus("fast")
        assert len(corpus) == 0
        assert search.search("anything", corpus=corpus) == []

    def test_filters_to_one_source(self, two_books):
        one, _ = two_books
        corpus = search.load_corpus("fast", source="Book One")
        assert set(corpus.metas) == {one.source_id}
        assert all(c.source_id == one.source_id for c in corpus.chunks)

    def test_search_honours_the_source_filter(self, two_books):
        one, _ = two_books
        results = search.search("border desert", k=3, source="Book One")
        assert results
        assert all(r.meta.source_id == one.source_id for r in results)

    def test_unknown_source_filter_raises(self, two_books):
        with pytest.raises(index.SourceNotFoundError):
            search.load_corpus("fast", source="Book Three")

    def test_source_without_the_tier_is_skipped_not_fatal(self, two_books):
        # Searching at a tier only one book has must still search that book.
        one, _ = two_books
        index.add_result = None
        corpus = search.load_corpus("best")
        assert len(corpus) == 0
        assert len(corpus.skipped) == 2

    def test_results_carry_source_metadata(self, two_books):
        result = search.search("knee pain clinic", k=1)[0]
        assert result.meta.title in {"Book One", "Book Two"}
        assert result.meta.author == ""
