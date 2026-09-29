"""Tests for key-term extraction: counting, scoring, pruning, and the re-rank.

Chunks are built with the real chunker so the overlap being de-duplicated is
the overlap the library actually stores. Embeddings are stubbed, so nothing
here downloads a model.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from dsearch import index, terms
from dsearch.chunk import Chunk, chunk
from dsearch.extract import Page
from dsearch.index import SourceMeta

FARM = (
    "The crew boss watched the pickers. "
    "Pickers filled each strawberry flat. "
    "The crew boss weighed the strawberry flat. "
    "Workers rested at noon."
)
CLINIC = (
    "The physician examined the patient. "
    "A patient waited at the clinic. "
    "The physician left the clinic early. "
    "Workers rested at noon."
)
PAGES_PER_CHAPTER = 3


def make_meta(source_id: str, title="Fresh Fruit, Broken Bodies", author="Seth Holmes", **extra):
    return SourceMeta(
        source_id=source_id,
        filename=f"{source_id}.pdf",
        title=title,
        author=author,
        added="2026-09-28",
        page_count=PAGES_PER_CHAPTER,
        **extra,
    )


def make_chunks(source_id: str, bodies: list[str], chapter: str | None, first_page=1, folio=None):
    """Chunk one body per page with the library's default size and overlap."""
    pages = [
        Page(
            pdf_page=first_page + offset,
            text=body,
            printed_page=None if folio is None else folio + offset,
            chapter=chapter,
        )
        for offset, body in enumerate(bodies)
    ]
    return chunk(pages, source_id=source_id)


@pytest.fixture
def split_book():
    """One book added as two single-chapter PDFs, like the sample library."""
    chunks = {
        "farm": make_chunks("farm", [FARM] * PAGES_PER_CHAPTER, "Segregation on the Farm"),
        "clinic": make_chunks("clinic", [CLINIC] * PAGES_PER_CHAPTER, "Doctors", folio=111),
    }
    metas = {"farm": make_meta("farm"), "clinic": make_meta("clinic")}
    return chunks, metas


def texts_of(chapter: terms.ChapterTerms) -> list[str]:
    return [term.text.lower() for term in chapter.terms]


def term_named(chapter: terms.ChapterTerms, text: str) -> terms.Term:
    return next(term for term in chapter.terms if term.text.lower() == text)


class TestUniqueSentences:
    def test_overlapping_chunks_yield_each_sentence_once(self):
        chunks = make_chunks("s", [FARM], "One")
        assert len(chunks) == 2  # The precondition: the windows really overlap.
        sentences = [sentence for _, sentence in terms.unique_sentences(chunks, overlap=1)]
        assert sentences == [
            "The crew boss watched the pickers.",
            "Pickers filled each strawberry flat.",
            "The crew boss weighed the strawberry flat.",
            "Workers rested at noon.",
        ]

    def test_a_new_page_starts_fresh(self):
        chunks = make_chunks("s", [FARM, FARM], "One")
        sentences = [sentence for _, sentence in terms.unique_sentences(chunks, overlap=1)]
        assert len(sentences) == 8

    def test_a_new_source_starts_fresh_even_on_the_same_page_number(self):
        chunks = make_chunks("a", ["One. Two."], "One") + make_chunks("b", ["Three. Four."], "One")
        sentences = [sentence for _, sentence in terms.unique_sentences(chunks, overlap=1)]
        assert sentences == ["One.", "Two.", "Three.", "Four."]

    def test_zero_overlap_skips_nothing(self):
        pages = [Page(pdf_page=1, text=FARM)]
        chunks = chunk(pages, size=2, overlap=0, source_id="s")
        sentences = [sentence for _, sentence in terms.unique_sentences(chunks, overlap=0)]
        assert len(sentences) == 4

    def test_yields_the_chunk_the_sentence_came_from(self):
        chunks = make_chunks("s", [FARM], "One")
        owners = [item.chunk_id for item, _ in terms.unique_sentences(chunks, overlap=1)]
        assert owners == [0, 0, 0, 1]


class TestSegmentWords:
    def test_punctuation_separates_segments(self):
        assert terms.segment_words("The workers, farm owners; and crews.") == [
            ["The", "workers"],
            ["farm", "owners"],
            ["and", "crews"],
        ]

    def test_a_dash_separates_segments(self):
        assert terms.segment_words("The pickers — mostly Triqui — stayed") == [
            ["The", "pickers"],
            ["mostly", "Triqui"],
            ["stayed"],
        ]

    def test_keeps_apostrophes_and_hyphens_inside_a_word(self):
        assert terms.segment_words("Doctors don’t see farm-workers") == [
            ["Doctors", "don’t", "see", "farm-workers"]
        ]

    def test_drops_numbers(self):
        assert terms.segment_words("In 2004 about 95 percent") == [["In", "about", "percent"]]

    def test_keeps_accented_letters(self):
        assert terms.segment_words("San Miguel, Oaxaca, México") == [
            ["San", "Miguel"],
            ["Oaxaca"],
            ["México"],
        ]

    def test_empty(self):
        assert terms.segment_words("  ... ") == []


class TestFoldPlurals:
    def fold(self, *words: str) -> dict[str, str]:
        return terms._fold_plurals(set(words))

    def test_a_plural_is_counted_with_its_singular(self):
        assert self.fold("patient", "patients")["patients"] == "patient"

    def test_a_plural_in_ies_is_counted_with_its_singular(self):
        assert self.fold("berry", "berries")["berries"] == "berry"

    def test_a_plural_with_no_singular_in_the_book_is_kept(self):
        assert self.fold("pickers")["pickers"] == "pickers"

    def test_a_double_s_is_not_a_plural(self):
        assert self.fold("bos", "boss")["boss"] == "boss"

    def test_a_short_stem_is_not_folded(self):
        assert self.fold("a", "as", "it", "its")["its"] == "its"


class TestDistinctiveness:
    def test_a_term_no_more_common_than_elsewhere_scores_zero(self):
        assert terms.distinctiveness(count=5, size=100, rest_count=5, rest_size=100) == 0.0

    def test_a_term_rarer_than_elsewhere_scores_zero(self):
        assert terms.distinctiveness(count=2, size=100, rest_count=20, rest_size=100) == 0.0

    def test_a_concentrated_term_scores_above_a_spread_one(self):
        concentrated = terms.distinctiveness(count=10, size=100, rest_count=0, rest_size=400)
        spread = terms.distinctiveness(count=10, size=100, rest_count=20, rest_size=400)
        assert concentrated > spread > 0

    def test_a_frequent_term_scores_above_a_rare_one_equally_concentrated(self):
        frequent = terms.distinctiveness(count=20, size=100, rest_count=0, rest_size=400)
        rare = terms.distinctiveness(count=4, size=100, rest_count=0, rest_size=400)
        assert frequent > rare

    def test_a_term_found_nowhere_else_has_a_finite_score(self):
        score = terms.distinctiveness(count=10, size=100, rest_count=0, rest_size=400)
        assert np.isfinite(score)

    def test_with_no_other_chapter_the_score_is_the_plain_rate(self):
        assert terms.distinctiveness(count=5, size=100, rest_count=0, rest_size=0) == 0.05

    def test_an_empty_chapter_scores_zero(self):
        assert terms.distinctiveness(count=0, size=0, rest_count=3, rest_size=10) == 0.0


class TestPrune:
    def test_a_word_mostly_inside_a_phrase_is_dropped_for_the_phrase(self):
        counts = Counter({"miguel": 10, "san miguel": 9, "san": 9})
        assert terms.prune(["miguel", "san miguel", "san"], counts, top=10) == ["san miguel"]

    def test_a_word_used_mostly_outside_the_phrase_survives_above_it(self):
        counts = Counter({"border": 40, "border patrol": 8})
        kept = terms.prune(["border", "border patrol"], counts, top=10)
        assert kept == ["border", "border patrol"]

    def test_a_word_ranked_below_a_phrase_containing_it_is_dropped(self):
        counts = Counter({"border patrol": 8, "patrol": 20})
        assert terms.prune(["border patrol", "patrol"], counts, top=10) == ["border patrol"]

    def test_a_word_appears_in_a_limited_number_of_terms(self):
        ranked = ["embodied", "embodied anthropology", "embodied experiences", "violence"]
        counts = Counter({"embodied": 40, "embodied anthropology": 5, "embodied experiences": 5})
        assert terms.prune(ranked, counts, top=10) == [
            "embodied",
            "embodied anthropology",
            "violence",
        ]

    def test_keeps_only_the_best_top(self):
        counts = Counter({"a": 3, "b": 3, "c": 3})
        assert terms.prune(["a", "b", "c"], counts, top=2) == ["a", "b"]

    def test_dropped_terms_do_not_use_up_places(self):
        counts = Counter({"san miguel": 9, "miguel": 9, "coyote": 5})
        assert terms.prune(["san miguel", "miguel", "coyote"], counts, top=2) == [
            "san miguel",
            "coyote",
        ]


class TestChapterDirection:
    def test_points_away_from_the_rest_of_the_book(self):
        chapter = np.array([[1.0, 0.0]])
        book = np.array([[1.0, 0.0], [0.0, 1.0]])
        direction = terms.chapter_direction(chapter, book)
        assert direction[0] > 0 > direction[1]

    def test_is_a_unit_vector(self):
        chapter = np.array([[0.6, 0.8], [1.0, 0.0]])
        book = np.array([[0.6, 0.8], [1.0, 0.0], [0.0, 1.0]])
        assert np.linalg.norm(terms.chapter_direction(chapter, book)) == pytest.approx(1.0)

    def test_a_one_chapter_book_falls_back_to_the_centroid(self):
        vectors = np.array([[0.0, 1.0], [0.0, 1.0]])
        assert terms.chapter_direction(vectors, vectors) == pytest.approx(np.array([0.0, 1.0]))


class TestExtractTerms:
    def test_each_chapter_gets_its_own_terms(self, split_book):
        book = terms.extract_terms(*split_book)
        farm, clinic = book.chapters
        assert {"crew boss", "pickers", "strawberry flat"} <= set(texts_of(farm))
        assert {"physician", "patient", "clinic"} <= set(texts_of(clinic))

    def test_terms_shared_by_every_chapter_are_left_out(self, split_book):
        book = terms.extract_terms(*split_book)
        for chapter in book.chapters:
            assert not {"workers", "rested", "noon"} & set(texts_of(chapter))

    def test_stopwords_are_never_terms(self, split_book):
        book = terms.extract_terms(*split_book)
        for chapter in book.chapters:
            assert not {"the", "each", "at"} & set(texts_of(chapter))

    def test_counts_are_not_inflated_by_chunk_overlap(self, split_book):
        # "crew boss" is in sentences 1 and 3 of each page. Sentence 3 is in
        # both of the page's chunks, so counting chunk text would give 9.
        farm = terms.extract_terms(*split_book).chapters[0]
        assert term_named(farm, "crew boss").count == 2 * PAGES_PER_CHAPTER

    def test_a_plural_and_its_singular_are_one_term(self):
        body = "The patient waited. Two patients left. Another patient came. Patients talked."
        chunks = {
            "a": make_chunks("a", [body], "One"),
            "b": make_chunks("b", [FARM], "Two"),
        }
        metas = {"a": make_meta("a"), "b": make_meta("b")}
        chapter = terms.extract_terms(chunks, metas).chapters[0]
        matching = [term for term in chapter.terms if term.text.lower().startswith("patient")]
        assert len(matching) == 1
        assert matching[0].count == 4

    def test_a_term_is_shown_in_its_most_common_spelling(self):
        body = "They left San Miguel. San Miguel was far. He missed San Miguel. It rained."
        chunks = {
            "a": make_chunks("a", [body], "One"),
            "b": make_chunks("b", [FARM], "Two"),
        }
        metas = {"a": make_meta("a"), "b": make_meta("b")}
        chapter = terms.extract_terms(chunks, metas).chapters[0]
        shown = [term.text for term in chapter.terms]
        assert "San Miguel" in shown
        assert "Miguel" not in shown
        assert "San" not in shown

    def test_a_phrase_is_shown_as_it_was_written_not_word_by_word(self):
        # "patrol" alone is mostly capitalised, "border" alone mostly is not;
        # joining each word's commonest spelling would print "border Patrol".
        body = (
            "The border patrol came. The border patrol left. A border patrol truck passed. "
            "The Patrol waited. The Patrol watched. The Patrol moved. The Patrol slept."
        )
        chunks = {
            "a": make_chunks("a", [body], "One"),
            "b": make_chunks("b", [FARM], "Two"),
        }
        metas = {"a": make_meta("a"), "b": make_meta("b")}
        shown = [term.text for term in terms.extract_terms(chunks, metas).chapters[0].terms]
        assert "border patrol" in shown
        assert "border Patrol" not in shown

    def test_reports_the_page_where_a_term_is_densest(self):
        dense = CLINIC + " The physician spoke. The physician nodded."
        chunks = {
            "farm": make_chunks("farm", [FARM] * 3, "Farm"),
            "clinic": make_chunks("clinic", [CLINIC, dense, CLINIC], "Doctors", folio=111),
        }
        metas = {"farm": make_meta("farm"), "clinic": make_meta("clinic")}
        clinic = terms.extract_terms(chunks, metas).chapters[1]
        physician = term_named(clinic, "physician")
        assert (physician.source_id, physician.pdf_page, physician.printed_page) == (
            "clinic",
            2,
            112,
        )
        assert physician.page_label == "p. 112 (PDF 2)"

    def test_a_tie_for_densest_page_goes_to_the_earliest(self, split_book):
        clinic = terms.extract_terms(*split_book).chapters[1]
        assert term_named(clinic, "physician").pdf_page == 1

    def test_the_page_label_without_a_folio_names_the_pdf_page(self, split_book):
        farm = terms.extract_terms(*split_book).chapters[0]
        assert term_named(farm, "pickers").page_label == "PDF p. 1"

    def test_chapters_of_one_pdf_are_contrasted_with_each_other(self):
        whole = make_chunks("book", [FARM] * 3, "Farm") + [
            Chunk(
                chunk_id=item.chunk_id + 100,
                source_id="book",
                text=item.text,
                sentences=item.sentences,
                pdf_page=item.pdf_page + 3,
                chapter="Doctors",
            )
            for item in make_chunks("book", [CLINIC] * 3, "Doctors")
        ]
        book = terms.extract_terms({"book": whole}, {"book": make_meta("book")})
        assert [chapter.label for chapter in book.chapters] == ["Farm", "Doctors"]
        assert "physician" in texts_of(book.chapters[1])
        assert "physician" not in texts_of(book.chapters[0])

    def test_pages_without_a_chapter_title_are_grouped_under_the_filename(self):
        chunks = {
            "a": make_chunks("a", [FARM] * 3, None),
            "b": make_chunks("b", [CLINIC] * 3, None),
        }
        metas = {"a": make_meta("a"), "b": make_meta("b")}
        book = terms.extract_terms(chunks, metas)
        assert [(c.label, c.titled) for c in book.chapters] == [("a.pdf", False), ("b.pdf", False)]
        assert "pickers" in texts_of(book.chapters[0])

    def test_a_single_chapter_is_ranked_by_frequency_and_says_so(self):
        chunks = {"a": make_chunks("a", [FARM] * 3, "Farm")}
        book = terms.extract_terms(chunks, {"a": make_meta("a")})
        assert book.contrasted is False
        assert texts_of(book.chapters[0])

    def test_rare_terms_are_not_candidates(self):
        body = FARM + " A coyote appeared."
        chunks = {
            "a": make_chunks("a", [body, FARM, FARM], "Farm"),
            "b": make_chunks("b", [CLINIC] * 3, "Doctors"),
        }
        metas = {"a": make_meta("a"), "b": make_meta("b")}
        assert "coyote" not in texts_of(terms.extract_terms(chunks, metas).chapters[0])

    def test_respects_top(self, split_book):
        book = terms.extract_terms(*split_book, top=2)
        assert all(len(chapter.terms) == 2 for chapter in book.chapters)

    def test_terms_are_in_descending_score_order(self, split_book):
        for chapter in terms.extract_terms(*split_book).chapters:
            scores = [term.score for term in chapter.terms]
            assert scores == sorted(scores, reverse=True)

    def test_every_term_occurs_verbatim_in_the_source(self, split_book):
        # The core guarantee: nothing is generated.
        chunks, metas = split_book
        for chapter in terms.extract_terms(chunks, metas).chapters:
            text = " ".join(c.text for sid in chapter.source_ids for c in chunks[sid]).lower()
            for term in chapter.terms:
                assert term.text.lower() in text

    def test_carries_the_title_and_author(self, split_book):
        book = terms.extract_terms(*split_book)
        assert (book.title, book.author) == ("Fresh Fruit, Broken Bodies", "Seth Holmes")

    def test_rejects_non_positive_top(self, split_book):
        with pytest.raises(ValueError, match="top must be at least 1"):
            terms.extract_terms(*split_book, top=0)

    def test_an_empty_book_has_no_chapters(self):
        assert terms.extract_terms({}, {}).chapters == []

    def test_lexical_only_loads_no_model(self, split_book, monkeypatch):
        def explode(*args, **kwargs):
            raise AssertionError("a lexical-only run must not embed anything")

        monkeypatch.setattr(index, "embed_texts", explode)
        book = terms.extract_terms(*split_book)
        assert book.tier is None


# The stub embedder's axes: one for what the clinic chapter is about, one for
# the farm chapter.
CLINIC_AXIS = np.array([1.0, 0.0], dtype=np.float32)
FARM_AXIS = np.array([0.0, 1.0], dtype=np.float32)


def axis_embed(texts, tier=index.DEFAULT_TIER, **_):
    """Place "clinic" and "patient" on the clinic axis, everything else off it."""
    placed = {"clinic": CLINIC_AXIS, "patient": 0.5 * CLINIC_AXIS + 0.5 * FARM_AXIS}
    return np.vstack([placed.get(text.lower(), FARM_AXIS) for text in texts])


class TestRerank:
    @pytest.fixture
    def book(self):
        # "physician" is the most frequent clinic term, so it leads the lexical
        # ranking; the stub embedder puts it on the farm axis, so the re-rank
        # must move it down.
        body = (
            "The physician spoke. The physician nodded. The physician left. "
            "The physician returned. The clinic opened. The clinic closed. "
            "The clinic moved. The patient waited. A patient left. The patient slept."
        )
        chunks = {
            "farm": make_chunks("farm", [FARM] * 3, "Farm"),
            "clinic": make_chunks("clinic", [body] * 2, "Doctors"),
        }
        metas = {"farm": make_meta("farm"), "clinic": make_meta("clinic")}
        vectors = {
            "farm": np.tile(FARM_AXIS, (len(chunks["farm"]), 1)),
            "clinic": np.tile(CLINIC_AXIS, (len(chunks["clinic"]), 1)),
        }
        return chunks, metas, vectors

    def test_the_lexical_ranking_leads_with_the_most_distinctive_count(self, book):
        chunks, metas, _ = book
        clinic = terms.extract_terms(chunks, metas).chapters[1]
        assert texts_of(clinic)[:3] == ["physician", "clinic", "patient"]

    def test_the_rerank_promotes_the_term_closest_to_the_chapter(self, book, monkeypatch):
        monkeypatch.setattr(index, "embed_texts", axis_embed)
        chunks, metas, vectors = book
        clinic = terms.extract_terms(chunks, metas, vectors=vectors, tier="fast").chapters[1]
        assert texts_of(clinic)[0] == "clinic"

    def test_the_rerank_reorders_but_never_invents_a_term(self, book, monkeypatch):
        monkeypatch.setattr(index, "embed_texts", axis_embed)
        chunks, metas, vectors = book
        lexical = terms.extract_terms(chunks, metas).chapters[1]
        reranked = terms.extract_terms(chunks, metas, vectors=vectors).chapters[1]
        assert set(texts_of(reranked)) == set(texts_of(lexical))

    def test_counts_and_pages_are_unchanged_by_the_rerank(self, book, monkeypatch):
        monkeypatch.setattr(index, "embed_texts", axis_embed)
        chunks, metas, vectors = book
        lexical = terms.extract_terms(chunks, metas).chapters[1]
        reranked = terms.extract_terms(chunks, metas, vectors=vectors).chapters[1]
        for term in reranked.terms:
            before = term_named(lexical, term.text.lower())
            assert (term.count, term.pdf_page) == (before.count, before.pdf_page)

    def test_records_the_tier_used(self, book, monkeypatch):
        monkeypatch.setattr(index, "embed_texts", axis_embed)
        chunks, metas, vectors = book
        assert terms.extract_terms(chunks, metas, vectors=vectors, tier="balanced").tier == (
            "balanced"
        )


class TestGroupBooks:
    def test_sources_sharing_title_and_author_are_one_book(self):
        groups = terms.group_books([make_meta("a"), make_meta("b")])
        assert [[m.source_id for m in group] for group in groups] == [["a", "b"]]

    def test_matching_ignores_case_and_surrounding_space(self):
        groups = terms.group_books(
            [make_meta("a"), make_meta("b", title="  fresh fruit, broken bodies ")]
        )
        assert len(groups) == 1

    def test_a_different_title_is_a_different_book(self):
        groups = terms.group_books([make_meta("a"), make_meta("b", title="Pathologies of Power")])
        assert len(groups) == 2

    def test_the_same_title_by_another_author_is_a_different_book(self):
        groups = terms.group_books([make_meta("a"), make_meta("b", author="Paul Farmer")])
        assert len(groups) == 2

    def test_books_keep_library_order(self):
        groups = terms.group_books(
            [make_meta("a", title="Second"), make_meta("b", title="First"), make_meta("c")]
        )
        assert [group[0].source_id for group in groups] == ["a", "b", "c"]

    def test_empty_library(self):
        assert terms.group_books([]) == []


def hashed_embed(texts, tier=index.DEFAULT_TIER, *, pages=None, page_count=None, progress=None):
    """Deterministic hashed bag-of-words, so indexing needs no model."""
    dim = 16
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in text.lower().split():
            out[row, sum(map(ord, token)) % dim] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-9)


class TestKeyTerms:
    @pytest.fixture(autouse=True)
    def library(self, tmp_path, monkeypatch, pdf_factory):
        monkeypatch.setenv("DSEARCH_HOME", str(tmp_path / "lib"))
        monkeypatch.setattr(index, "embed_texts", hashed_embed)
        for name, body in (("farm.pdf", FARM), ("clinic.pdf", CLINIC)):
            path = pdf_factory(name=name, pages=[body] * PAGES_PER_CHAPTER)
            index.add(path, author="Seth Holmes", title="Fresh Fruit, Broken Bodies")
        other = pdf_factory(name="other.pdf", pages=["Power shapes illness. " * 4] * 3)
        index.add(other, author="Paul Farmer", title="Pathologies of Power")

    def test_one_entry_per_book(self):
        books = terms.key_terms()
        assert [book.title for book in books] == [
            "Fresh Fruit, Broken Bodies",
            "Pathologies of Power",
        ]

    def test_split_pdfs_of_one_book_are_contrasted_with_each_other(self):
        book = terms.key_terms()[0]
        assert [chapter.label for chapter in book.chapters] == ["farm.pdf", "clinic.pdf"]
        assert "pickers" in texts_of(book.chapters[0])
        assert "workers" not in texts_of(book.chapters[0])

    def test_reranks_at_the_requested_tier_by_default(self):
        assert terms.key_terms()[0].tier == "fast"

    def test_no_rerank_is_lexical_only(self, monkeypatch):
        def explode(*args, **kwargs):
            raise AssertionError("--no-rerank must not embed anything")

        monkeypatch.setattr(index, "embed_texts", explode)
        book = terms.key_terms(rerank=False)[0]
        assert book.tier is None
        assert book.note is None

    def test_a_missing_tier_falls_back_to_lexical_with_a_note(self, monkeypatch):
        def explode(*args, **kwargs):
            raise AssertionError("a missing tier must not start an embedding run")

        monkeypatch.setattr(index, "embed_texts", explode)
        book = terms.key_terms(tier="balanced")[0]
        assert book.tier is None
        assert "dsearch embed --tier balanced" in book.note
        assert texts_of(book.chapters[0])

    def test_source_reports_one_chapter_but_still_contrasts_with_the_book(self):
        books = terms.key_terms(source="clinic.pdf")
        assert len(books) == 1
        assert [chapter.label for chapter in books[0].chapters] == ["clinic.pdf"]
        assert "physician" in texts_of(books[0].chapters[0])
        assert "workers" not in texts_of(books[0].chapters[0])

    def test_unknown_source(self):
        with pytest.raises(index.SourceNotFoundError):
            terms.key_terms(source="nope")

    def test_unknown_tier(self):
        with pytest.raises(ValueError, match="Unknown tier"):
            terms.key_terms(tier="turbo")

    def test_empty_library(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DSEARCH_HOME", str(tmp_path / "empty"))
        assert terms.key_terms() == []
