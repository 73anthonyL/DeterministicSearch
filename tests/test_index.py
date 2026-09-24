"""Tests for the on-disk library: hashing, add/list/remove, and cache-by-hash.

Embedding is stubbed out so these run without downloading a model; the real
model path is exercised in the end-to-end verification.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from dsearch import index
from dsearch.extract import NoTextLayerError

BODY = (
    "The workers moved down the rows before dawn, backs bent to the plants. "
    "They filled box after box in the gray light of the field. "
    "A foreman watched from the edge of the row and said nothing at all."
)

EMBED_DIM = 8


@pytest.fixture(autouse=True)
def library_home(tmp_path, monkeypatch):
    """Point the library at a temp dir so tests never touch a real ~/.dsearch."""
    monkeypatch.setenv("DSEARCH_HOME", str(tmp_path / "lib"))
    return tmp_path / "lib"


@pytest.fixture(autouse=True)
def stub_embeddings(monkeypatch):
    """Replace the model with a cheap deterministic hash embedding."""
    calls = {"count": 0, "texts": 0}

    def fake_embed(texts, tier=index.DEFAULT_TIER, *, pages=None, page_count=None, progress=None):
        calls["count"] += 1
        calls["texts"] += len(texts)
        if progress and page_count:
            progress(page_count, page_count)
        out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for position, token in enumerate(text.lower().split()):
                out[row, hash(token) % EMBED_DIM] += 1.0 / (position + 1)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)

    monkeypatch.setattr(index, "embed_texts", fake_embed)
    return calls


@pytest.fixture
def book(pdf_factory):
    return pdf_factory(
        name="fresh_fruit.pdf",
        pages=[f"{BODY} Page {n}." for n in range(1, 6)],
        folios=[str(n) for n in range(1, 6)],
    )


class TestPaths:
    def test_home_honours_env_override(self, library_home):
        assert index.home() == library_home

    def test_source_layout(self):
        assert index.source_dir("abc").name == "abc"
        assert index.vectors_path("abc", "fast").name == "vectors_fast.npy"


class TestFileHash:
    def test_is_stable_for_identical_bytes(self, tmp_path):
        a, b = tmp_path / "a.bin", tmp_path / "b.bin"
        a.write_bytes(b"identical")
        b.write_bytes(b"identical")
        assert index.file_hash(a) == index.file_hash(b)

    def test_differs_for_different_bytes(self, tmp_path):
        a, b = tmp_path / "a.bin", tmp_path / "b.bin"
        a.write_bytes(b"one")
        b.write_bytes(b"two")
        assert index.file_hash(a) != index.file_hash(b)

    def test_is_a_full_sha256(self, book):
        assert len(index.file_hash(book)) == 64


class TestAdd:
    def test_writes_the_expected_files(self, book):
        result = index.add(book, author="Seth Holmes", title="Fresh Fruit")
        folder = index.source_dir(result.meta.source_id)
        assert (folder / "meta.json").is_file()
        assert (folder / "chunks.jsonl").is_file()
        assert (folder / "vectors_fast.npy").is_file()
        assert index.library_path().is_file()

    def test_records_metadata_from_flags(self, book):
        meta = index.add(book, author="Seth Holmes", title="Fresh Fruit").meta
        assert meta.author == "Seth Holmes"
        assert meta.title == "Fresh Fruit"
        assert meta.filename == "fresh_fruit.pdf"
        assert meta.page_count == 5
        assert meta.chunk_count > 0
        assert meta.tiers == ["fast"]
        assert meta.added

    def test_falls_back_to_pdf_metadata(self, pdf_factory):
        path = pdf_factory(
            pages=[BODY] * 3,
            metadata={"author": "Embedded Author", "title": "Embedded Title"},
        )
        meta = index.add(path).meta
        assert meta.author == "Embedded Author"
        assert meta.title == "Embedded Title"

    def test_falls_back_to_filename(self, pdf_factory):
        path = pdf_factory(name="Chapter3.pdf", pages=[BODY] * 3)
        meta = index.add(path).meta
        assert meta.title == "Chapter3"
        assert meta.author == ""

    def test_flags_beat_pdf_metadata(self, pdf_factory):
        path = pdf_factory(pages=[BODY] * 3, metadata={"author": "Embedded", "title": "Embedded"})
        meta = index.add(path, author="Seth Holmes", title="Fresh Fruit").meta
        assert (meta.author, meta.title) == ("Seth Holmes", "Fresh Fruit")

    def test_vectors_align_with_chunks(self, book):
        result = index.add(book)
        chunks = index.load_chunks(result.meta.source_id)
        vectors = index.load_vectors(result.meta.source_id, "fast")
        assert vectors.shape[0] == len(chunks)
        assert [c.chunk_id for c in chunks] == list(range(len(chunks)))

    def test_chunks_carry_the_source_id(self, book):
        result = index.add(book)
        chunks = index.load_chunks(result.meta.source_id)
        assert all(c.source_id == result.meta.source_id for c in chunks)

    def test_honours_chunk_size_and_overlap(self, book):
        result = index.add(book, chunk_size=2, overlap=0)
        assert result.meta.chunk_size == 2
        assert result.meta.overlap == 0
        assert all(len(c.sentences) <= 2 for c in index.load_chunks(result.meta.source_id))

    def test_reports_progress_in_pages(self, book):
        seen: list[tuple[int, int]] = []
        index.add(book, progress=lambda done, total: seen.append((done, total)))
        assert seen
        assert seen[-1] == (5, 5)

    def test_rejects_unknown_tier(self, book):
        with pytest.raises(ValueError, match="Unknown tier"):
            index.add(book, tier="turbo")

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            index.add(tmp_path / "nope.pdf")

    def test_scanned_pdf_is_rejected(self, pdf_factory):
        path = pdf_factory(pages=[""] * 5)
        with pytest.raises(NoTextLayerError):
            index.add(path)


class TestAddIsCachedByHash:
    def test_same_file_is_not_re_embedded(self, book, stub_embeddings):
        first = index.add(book)
        after_first = stub_embeddings["texts"]
        second = index.add(book)
        assert second.already_indexed is True
        assert second.embedded_tier is None
        assert second.meta.source_id == first.meta.source_id
        assert stub_embeddings["texts"] == after_first  # No further embedding.

    def test_renamed_but_identical_file_is_the_same_source(self, book, tmp_path):
        first = index.add(book)
        copy = tmp_path / "renamed.pdf"
        copy.write_bytes(book.read_bytes())
        second = index.add(copy)
        assert second.already_indexed is True
        assert second.meta.source_id == first.meta.source_id
        assert len(index.list_sources()) == 1

    def test_different_file_is_a_new_source(self, pdf_factory):
        one = pdf_factory(name="one.pdf", pages=[f"{BODY} One."] * 3)
        two = pdf_factory(name="two.pdf", pages=[f"{BODY} Two is different."] * 3)
        assert index.add(one).meta.source_id != index.add(two).meta.source_id
        assert len(index.list_sources()) == 2

    def test_new_tier_reuses_chunks_and_embeds_only_vectors(self, book, stub_embeddings):
        index.add(book, tier="fast")
        batches_after_fast = stub_embeddings["count"]
        result = index.add(book, tier="balanced")

        assert result.already_indexed is False
        assert result.embedded_tier == "balanced"
        assert stub_embeddings["count"] > batches_after_fast  # Embedded again...
        assert sorted(result.meta.tiers) == ["balanced", "fast"]
        assert index.vectors_path(result.meta.source_id, "fast").is_file()
        assert index.vectors_path(result.meta.source_id, "balanced").is_file()

    def test_second_add_of_a_known_tier_is_still_a_no_op(self, book):
        index.add(book, tier="fast")
        index.add(book, tier="balanced")
        assert index.add(book, tier="fast").already_indexed is True


class TestLibrary:
    def test_empty_library(self):
        assert index.list_sources() == []

    def test_lists_sources_in_insertion_order(self, pdf_factory):
        for n in (1, 2, 3):
            index.add(pdf_factory(name=f"book{n}.pdf", pages=[f"{BODY} Book {n}."] * 3))
        assert [s.filename for s in index.list_sources()] == [
            "book1.pdf",
            "book2.pdf",
            "book3.pdf",
        ]

    def test_warns_past_the_limit_without_blocking(self, pdf_factory):
        results = [
            index.add(pdf_factory(name=f"b{n}.pdf", pages=[f"{BODY} Book number {n}."] * 2))
            for n in range(index.SOURCE_WARN_LIMIT + 1)
        ]
        assert results[-2].warning is None
        assert "past the" in (results[-1].warning or "")
        assert len(index.list_sources()) == index.SOURCE_WARN_LIMIT + 1

    def test_survives_a_corrupt_library_file(self, book):
        index.add(book)
        index.library_path().write_text("{ not json", encoding="utf-8")
        assert index.list_sources() == []

    def test_meta_json_round_trips(self, book):
        meta = index.add(book, author="Seth Holmes", title="Fresh Fruit").meta
        stored = json.loads((index.source_dir(meta.source_id) / "meta.json").read_text())
        assert index.SourceMeta.from_dict(stored) == meta


class TestResolveSource:
    def test_by_full_id(self, book):
        meta = index.add(book).meta
        assert index.resolve_source(meta.source_id).source_id == meta.source_id

    def test_by_short_id(self, book):
        meta = index.add(book).meta
        assert index.resolve_source(meta.short_id).source_id == meta.source_id

    def test_by_filename_and_stem(self, book):
        meta = index.add(book).meta
        assert index.resolve_source("fresh_fruit.pdf").source_id == meta.source_id
        assert index.resolve_source("fresh_fruit").source_id == meta.source_id

    def test_by_title_case_insensitively(self, book):
        meta = index.add(book, title="Fresh Fruit").meta
        assert index.resolve_source("fresh fruit").source_id == meta.source_id

    def test_unknown_identifier(self, book):
        index.add(book)
        with pytest.raises(index.SourceNotFoundError):
            index.resolve_source("no-such-book")

    def test_empty_library(self):
        with pytest.raises(index.SourceNotFoundError, match="library is empty"):
            index.resolve_source("anything")

    def test_ambiguous_identifier(self, pdf_factory):
        index.add(pdf_factory(name="one.pdf", pages=[f"{BODY} One."] * 2), title="Shared Title")
        index.add(pdf_factory(name="two.pdf", pages=[f"{BODY} Two."] * 2), title="Shared Title")
        with pytest.raises(index.AmbiguousSourceError, match="matches several"):
            index.resolve_source("Shared Title")


class TestRemove:
    def test_deletes_folder_and_library_row(self, book):
        meta = index.add(book).meta
        removed = index.remove(meta.short_id)
        assert removed.source_id == meta.source_id
        assert not index.source_dir(meta.source_id).exists()
        assert index.list_sources() == []

    def test_leaves_other_sources_alone(self, pdf_factory):
        keep = index.add(pdf_factory(name="keep.pdf", pages=[f"{BODY} Keep."] * 2)).meta
        index.add(pdf_factory(name="drop.pdf", pages=[f"{BODY} Drop."] * 2))
        index.remove("drop")
        assert [s.source_id for s in index.list_sources()] == [keep.source_id]
        assert index.source_dir(keep.source_id).exists()

    def test_unknown_source(self, book):
        index.add(book)
        with pytest.raises(index.SourceNotFoundError):
            index.remove("nope")

    def test_removed_source_can_be_added_again(self, book):
        meta = index.add(book).meta
        index.remove(meta.short_id)
        assert index.add(book).already_indexed is False


class TestLoaders:
    def test_load_chunks_for_unknown_source(self):
        with pytest.raises(index.SourceNotFoundError):
            index.load_chunks("0" * 64)

    def test_load_vectors_for_missing_tier(self, book):
        meta = index.add(book, tier="fast").meta
        with pytest.raises(index.SourceNotFoundError, match="no 'best' vectors"):
            index.load_vectors(meta.source_id, "best")

    def test_load_vectors_rejects_unknown_tier(self, book):
        meta = index.add(book).meta
        with pytest.raises(ValueError, match="Unknown tier"):
            index.load_vectors(meta.source_id, "turbo")
