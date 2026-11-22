"""`dsearch stats` aggregates files on disk; these check the arithmetic."""

from __future__ import annotations

import pytest

from dsearch import index
from dsearch.stats import LibraryStats, library_stats


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("DSEARCH_HOME", str(tmp_path / "lib"))


def make_source(source_id: str, *, pages: int, chunks: int, tiers: dict[str, int]):
    """Register a source and write dummy vector files of the given byte sizes."""
    meta = index.SourceMeta(
        source_id=source_id,
        filename=f"{source_id}.pdf",
        title=source_id,
        author="A",
        added="2026-01-01",
        page_count=pages,
        chunk_count=chunks,
    )
    index.save_meta(meta)
    for tier, size in tiers.items():
        index.vectors_path(source_id, tier).write_bytes(b"\0" * size)
    return meta


class TestLibraryStats:
    def test_missing_library_is_all_zeros_and_not_created(self, tmp_path):
        stats = library_stats()
        assert stats.sources == 0
        assert stats.chunks_per_page == 0.0
        assert stats.tier_bytes == dict.fromkeys(index.TIERS, 0)
        assert not (tmp_path / "lib").exists()

    def test_totals_sum_across_sources(self):
        make_source("a" * 64, pages=10, chunks=40, tiers={"fast": 100})
        make_source("b" * 64, pages=5, chunks=10, tiers={"fast": 50, "balanced": 200})
        stats = library_stats()
        assert (stats.sources, stats.pages, stats.chunks) == (2, 15, 50)
        assert stats.tier_bytes == {"fast": 150, "balanced": 200, "best": 0}
        assert stats.tier_sources == {"fast": 2, "balanced": 1, "best": 0}

    def test_total_bytes_covers_the_whole_source_folder(self):
        make_source("a" * 64, pages=1, chunks=1, tiers={"fast": 100})
        assert library_stats().total_bytes >= 100

    def test_chunks_per_page(self):
        assert LibraryStats(pages=4, chunks=10).chunks_per_page == 2.5

    def test_to_dict_includes_the_derived_ratio(self):
        payload = LibraryStats(pages=3, chunks=10).to_dict()
        assert payload["chunks_per_page"] == 3.33
        assert payload["sources"] == 0
