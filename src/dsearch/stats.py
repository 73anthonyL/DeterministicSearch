"""Library statistics: what is indexed and how much space it takes.

Pure reads of files already on disk. Nothing here loads a model or touches
the network, so `dsearch stats` is instant even on a large library.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from dsearch.doctor import directory_size
from dsearch.index import TIERS, home, list_sources, source_dir, vectors_path


@dataclass
class LibraryStats:
    """Totals across every source, plus the per-tier vector footprint."""

    sources: int = 0
    pages: int = 0
    chunks: int = 0
    total_bytes: int = 0
    # Bytes of vector files per tier, so users see what `balanced` costs them.
    tier_bytes: dict[str, int] = field(default_factory=dict)
    # Sources that have vectors for each tier.
    tier_sources: dict[str, int] = field(default_factory=dict)

    @property
    def chunks_per_page(self) -> float:
        return self.chunks / self.pages if self.pages else 0.0

    def to_dict(self) -> dict:
        return {**asdict(self), "chunks_per_page": round(self.chunks_per_page, 2)}


def _file_size(source_id: str, tier: str) -> int | None:
    path = vectors_path(source_id, tier)
    return path.stat().st_size if path.is_file() else None


def library_stats() -> LibraryStats:
    """Aggregate the library on disk; an empty or missing library gives zeros."""
    stats = LibraryStats(
        tier_bytes=dict.fromkeys(TIERS, 0),
        tier_sources=dict.fromkeys(TIERS, 0),
    )
    if not home().is_dir():
        return stats

    for meta in list_sources():
        stats.sources += 1
        stats.pages += meta.page_count
        stats.chunks += meta.chunk_count
        stats.total_bytes += directory_size(source_dir(meta.source_id))
        for tier in TIERS:
            size = _file_size(meta.source_id, tier)
            if size is not None:
                stats.tier_bytes[tier] += size
                stats.tier_sources[tier] += 1
    return stats
