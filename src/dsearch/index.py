"""The on-disk library: adding, listing, and removing indexed sources.

Layout under `~/.dsearch/` (override with `DSEARCH_HOME`):

    library.json                     list of every indexed source
    sources/<sha256>/meta.json       author, title, filename, added, pages
    sources/<sha256>/chunks.jsonl    one Chunk per line, in chunk_id order
    sources/<sha256>/vectors_<tier>.npy   float32 (n_chunks, dim), row i = chunk i

Keying folders by the SHA-256 of the file bytes makes re-adding the same PDF a
no-op no matter what it has been renamed to, and makes a genuinely different
file a separate source even under the same name.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from dsearch.chunk import DEFAULT_OVERLAP, DEFAULT_SIZE, Chunk, chunk
from dsearch.extract import extract

# Embedding quality tiers, from the spec.
TIERS: dict[str, str] = {
    "fast": "all-MiniLM-L6-v2",
    "balanced": "all-mpnet-base-v2",
    "best": "BAAI/bge-base-en-v1.5",
}
DEFAULT_TIER = "fast"

# Past this many sources, searches slow down noticeably. A warning, never a block.
SOURCE_WARN_LIMIT = 10

# Texts per embedding batch. Large enough to keep the model busy, small enough
# that progress moves visibly on a laptop CPU.
EMBED_BATCH = 32

# Bytes per read when hashing, so a 500MB PDF does not land in memory at once.
HASH_BLOCK = 1 << 20

# Progress callbacks receive (pages_done, pages_total).
ProgressFn = Callable[[int, int], None]

_MODEL_CACHE: dict[str, object] = {}


class SourceNotFoundError(Exception):
    """Raised when an identifier matches no source in the library."""


class AmbiguousSourceError(Exception):
    """Raised when an identifier matches more than one source."""


@dataclass
class SourceMeta:
    """What the library knows about one indexed PDF."""

    source_id: str
    filename: str
    title: str
    author: str
    added: str
    page_count: int
    chunk_count: int = 0
    chunk_size: int = DEFAULT_SIZE
    overlap: int = DEFAULT_OVERLAP
    tiers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> SourceMeta:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def short_id(self) -> str:
        """The first 8 hex characters, which is what the CLI shows and accepts."""
        return self.source_id[:8]


@dataclass
class AddResult:
    """Outcome of `add`, so callers can report it without re-deriving anything."""

    meta: SourceMeta
    already_indexed: bool
    embedded_tier: str | None
    warning: str | None = None


def home() -> Path:
    """The library root, honouring `DSEARCH_HOME`."""
    return Path(os.environ.get("DSEARCH_HOME") or Path.home() / ".dsearch")


def sources_dir() -> Path:
    return home() / "sources"


def source_dir(source_id: str) -> Path:
    return sources_dir() / source_id


def library_path() -> Path:
    return home() / "library.json"


def vectors_path(source_id: str, tier: str) -> Path:
    return source_dir(source_id) / f"vectors_{tier}.npy"


def _validate_tier(tier: str) -> str:
    if tier not in TIERS:
        raise ValueError(f"Unknown tier {tier!r}. Choose one of: {', '.join(TIERS)}")
    return tier


def file_hash(path: str | Path) -> str:
    """SHA-256 of the file's bytes — the source's identity."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(HASH_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    """Write JSON atomically, so an interrupted add cannot corrupt the library."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def load_library() -> list[SourceMeta]:
    """Every source in the library, in the order it was added."""
    path = library_path()
    if not path.is_file():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return [SourceMeta.from_dict(row) for row in rows]


def save_library(sources: Iterable[SourceMeta]) -> None:
    _write_json(library_path(), [s.to_dict() for s in sources])


def list_sources() -> list[SourceMeta]:
    """Public alias for `load_library`, matching the `dsearch list` command."""
    return load_library()


def resolve_source(identifier: str) -> SourceMeta:
    """Find one source by id prefix, filename, or title.

    Raises SourceNotFoundError or AmbiguousSourceError rather than guessing,
    because the caller may be about to delete what it resolves.
    """
    sources = load_library()
    if not sources:
        raise SourceNotFoundError("The library is empty. Add a PDF with `dsearch add`.")

    needle = identifier.strip().lower()
    exact = [s for s in sources if s.source_id.lower() == needle]
    if exact:
        return exact[0]

    matches = [
        s
        for s in sources
        if s.source_id.lower().startswith(needle)
        or s.filename.lower() == needle
        or Path(s.filename).stem.lower() == needle
        or s.title.lower() == needle
    ]
    if not matches:
        raise SourceNotFoundError(f"No source matches {identifier!r}.")
    if len(matches) > 1:
        names = ", ".join(f"{s.short_id} ({s.title})" for s in matches)
        raise AmbiguousSourceError(f"{identifier!r} matches several sources: {names}")
    return matches[0]


def load_model(tier: str):
    """Load (and cache for this process) the SentenceTransformer for a tier.

    Imported lazily: `sentence_transformers` pulls in torch, which costs seconds
    of startup that `dsearch --help` and `dsearch list` should not pay.
    """
    _validate_tier(tier)
    if tier not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[tier] = SentenceTransformer(TIERS[tier])
    return _MODEL_CACHE[tier]


def embed_texts(
    texts: list[str],
    tier: str = DEFAULT_TIER,
    *,
    pages: list[int] | None = None,
    page_count: int | None = None,
    progress: ProgressFn | None = None,
) -> np.ndarray:
    """Embed `texts`, reporting progress in pages of the source.

    Vectors are L2-normalised so that cosine similarity is a plain dot product.
    `pages[i]` is the pdf page that `texts[i]` came from; it is used only to
    translate batch progress into the page counts a reader recognises.
    """
    model = load_model(tier)
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)

    total_pages = page_count or (max(pages) if pages else 1)
    out: list[np.ndarray] = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start : start + EMBED_BATCH]
        out.append(
            model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )
        if progress:
            done = pages[min(start + len(batch), len(pages)) - 1] if pages else total_pages
            progress(min(done, total_pages), total_pages)

    return np.vstack(out).astype(np.float32)


def load_chunks(source_id: str) -> list[Chunk]:
    """Read a source's chunks back, in `chunk_id` order."""
    path = source_dir(source_id) / "chunks.jsonl"
    if not path.is_file():
        raise SourceNotFoundError(f"No chunks stored for source {source_id[:8]}.")
    with open(path, encoding="utf-8") as handle:
        return [Chunk.from_dict(json.loads(line)) for line in handle if line.strip()]


def save_chunks(source_id: str, chunks: list[Chunk]) -> None:
    path = source_dir(source_id) / "chunks.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".jsonl.tmp")
    with open(temp, "w", encoding="utf-8") as handle:
        for item in chunks:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    temp.replace(path)


def load_vectors(source_id: str, tier: str = DEFAULT_TIER) -> np.ndarray:
    """Read a source's embedding matrix; row i is chunk i."""
    path = vectors_path(source_id, _validate_tier(tier))
    if not path.is_file():
        raise SourceNotFoundError(
            f"Source {source_id[:8]} has no {tier!r} vectors. "
            f"Run `dsearch add` on it again with --tier {tier}."
        )
    return np.load(path)


def _meta_from_pdf(
    path: Path,
    source_id: str,
    author: str | None,
    title: str | None,
    page_count: int,
) -> SourceMeta:
    """Build SourceMeta, preferring explicit flags over PDF metadata over filename."""
    import pymupdf

    pdf_author = pdf_title = ""
    try:
        with pymupdf.open(path) as doc:
            pdf_author = (doc.metadata or {}).get("author", "") or ""
            pdf_title = (doc.metadata or {}).get("title", "") or ""
    except Exception:
        pass

    return SourceMeta(
        source_id=source_id,
        filename=path.name,
        title=(title or pdf_title.strip() or path.stem),
        author=(author or pdf_author.strip() or ""),
        added=date.today().isoformat(),
        page_count=page_count,
    )


def add(
    pdf_path: str | Path,
    *,
    author: str | None = None,
    title: str | None = None,
    tier: str = DEFAULT_TIER,
    chunk_size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    progress: ProgressFn | None = None,
) -> AddResult:
    """Index a PDF into the library.

    Re-adding a file whose bytes are already indexed at this tier does no work
    and reports that. Re-adding it at a *new* tier keeps the stored chunks and
    embeds only the missing vectors, so switching quality never re-extracts.

    Raises NoTextLayerError for scans and FileNotFoundError for a bad path.
    """
    _validate_tier(tier)
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"No such PDF: {path}")

    source_id = file_hash(path)
    library = load_library()
    existing = next((s for s in library if s.source_id == source_id), None)

    if existing and tier in existing.tiers and vectors_path(source_id, tier).is_file():
        return AddResult(meta=existing, already_indexed=True, embedded_tier=None)

    if existing and (source_dir(source_id) / "chunks.jsonl").is_file():
        # Known file, new tier: reuse the chunks and embed only what is missing.
        meta = existing
        chunks = load_chunks(source_id)
    else:
        pages = extract(path)
        chunks = chunk(pages, size=chunk_size, overlap=overlap, source_id=source_id)
        meta = _meta_from_pdf(path, source_id, author, title, page_count=len(pages))
        meta.chunk_count = len(chunks)
        meta.chunk_size = chunk_size
        meta.overlap = overlap
        save_chunks(source_id, chunks)

    vectors = embed_texts(
        [c.text for c in chunks],
        tier,
        pages=[c.pdf_page for c in chunks],
        page_count=meta.page_count,
        progress=progress,
    )
    vectors_path(source_id, tier).parent.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path(source_id, tier), vectors)

    if tier not in meta.tiers:
        meta.tiers.append(tier)
    _write_json(source_dir(source_id) / "meta.json", meta.to_dict())

    library = [s for s in library if s.source_id != source_id] + [meta]
    save_library(library)

    warning = None
    if len(library) > SOURCE_WARN_LIMIT:
        warning = (
            f"The library now holds {len(library)} sources (past the "
            f"{SOURCE_WARN_LIMIT} that searches stay fast at). "
            "Consider `dsearch remove` for texts you are done with."
        )
    return AddResult(
        meta=meta,
        already_indexed=False,
        embedded_tier=tier,
        warning=warning,
    )


def remove(identifier: str) -> SourceMeta:
    """Delete a source's folder and drop it from the library."""
    meta = resolve_source(identifier)
    shutil.rmtree(source_dir(meta.source_id), ignore_errors=True)
    save_library([s for s in load_library() if s.source_id != meta.source_id])
    return meta
