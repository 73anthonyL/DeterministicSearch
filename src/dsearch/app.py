"""Streamlit front end for dsearch.

A thin wrapper over the same functions the CLI calls — no retrieval logic lives
here. Run it with:

    streamlit run src/dsearch/app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from dsearch import __version__
from dsearch.chunk import DEFAULT_OVERLAP, DEFAULT_SIZE
from dsearch.cite import citation_block, mla
from dsearch.extract import NoTextLayerError
from dsearch.index import (
    DEFAULT_TIER,
    INDEX_VERSION,
    TIERS,
    AmbiguousSourceError,
    SourceNotFoundError,
    add,
    list_sources,
    remove,
    resolve_source,
    upgrade_stale,
)
from dsearch.search import DEFAULT_K, Result, search

# Neighbouring-chunk context shown around a result, in characters.
NEIGHBOUR_CHARS = 240

# Bounds for the "search deeper" slider.
MIN_K, MAX_K = 1, 20

TIER_NOTES = {
    "fast": "all-MiniLM-L6-v2 — quickest, smallest download",
    "balanced": "all-mpnet-base-v2 — better recall, slower",
    "best": "BAAI/bge-base-en-v1.5 — highest quality, slowest",
}

DIM = "#6b7280"


def _truncate(text: str, limit: int = NEIGHBOUR_CHARS) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _dim(text: str) -> str:
    return f"<span style='color:{DIM}'>{text}</span>"


def _passage_html(result: Result) -> str:
    """The passage with context dimmed and the matching sentence bold."""
    parts = []
    if result.before:
        parts.append(_dim(_truncate(result.before.text)))

    body = []
    for position, sentence in enumerate(result.chunk.sentences):
        if position == result.best_sentence:
            body.append(f"<strong>{sentence}</strong>")
        else:
            body.append(_dim(sentence))
    parts.append(" ".join(body))

    if result.after:
        parts.append(_dim(_truncate(result.after.text)))

    return "<br><br>".join(parts)


def _render_result(result: Result, position: int) -> None:
    """One result card, mirroring the CLI panel."""
    with st.container(border=True):
        header = [f"**{result.meta.title}**"]
        if result.chunk.chapter:
            header.append(f"*{result.chunk.chapter}*")
        header.append(result.page_label)
        st.markdown(f"{position}. " + "  ·  ".join(header))

        st.markdown(_passage_html(result), unsafe_allow_html=True)

        st.markdown(f":green[{mla(result)}]")
        with st.expander("Copy citation"):
            # st.code gives a one-click copy button, which is the whole point.
            st.code(citation_block(result), language=None, wrap_lines=True)


def _index_upload(upload, author: str, title: str, tier: str, size: int, overlap: int) -> None:
    """Write the upload to a temp file and index it, showing page progress.

    Indexing is keyed by the SHA-256 of the bytes, so re-uploading the same file
    costs nothing — the hash is checked before any work happens.
    """
    # Write into a temp *directory* under the upload's own name, so the library
    # records "Chapter2.pdf" rather than an opaque "tmpjzua04hj.pdf" that no
    # user could later recognise or pass to `dsearch remove`.
    temp_dir = Path(tempfile.mkdtemp())
    temp_path = temp_dir / Path(upload.name).name
    temp_path.write_bytes(upload.getbuffer())

    bar = st.progress(0.0, text="Reading pages…")

    def tick(done: int, total: int) -> None:
        bar.progress(min(done / max(total, 1), 1.0), text=f"Embedding page {done} of {total}")

    try:
        result = add(
            temp_path,
            author=author.strip() or None,
            title=title.strip() or Path(upload.name).stem,
            tier=tier,
            chunk_size=size,
            overlap=overlap,
            progress=tick,
            notify=st.info,
        )
    except NoTextLayerError as exc:
        bar.empty()
        st.error(str(exc))
        return
    except (ValueError, FileNotFoundError) as exc:
        bar.empty()
        st.error(str(exc))
        return
    finally:
        temp_path.unlink(missing_ok=True)
        temp_dir.rmdir()

    bar.empty()
    meta = result.meta
    if result.already_indexed:
        st.info(f"**{meta.title}** is already indexed at the *{tier}* tier — nothing to do.")
    else:
        st.success(
            f"Indexed **{meta.title}** — {meta.page_count} pages, "
            f"{meta.chunk_count} chunks, *{tier}* tier."
        )
    if result.warning:
        st.warning(result.warning)


def _upgrade_stale_sources(source: str | None) -> None:
    """Rebuild any in-scope source indexed by an older pipeline, with a page bar."""
    scoped = [resolve_source(source)] if source else list_sources()
    if not any(meta.index_version < INDEX_VERSION for meta in scoped):
        return
    bar = st.progress(0.0, text="Re-indexing…")

    def tick(done: int, total: int) -> None:
        bar.progress(min(done / max(total, 1), 1.0), text=f"Re-indexing page {done} of {total}")

    upgrade_stale(scoped, progress=tick, notify=st.info)
    bar.empty()


def _sidebar() -> tuple[str, str | None]:
    """Library management. Returns the chosen tier and source filter.

    Every widget carries an explicit `key`. Without one, Streamlit identifies a
    widget by its position in the tree, and this sidebar changes shape as soon
    as the library stops being empty (the "Search in" selector appears). That
    shift silently dropped the author typed before the first index, so the
    citation came out with no author at all.
    """
    with st.sidebar:
        st.header("Library")
        tier = st.selectbox(
            "Quality tier",
            list(TIERS),
            index=list(TIERS).index(DEFAULT_TIER),
            help="Which embedding model to index and search with.",
            key="tier",
        )
        st.caption(TIER_NOTES[tier])

        sources = list_sources()
        if sources:
            names = ["All sources"] + [s.title for s in sources]
            chosen = st.selectbox("Search in", names, key="source_filter")
            source = None if chosen == "All sources" else chosen
        else:
            source = None
            st.caption("No sources yet. Add a PDF on the right.")

        st.divider()
        st.subheader("Add a PDF")
        upload = st.file_uploader("Drag and drop a PDF", type=["pdf"], key="upload")
        author = st.text_input("Author", placeholder="Seth Holmes", key="author")
        title = st.text_input("Title", placeholder="Fresh Fruit, Broken Bodies", key="title")
        with st.expander("Chunking"):
            size = st.number_input("Sentences per chunk", 1, 10, DEFAULT_SIZE, key="chunk_size")
            overlap = st.number_input("Overlap", 0, 9, DEFAULT_OVERLAP, key="overlap")
        if upload is not None and st.button("Index this PDF", type="primary", key="index"):
            if overlap >= size:
                st.error("Overlap must be smaller than the chunk size.")
            else:
                _index_upload(upload, author, title, tier, int(size), int(overlap))
                st.rerun()

        if sources:
            st.divider()
            st.subheader("Indexed sources")
            for meta in sources:
                columns = st.columns([5, 1])
                columns[0].markdown(
                    f"**{meta.title}**  \n"
                    f"<span style='color:{DIM}'>{meta.author or 'unknown author'} · "
                    f"{meta.page_count} pages · {', '.join(meta.tiers)}</span>",
                    unsafe_allow_html=True,
                )
                if columns[1].button("✕", key=f"rm-{meta.source_id}", help="Remove"):
                    remove(meta.source_id)
                    st.rerun()

        st.divider()
        st.caption(f"dsearch {__version__} — retrieval only, no text is generated.")
    return tier, source


def main() -> None:
    st.set_page_config(page_title="dsearch", page_icon="🔎", layout="wide")
    st.title("🔎 dsearch")
    st.caption(
        "Semantic evidence search for your PDFs. Every result is a verbatim passage "
        "from the source, with the page it came from."
    )

    tier, source = _sidebar()

    query = st.text_area(
        "What are you looking for?",
        height=120,
        key="query",
        placeholder=(
            "Ask a question, name a concept, or paste a whole paragraph of your own thinking — "
            "long queries are scored sentence by sentence."
        ),
    )
    k = st.slider("Passages to return", MIN_K, MAX_K, DEFAULT_K, key="k")

    if not st.button("Search", type="primary", key="search") or not query.strip():
        if not list_sources():
            st.info("Add a PDF in the sidebar to get started.")
        return

    try:
        _upgrade_stale_sources(source)
        with st.spinner("Searching…"):
            results = search(query, k=k, tier=tier, source=source)
    except (SourceNotFoundError, AmbiguousSourceError, ValueError) as exc:
        st.error(str(exc))
        return

    if not results:
        st.warning(
            "No passages found. Check that your sources are indexed at the "
            f"*{tier}* tier in the sidebar."
        )
        return

    st.markdown(f"**{len(results)}** passage(s)")
    for position, result in enumerate(results, start=1):
        _render_result(result, position)


# Streamlit executes the script with __name__ == "__main__"; the guard keeps the
# module importable so its formatting helpers can be unit-tested.
if __name__ == "__main__":
    main()
