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
    SourceMeta,
    SourceNotFoundError,
    TierGap,
    add,
    available_tiers,
    edit,
    embed_source,
    format_duration,
    list_sources,
    remove,
    resolve_source,
    tier_gap,
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


def _index_uploads(uploads, author: str, title: str, tier: str, size: int, overlap: int) -> None:
    """Index each upload in turn, with one bar showing book and page progress.

    Indexing is keyed by the SHA-256 of the bytes, so re-uploading the same file
    costs nothing — the hash is checked before any work happens. Author and
    title typed in the sidebar apply only when a single file is uploaded; with
    several, each PDF's own metadata is used and the table below corrects it.
    """
    total_books = len(uploads)
    bar = st.progress(0.0, text="Reading pages…")

    for position, upload in enumerate(uploads, start=1):
        book = f"Book {position} of {total_books}" if total_books > 1 else upload.name

        def tick(done: int, total: int, book: str = book, position: int = position) -> None:
            # Overall progress: books finished, plus this book's fraction.
            fraction = (position - 1 + done / max(total, 1)) / total_books
            bar.progress(min(fraction, 1.0), text=f"{book} — embedding page {done} of {total}")

        result = _index_one(upload, author, title, tier, size, overlap, tick)
        if result is None:
            continue
        meta = result.meta
        if result.already_indexed:
            st.info(f"**{meta.title}** is already indexed at the *{tier}* tier — skipped.")
        else:
            verb = "Rebuilt" if result.reindexed else "Indexed"
            st.success(
                f"{verb} **{meta.title}** — {meta.page_count} pages, "
                f"{meta.chunk_count} chunks, *{tier}* tier."
            )
        if result.warning:
            st.warning(result.warning)
    bar.empty()


def _index_one(upload, author: str, title: str, tier: str, size: int, overlap: int, tick):
    """Write one upload to a temp file and index it; None if it failed."""
    # Write into a temp *directory* under the upload's own name, so the library
    # records "Chapter2.pdf" rather than an opaque "tmpjzua04hj.pdf" that no
    # user could later recognise or pass to `dsearch remove`.
    temp_dir = Path(tempfile.mkdtemp())
    temp_path = temp_dir / Path(upload.name).name
    temp_path.write_bytes(upload.getbuffer())
    try:
        return add(
            temp_path,
            author=author.strip() or None,
            title=title.strip() or None,
            tier=tier,
            chunk_size=size,
            overlap=overlap,
            progress=tick,
            notify=st.info,
        )
    except (NoTextLayerError, ValueError, FileNotFoundError) as exc:
        st.error(f"{upload.name}: {exc}")
        return None
    finally:
        temp_path.unlink(missing_ok=True)
        temp_dir.rmdir()


# Columns of the library table. Only author and title are editable; the rest
# describe the index and are shown for orientation.
LIBRARY_COLUMNS = ("Title", "Author", "Pages", "Tiers", "ID")


def _library_rows(sources: list[SourceMeta]) -> list[dict[str, object]]:
    return [
        {
            "Title": meta.title,
            "Author": meta.author,
            "Pages": meta.page_count,
            "Tiers": ", ".join(meta.tiers),
            "ID": meta.short_id,
        }
        for meta in sources
    ]


def _apply_table_edits(edited_rows: dict, source_ids: list[str]) -> list[SourceMeta]:
    """Push the table's edited cells through `edit`, the same path as the CLI.

    `edited_rows` is Streamlit's `{row_index: {column: value}}`; only the
    author and title columns can change, so anything else is ignored.
    """
    updated = []
    for row, changes in edited_rows.items():
        author = changes.get("Author")
        title = changes.get("Title")
        if author is None and title is None:
            continue
        updated.append(edit(source_ids[int(row)], author=author, title=title))
    return updated


def _library_table(sources: list[SourceMeta]) -> None:
    """The library as an editable table; edits are saved as they are made."""
    source_ids = [meta.source_id for meta in sources]
    st.session_state["library_source_ids"] = source_ids

    def on_change() -> None:
        state = st.session_state.get("library_table") or {}
        _apply_table_edits(state.get("edited_rows", {}), st.session_state["library_source_ids"])

    st.data_editor(
        _library_rows(sources),
        key="library_table",
        on_change=on_change,
        hide_index=True,
        disabled=[c for c in LIBRARY_COLUMNS if c not in ("Title", "Author")],
        column_config={
            "Title": st.column_config.TextColumn(help="Used in citations. Click to edit."),
            "Author": st.column_config.TextColumn(help="Used in citations. Click to edit."),
        },
    )


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


def _tiers_summary(sources: list[SourceMeta]) -> str:
    """One line per source naming the tiers it already has, for the sidebar."""
    lines = []
    for meta in sources:
        tiers = ", ".join(available_tiers(meta.source_id)) or "none"
        lines.append(f"**{meta.title}**: {tiers}")
    return "Tiers already embedded —  \n" + "  \n".join(lines)


def _gap_message(gap: TierGap, total: int) -> str:
    """The warning shown before a search that would first need to embed."""
    names = ", ".join(f"**{m.title}** ({m.page_count} pages)" for m in gap.missing)
    basis = "measured on this machine" if gap.rate_measured else "a default estimate until measured"
    text = (
        f"{len(gap.missing)} of {total} source(s) have no *{gap.tier}* vectors: {names} — "
        f"{gap.pages} pages in all. Embedding them would take {format_duration(gap.seconds)} "
        f"({basis}) and adds only *{gap.tier}* vectors; nothing else is touched."
    )
    if gap.alternatives:
        text += f" Or switch the tier to *{gap.alternatives[0]}*, which every source already has."
    return text


def _embed_missing(gap: TierGap) -> None:
    """Embed only the missing sources at only `gap.tier`, with one combined bar."""
    total_books = len(gap.missing)
    bar = st.progress(0.0, text="Loading model…")
    for position, meta in enumerate(gap.missing, start=1):

        def tick(done: int, total: int, position: int = position, title: str = meta.title) -> None:
            fraction = (position - 1 + done / max(total, 1)) / total_books
            bar.progress(
                min(fraction, 1.0),
                text=f"Book {position} of {total_books}: {title} — page {done} of {total}",
            )

        embed_source(meta, gap.tier, progress=tick)
    bar.empty()


def _ensure_tier(tier: str, source: str | None) -> bool:
    """True when every in-scope source has `tier` vectors — embedding on request.

    A search at a tier the library lacks is not started silently: the gap and
    its cost are shown and the search waits for "Embed now".
    """
    scoped = [resolve_source(source)] if source else list_sources()
    gap = tier_gap(tier, scoped)
    if gap.is_empty:
        return True
    st.warning(_gap_message(gap, len(scoped)))
    if not st.button(f"Embed now ({format_duration(gap.seconds)})", key="embed_now"):
        return False
    _embed_missing(gap)
    return True


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
            st.caption(_tiers_summary(sources))
            names = ["All sources"] + [s.title for s in sources]
            chosen = st.selectbox("Search in", names, key="source_filter")
            source = None if chosen == "All sources" else chosen
        else:
            source = None
            st.caption("No sources yet. Add a PDF on the right.")

        st.divider()
        st.subheader("Add a PDF")
        uploads = st.file_uploader(
            "Drag and drop PDFs", type=["pdf"], accept_multiple_files=True, key="upload"
        )
        if len(uploads) > 1:
            author = title = ""
            st.caption("Metadata comes from each PDF; correct it in the library table.")
        else:
            author = st.text_input("Author", placeholder="Seth Holmes", key="author")
            title = st.text_input("Title", placeholder="Fresh Fruit, Broken Bodies", key="title")
        with st.expander("Chunking"):
            size = st.number_input("Sentences per chunk", 1, 10, DEFAULT_SIZE, key="chunk_size")
            overlap = st.number_input("Overlap", 0, 9, DEFAULT_OVERLAP, key="overlap")
        label = f"Index {len(uploads)} PDFs" if len(uploads) > 1 else "Index this PDF"
        if uploads and st.button(label, type="primary", key="index"):
            if overlap >= size:
                st.error("Overlap must be smaller than the chunk size.")
            else:
                _index_uploads(uploads, author, title, tier, int(size), int(overlap))
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

    sources = list_sources()
    if sources:
        with st.expander(f"Library — {len(sources)} source(s)", expanded=False):
            _library_table(sources)

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

    # Clicking "Embed now" reruns the script, so the search that asked for it
    # is remembered in session state and resumed once the tier is present.
    if st.button("Search", type="primary", key="search") and query.strip():
        st.session_state["pending"] = {"query": query, "k": k, "tier": tier, "source": source}
    pending = st.session_state.get("pending")
    if not pending:
        if not sources:
            st.info("Add a PDF in the sidebar to get started.")
        return

    try:
        _upgrade_stale_sources(pending["source"])
        if not _ensure_tier(pending["tier"], pending["source"]):
            return
        with st.spinner("Searching…"):
            results = search(
                pending["query"], k=pending["k"], tier=pending["tier"], source=pending["source"]
            )
    except (SourceNotFoundError, AmbiguousSourceError, ValueError) as exc:
        st.error(str(exc))
        return
    tier = pending["tier"]

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
