"""The `dsearch` command line: add, list, edit, remove, search, embed, terms, doctor, ui."""

from __future__ import annotations

import dataclasses
import glob
import importlib.resources
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from dsearch import __version__, log
from dsearch.chunk import DEFAULT_OVERLAP, DEFAULT_SIZE
from dsearch.cite import citation_block, mla, parenthetical, quoted_line
from dsearch.doctor import diagnostics, format_bytes
from dsearch.extract import NoTextLayerError
from dsearch.index import (
    TIERS,
    AddResult,
    AmbiguousSourceError,
    ModelUnavailableError,
    SourceMeta,
    SourceNotFoundError,
    Tier,
    TierGap,
    available_tiers,
    embed_source,
    format_duration,
    home,
    is_stale,
    resolve_sources,
    tier_gap,
    upgrade_stale,
)
from dsearch.index import (
    add as index_add,
)
from dsearch.index import (
    edit as index_edit,
)
from dsearch.index import (
    list_sources as index_list,
)
from dsearch.index import (
    remove as index_remove,
)
from dsearch.search import DEFAULT_K, Result, load_corpus
from dsearch.search import search as run_search
from dsearch.terms import DEFAULT_TOP, BookTerms, key_terms

# Context from a neighbouring chunk is trimmed to this many characters so a
# `--k 20` search stays readable in one screenful.
NEIGHBOUR_CHARS = 180

app = typer.Typer(
    name="dsearch",
    help="Local-first semantic evidence search for academic PDFs. Retrieval only.",
    no_args_is_help=True,
)
console = Console()
errors = Console(stderr=True)

_TIER_HELP = f"Embedding quality: {' | '.join(TIERS)}."
_JSON_HELP = "Print JSON to stdout instead of a table or panels."

# Set by --quiet: informational notes are dropped, errors are not.
_quiet = False


def _fail(message: str) -> NoReturn:
    """Print a clean error and exit 1, rather than a traceback.

    The message is escaped: it carries filenames and user input, and a path
    such as `[draft].pdf` must print as typed rather than be read as markup.
    """
    errors.print(f"[bold red]Error[/bold red] {escape(message)}")
    raise typer.Exit(code=1)


def _truncate(text: str, limit: int = NEIGHBOUR_CHARS) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _header(result: Result) -> str:
    """Source title, chapter if known, and the page reference."""
    parts = [f"[bold]{result.meta.title}[/bold]"]
    if result.chunk.chapter:
        parts.append(f"[italic]{result.chunk.chapter}[/italic]")
    parts.append(result.page_label)
    return "  ·  ".join(parts)


def _body(result: Result) -> Group:
    """Context dimmed, the matching sentence bold."""
    lines: list[Text] = []

    if result.before:
        lines.append(Text(_truncate(result.before.text), style="dim"))

    passage = Text()
    for position, sentence in enumerate(result.chunk.sentences):
        if position:
            passage.append(" ")
        if position == result.best_sentence:
            passage.append(sentence, style="bold")
        else:
            passage.append(sentence, style="dim")
    lines.append(passage)

    if result.after:
        lines.append(Text(_truncate(result.after.text), style="dim"))

    return Group(*lines)


def _panel(result: Result, position: int) -> Panel:
    """A result panel: passage, then the reference line and the quotable line."""
    footer = Group(
        Text(mla(result), style="green"),
        Text(quoted_line(result), style="dim"),
    )
    return Panel(
        Group(_body(result), Text(), footer),
        title=f"[dim]{position}.[/dim] {_header(result)}",
        title_align="left",
        border_style="blue",
        padding=(1, 2),
    )


class _PageColumn(ProgressColumn):
    """Progress in pages of the PDF, the unit a reader recognises.

    The page count is only known once extraction finishes, which is after the
    bar is already on screen, so an unknown total renders as blank rather than
    crashing on a None format.
    """

    def render(self, task: Task) -> Text:
        if not task.total:
            return Text("reading pages…", style="dim")
        return Text(f"page {int(task.completed)}/{int(task.total)}", style="dim")


def _page_progress() -> Progress:
    """A transient progress bar that counts pages, shared by every embedding job."""
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        _PageColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


def _note(message: str) -> None:
    """An informational line, on stderr so that `--json` output stays parseable."""
    if not _quiet:
        errors.print(f"[yellow]Note[/yellow] {message}")


def _emit_json(payload: object) -> None:
    """Machine-readable output: plain stdout, no Rich wrapping or styling."""
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _result_json(result: Result) -> dict:
    """A result plus the three citation forms the panel shows."""
    return {
        **result.to_dict(),
        "citation": {
            "mla": mla(result),
            "in_text": parenthetical(result),
            "quoted_line": quoted_line(result),
            "block": citation_block(result),
        },
    }


def _scoped_sources(source: str | None) -> list[SourceMeta]:
    """The sources a search or embed will touch: one file, one book by title, or all."""
    return resolve_sources(source) if source else index_list()


def _upgrade_stale_sources(sources: list[SourceMeta]) -> list[SourceMeta]:
    """Rebuild any in-scope source indexed by an older pipeline before searching."""
    with _page_progress() as progress:
        task = progress.add_task("Re-indexing", total=None)

        def tick(done: int, total: int) -> None:
            progress.update(task, completed=done, total=total)

        return upgrade_stale(sources, progress=tick, notify=_note)


def _describe_gap(gap: TierGap, total: int) -> None:
    """Say which sources lack the tier and how long filling it should take."""
    names = ", ".join(f"{m.title} ({m.page_count} pages)" for m in gap.missing)
    console.print(
        f"[yellow]Note[/yellow] {len(gap.missing)} of {total} source(s) have no "
        f"[bold]{gap.tier}[/bold] vectors: {names} — {gap.pages} pages in all."
    )
    basis = "measured on this machine" if gap.rate_measured else "default estimate until measured"
    console.print(
        f"       Embedding them would take {format_duration(gap.seconds)} "
        f"({basis}) and only adds [bold]{gap.tier}[/bold] vectors — nothing else is touched."
    )


def _confirm_embedding(gap: TierGap) -> bool:
    """Ask before a long job, naming the tier that needs no wait when there is one."""
    prompt = f"Embed {gap.tier} vectors for {len(gap.missing)} source(s) now?"
    if gap.alternatives:
        prompt += f" (or run with `--tier {gap.alternatives[0]}`, which all sources already have)"
    try:
        return typer.confirm(prompt, default=False)
    except typer.Abort:  # Non-interactive stdin: the safe answer is no.
        console.print()
        return False


def _embed_missing(gap: TierGap) -> None:
    """Embed only the missing sources at only `gap.tier`, with book and page progress."""
    with _page_progress() as progress:
        overall = None
        if len(gap.missing) > 1:
            overall = progress.add_task("Books", total=len(gap.missing))
        pages = progress.add_task("", total=None)

        def tick(done: int, total: int) -> None:
            progress.update(pages, completed=done, total=total)

        for position, meta in enumerate(gap.missing, start=1):
            if overall is not None:
                progress.update(
                    overall,
                    completed=position - 1,
                    description=f"Book {position} of {len(gap.missing)}: [bold]{meta.title}[/bold]",
                )
            progress.reset(pages, total=None, description=f"Embedding [bold]{meta.title}[/bold]")
            embed_source(meta, gap.tier, progress=tick)
            console.print(
                f"[green]Embedded[/green] [bold]{meta.title}[/bold] ({meta.short_id}) "
                f"at tier [bold]{gap.tier}[/bold]."
            )


def _expand_paths(patterns: list[str]) -> list[Path]:
    """Turn arguments into PDF paths, expanding globs the shell did not.

    A quoted pattern (`dsearch add "samples/*.pdf"`) and Windows shells both
    hand the glob through verbatim, so it is expanded here. A pattern that
    matches nothing is reported as a missing file rather than silently dropped.
    """
    paths: list[Path] = []
    for pattern in patterns:
        if any(char in pattern for char in "*?["):
            matches = sorted(Path(p) for p in glob.glob(pattern))
            if not matches:
                _fail(f"No files match {pattern!r}.")
            paths.extend(matches)
        else:
            paths.append(Path(pattern))
    return paths


def _report_add(result: AddResult, tier: str) -> None:
    """One line per file: what happened to it."""
    meta = result.meta
    if result.already_indexed:
        console.print(
            f"[yellow]Already indexed[/yellow] [bold]{meta.title}[/bold] "
            f"({meta.short_id}) at tier [bold]{tier}[/bold] — skipped."
        )
        return
    verb = "Rebuilt" if result.reindexed else "Indexed"
    console.print(
        f"[green]{verb}[/green] [bold]{meta.title}[/bold] ({meta.short_id}) — "
        f"{meta.page_count} pages, {meta.chunk_count} chunks, tier [bold]{tier}[/bold]."
    )


@app.command()
def add(
    pdfs: Annotated[list[str], typer.Argument(help="PDF paths or glob patterns.")],
    author: Annotated[
        str | None, typer.Option(help="Author, for citations (single PDF only).")
    ] = None,
    title: Annotated[
        str | None, typer.Option(help="Title, for citations (single PDF only).")
    ] = None,
    tier: Annotated[Tier, typer.Option(help=_TIER_HELP)] = Tier.fast,
    chunk_size: Annotated[int, typer.Option(min=1, help="Sentences per chunk.")] = DEFAULT_SIZE,
    overlap: Annotated[
        int, typer.Option(min=0, help="Sentences shared between chunks.")
    ] = DEFAULT_OVERLAP,
) -> None:
    """Index one or more PDFs into your library."""
    paths = _expand_paths(pdfs)
    if len(paths) > 1 and (author or title):
        _fail(
            "--author and --title apply to a single PDF. With several files the metadata "
            "comes from each PDF; set it afterwards with `dsearch edit <source> "
            '--author "..." --title "..."`.'
        )

    failures = 0
    warning: str | None = None
    with _page_progress() as progress:
        overall = None
        if len(paths) > 1:
            overall = progress.add_task("Books", total=len(paths))
        pages = progress.add_task("", total=None)

        def tick(done: int, total: int) -> None:
            progress.update(pages, completed=done, total=total)

        for position, path in enumerate(paths, start=1):
            if overall is not None:
                progress.update(overall, completed=position - 1)
                progress.update(
                    overall,
                    description=f"Book {position} of {len(paths)}: [bold]{path.name}[/bold]",
                )
            progress.reset(pages, total=None, description=f"Embedding [bold]{path.name}[/bold]")
            try:
                result = index_add(
                    path,
                    author=author,
                    title=title,
                    tier=tier.value,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    progress=tick,
                    notify=_note,
                )
            except ModelUnavailableError as exc:
                _fail(str(exc))  # Nothing else will succeed without the model.
            except (FileNotFoundError, NoTextLayerError, ValueError) as exc:
                failures += 1
                errors.print(f"[bold red]Error[/bold red] {escape(str(path))}: {escape(str(exc))}")
                continue
            _report_add(result, tier.value)
            warning = result.warning or warning

    if warning:
        _note(warning)
    if failures:
        raise typer.Exit(code=1)


@app.command("list")
def list_command(
    json_output: Annotated[bool, typer.Option("--json", help=_JSON_HELP)] = False,
) -> None:
    """Show every source in your library."""
    sources = index_list()
    if json_output:
        _emit_json(
            {
                "library": str(home()),
                "sources": [
                    {**meta.to_dict(), "tiers_available": available_tiers(meta.source_id)}
                    for meta in sources
                ],
            }
        )
        return
    if not sources:
        console.print("[dim]Your library is empty. Add a PDF with[/dim] dsearch add <file.pdf>")
        return

    table = Table(box=None, pad_edge=False, header_style="bold")
    for column in ("ID", "Title", "Author", "Pages", "Chunks", "Tiers", "Added"):
        table.add_column(column, justify="right" if column in {"Pages", "Chunks"} else "left")
    for source in sources:
        table.add_row(
            source.short_id,
            source.title,
            source.author or "[dim]unknown[/dim]",
            str(source.page_count),
            str(source.chunk_count),
            ", ".join(available_tiers(source.source_id)) or "[dim]none[/dim]",
            source.added,
        )
    console.print(table)
    console.print(f"\n[dim]{len(sources)} source(s) in {home()}[/dim]")


@app.command()
def edit(
    source: Annotated[str, typer.Argument(help="Source id, filename, or title.")],
    author: Annotated[str | None, typer.Option(help="New author, for citations.")] = None,
    title: Annotated[str | None, typer.Option(help="New title, for citations.")] = None,
) -> None:
    """Change a source's author or title. Chunks and vectors are untouched."""
    try:
        meta = index_edit(source, author=author, title=title)
    except (SourceNotFoundError, AmbiguousSourceError, ValueError) as exc:
        _fail(str(exc))
    console.print(
        f"[green]Updated[/green] ({meta.short_id}) — "
        f"[bold]{meta.title}[/bold] by {meta.author or '[dim]unknown[/dim]'}."
    )


@app.command()
def remove(
    source: Annotated[str, typer.Argument(help="Source id, filename, or title.")],
) -> None:
    """Delete a source from your library."""
    try:
        meta = index_remove(source)
    except (SourceNotFoundError, AmbiguousSourceError) as exc:
        _fail(str(exc))
    console.print(f"[green]Removed[/green] [bold]{meta.title}[/bold] ({meta.short_id}).")


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="A question, a phrase, or a whole paragraph.")],
    k: Annotated[
        int, typer.Option("--k", "-k", min=1, help="How many passages to return.")
    ] = DEFAULT_K,
    source: Annotated[
        str | None,
        typer.Option(help="Limit to one source, or by title to every file of one book."),
    ] = None,
    tier: Annotated[Tier, typer.Option(help=_TIER_HELP)] = Tier.fast,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Embed missing tiers without asking.")
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", help=_JSON_HELP)] = False,
) -> None:
    """Find page-cited passages that answer a query."""
    try:
        scoped = _upgrade_stale_sources(_scoped_sources(source))
        gap = tier_gap(tier.value, scoped)
        if not gap.is_empty:
            _describe_gap(gap, len(scoped))
            if not (yes or _confirm_embedding(gap)):
                errors.print("[dim]Nothing embedded; search not run.[/dim]")
                return
            _embed_missing(gap)
        corpus = load_corpus(tier.value, source)
        for name in corpus.skipped:
            _note(f"Skipped {name}: no usable {tier.value} vectors.")
        results = run_search(query, k=k, tier=tier.value, source=source, corpus=corpus)
    except (SourceNotFoundError, AmbiguousSourceError, ValueError, ModelUnavailableError) as exc:
        _fail(str(exc))

    if json_output:
        _emit_json([_result_json(result) for result in results])
        return
    if not results:
        console.print(
            "[yellow]No passages found.[/yellow] "
            "[dim]Check `dsearch list` — the library may be empty or indexed at another tier.[/dim]"
        )
        return

    console.print()
    for position, result in enumerate(results, start=1):
        console.print(_panel(result, position))


@app.command()
def embed(
    tier: Annotated[Tier, typer.Option(help=_TIER_HELP)] = Tier.fast,
    source: Annotated[
        list[str] | None,
        typer.Option(help="Limit to these sources (repeatable). Default: the whole library."),
    ] = None,
) -> None:
    """Pre-embed a quality tier so later searches at it never have to wait."""
    try:
        scoped = [m for name in source for m in resolve_sources(name)] if source else index_list()
        scoped = _upgrade_stale_sources(scoped)
        gap = tier_gap(tier.value, scoped)
    except (SourceNotFoundError, AmbiguousSourceError, ValueError, ModelUnavailableError) as exc:
        _fail(str(exc))

    if not scoped:
        console.print("[dim]Your library is empty. Add a PDF with[/dim] dsearch add <file.pdf>")
        return
    if gap.is_empty:
        console.print(
            f"[green]Nothing to do[/green] — all {len(scoped)} source(s) already have "
            f"[bold]{tier.value}[/bold] vectors."
        )
        return
    _describe_gap(gap, len(scoped))
    try:
        _embed_missing(gap)
    except ModelUnavailableError as exc:
        _fail(str(exc))


def _book_json(book: BookTerms) -> dict:
    """A book's terms as data, with each term's page label spelled out."""
    payload = dataclasses.asdict(book)
    for chapter, chapter_data in zip(book.chapters, payload["chapters"], strict=True):
        for term, term_data in zip(chapter.terms, chapter_data["terms"], strict=True):
            term_data["page_label"] = term.page_label
    return payload


def _book_heading(book: BookTerms) -> str:
    """Title and author, and how the terms under it were ranked."""
    heading = f"[bold]{book.title}[/bold]"
    if book.author:
        heading += f" — {book.author}"
    ranking = f"re-ranked at tier {book.tier}" if book.tier else "ranked by word counts"
    return f"{heading}  [dim]({ranking})[/dim]"


def _terms_table(book: BookTerms) -> Table:
    """One row per term, grouped under its chapter, with the evidence columns."""
    table = Table(box=None, pad_edge=False, header_style="bold")
    table.add_column("Chapter", style="italic", max_width=36)
    table.add_column("Term")
    table.add_column("Count", justify="right")
    table.add_column("Densest page", style="dim")
    for number, chapter in enumerate(book.chapters):
        if number:
            table.add_row()  # A blank line between chapters.
        if not chapter.terms:
            table.add_row(chapter.label, "[dim]no distinctive terms[/dim]", "", "")
        for position, term in enumerate(chapter.terms):
            label = chapter.label if position == 0 else ""
            table.add_row(label, term.text, str(term.count), term.page_label)
    return table


@app.command()
def terms(
    source: Annotated[
        str | None, typer.Option(help="Limit to one source's chapters, or one book's by title.")
    ] = None,
    tier: Annotated[Tier, typer.Option(help=_TIER_HELP)] = Tier.fast,
    top: Annotated[int, typer.Option(min=1, help="Terms to show per chapter.")] = DEFAULT_TOP,
    no_rerank: Annotated[
        bool,
        typer.Option("--no-rerank", help="Rank by word counts only; instant, loads no model."),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", help=_JSON_HELP)] = False,
) -> None:
    """Show the terms that set each chapter apart from the rest of its book."""
    try:
        for meta in _scoped_sources(source):
            if is_stale(meta):
                _note(
                    f"{meta.title or meta.filename} ({meta.short_id}) was indexed by an older "
                    "version; run `dsearch add` on its PDF again for cleaner terms."
                )
        with errors.status("Ranking terms…"):
            books = key_terms(source=source, tier=tier.value, top=top, rerank=not no_rerank)
    except (SourceNotFoundError, AmbiguousSourceError, ValueError, ModelUnavailableError) as exc:
        _fail(str(exc))

    if json_output:
        _emit_json([_book_json(book) for book in books])
        return
    if not books:
        console.print("[dim]Your library is empty. Add a PDF with[/dim] dsearch add <file.pdf>")
        return

    for book in books:
        console.print()
        console.print(_book_heading(book))
        if book.note:
            _note(book.note)
        if not book.contrasted:
            _note(
                "This book has a single chapter, so there is nothing to contrast it with: "
                "terms are ranked by frequency."
            )
        console.print(_terms_table(book))


def _doctor_report(info: dict) -> None:
    """The diagnostics as a short readable table."""
    table = Table(box=None, pad_edge=False, show_header=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("dsearch", info["dsearch"])
    table.add_row("python", f"{info['python']}  ({info['executable']})")
    table.add_row("platform", info["platform"])
    library = info["library"]
    state = (
        f"{library['sources']} source(s), {format_bytes(library['bytes'])}"
        if library["exists"]
        else "[dim]not created yet[/dim]"
    )
    table.add_row("library", f"{library['path']}  —  {state}")
    models = info["models"]
    table.add_row("model cache", models["cache"])
    for tier, detail in models["tiers"].items():
        cached = "[green]cached[/green]" if detail["cached"] else "[dim]not downloaded[/dim]"
        table.add_row(f"  {tier}", f"{detail['model']}  —  {cached}")
    table.add_row("offline", "yes (HF_HUB_OFFLINE)" if models["offline"] else "no")
    table.add_row("device", models["device"])
    for name, version in info["packages"].items():
        table.add_row(name, version or "[dim]not installed[/dim]")
    table.add_row(
        "web ui", "available" if info["ui_available"] else "[dim]pip install 'dsearch[ui]'[/dim]"
    )
    console.print(table)


@app.command()
def doctor(
    json_output: Annotated[bool, typer.Option("--json", help=_JSON_HELP)] = False,
) -> None:
    """Show versions, paths, cached models, and the compute device, for bug reports."""
    info = diagnostics()
    if json_output:
        _emit_json(info)
    else:
        _doctor_report(info)


@app.command()
def ui(
    port: Annotated[int | None, typer.Option(help="Port to serve on (default 8501).")] = None,
    headless: Annotated[
        bool, typer.Option("--headless", help="Do not open a browser tab.")
    ] = False,
) -> None:
    """Open the web UI in your browser. Needs the `ui` extra."""
    if importlib.util.find_spec("streamlit") is None:
        _fail(
            "The web UI needs streamlit. Install it with: pipx install 'dsearch[ui]' "
            "(or: pip install 'dsearch[ui]')."
        )
    # The app ships inside the package, so this works from a wheel as well as a
    # checkout. A subprocess rather than streamlit's bootstrap API: that API is
    # private and has changed shape between releases, and a child process keeps
    # Ctrl-C and the event loop entirely Streamlit's own.
    with importlib.resources.as_file(importlib.resources.files("dsearch") / "app.py") as script:
        command = [sys.executable, "-m", "streamlit", "run", str(script)]
        if port is not None:
            command += ["--server.port", str(port)]
        if headless:
            command += ["--server.headless", "true"]
        raise typer.Exit(code=subprocess.call(command))  # noqa: S603  # fixed argv, no shell


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"dsearch {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose", "-v", help="Log what the pipeline is doing (same as DSEARCH_LOG=DEBUG)."
        ),
    ] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Drop informational notes; errors still print.")
    ] = False,
) -> None:
    """Local-first semantic evidence search for academic PDFs."""
    global _quiet
    _quiet = quiet
    log.configure("DEBUG" if verbose else ("ERROR" if quiet else None), console=errors)


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
