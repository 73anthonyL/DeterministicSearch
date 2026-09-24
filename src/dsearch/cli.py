"""The `dsearch` command line: add, list, remove, search."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from dsearch import __version__
from dsearch.chunk import DEFAULT_OVERLAP, DEFAULT_SIZE
from dsearch.cite import mla, quoted_line
from dsearch.extract import NoTextLayerError
from dsearch.index import (
    DEFAULT_TIER,
    TIERS,
    AddResult,
    AmbiguousSourceError,
    SourceMeta,
    SourceNotFoundError,
    TierGap,
    available_tiers,
    embed_source,
    format_duration,
    home,
    resolve_source,
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
from dsearch.search import DEFAULT_K, Result
from dsearch.search import search as run_search

# Context from a neighbouring chunk is trimmed to this many characters so a
# `--k 20` search stays readable in one screenful.
NEIGHBOUR_CHARS = 180

app = typer.Typer(
    name="dsearch",
    help="Local-first semantic evidence search for academic PDFs. Retrieval only.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()
errors = Console(stderr=True)

_TIER_HELP = f"Embedding quality: {' | '.join(TIERS)}."


def _fail(message: str) -> None:
    """Print a clean error and exit 1, rather than a traceback."""
    errors.print(f"[bold red]Error[/bold red] {message}")
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

    def render(self, task) -> Text:
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
    console.print(f"[yellow]Note[/yellow] {message}")


def _scoped_sources(source: str | None) -> list[SourceMeta]:
    """The sources a search or embed will touch: one named source, or all."""
    return [resolve_source(source)] if source else index_list()


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
    tier: Annotated[str, typer.Option(help=_TIER_HELP)] = DEFAULT_TIER,
    chunk_size: Annotated[int, typer.Option(help="Sentences per chunk.")] = DEFAULT_SIZE,
    overlap: Annotated[
        int, typer.Option(help="Sentences shared between chunks.")
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
                    tier=tier,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    progress=tick,
                    notify=_note,
                )
            except (FileNotFoundError, NoTextLayerError, ValueError) as exc:
                failures += 1
                errors.print(f"[bold red]Error[/bold red] {path}: {exc}")
                continue
            _report_add(result, tier)
            warning = result.warning or warning

    if warning:
        _note(warning)
    if failures:
        raise typer.Exit(code=1)


@app.command("list")
def list_command() -> None:
    """Show every source in your library."""
    sources = index_list()
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
    k: Annotated[int, typer.Option("--k", "-k", help="How many passages to return.")] = DEFAULT_K,
    source: Annotated[str | None, typer.Option(help="Limit to one source.")] = None,
    tier: Annotated[str, typer.Option(help=_TIER_HELP)] = DEFAULT_TIER,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Embed missing tiers without asking.")
    ] = False,
) -> None:
    """Find page-cited passages that answer a query."""
    try:
        scoped = _upgrade_stale_sources(_scoped_sources(source))
        gap = tier_gap(tier, scoped)
        if not gap.is_empty:
            _describe_gap(gap, len(scoped))
            if not (yes or _confirm_embedding(gap)):
                console.print("[dim]Nothing embedded; search not run.[/dim]")
                return
            _embed_missing(gap)
        results = run_search(query, k=k, tier=tier, source=source)
    except (SourceNotFoundError, AmbiguousSourceError) as exc:
        _fail(str(exc))
    except ValueError as exc:
        _fail(str(exc))

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
    tier: Annotated[str, typer.Option(help=_TIER_HELP)] = DEFAULT_TIER,
    source: Annotated[
        list[str] | None,
        typer.Option(help="Limit to these sources (repeatable). Default: the whole library."),
    ] = None,
) -> None:
    """Pre-embed a quality tier so later searches at it never have to wait."""
    try:
        scoped = [resolve_source(name) for name in source] if source else index_list()
        scoped = _upgrade_stale_sources(scoped)
        gap = tier_gap(tier, scoped)
    except (SourceNotFoundError, AmbiguousSourceError, ValueError) as exc:
        _fail(str(exc))

    if not scoped:
        console.print("[dim]Your library is empty. Add a PDF with[/dim] dsearch add <file.pdf>")
        return
    if gap.is_empty:
        console.print(
            f"[green]Nothing to do[/green] — all {len(scoped)} source(s) already have "
            f"[bold]{tier}[/bold] vectors."
        )
        return
    _describe_gap(gap, len(scoped))
    _embed_missing(gap)


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
) -> None:
    """Local-first semantic evidence search for academic PDFs."""


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
