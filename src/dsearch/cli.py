"""The `dsearch` command line: add, list, remove, search."""

from __future__ import annotations

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
from dsearch.cite import mla, parenthetical
from dsearch.extract import NoTextLayerError
from dsearch.index import (
    DEFAULT_TIER,
    TIERS,
    AmbiguousSourceError,
    SourceNotFoundError,
    home,
)
from dsearch.index import (
    add as index_add,
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
    citation = mla(result)
    inline = parenthetical(result)
    footer = f"[green]{citation}[/green]"
    if inline:
        footer += f"  [dim]in-text: {inline}[/dim]"
    return Panel(
        Group(_body(result), Text(), _render_markup(footer)),
        title=f"[dim]{position}.[/dim] {_header(result)}",
        title_align="left",
        border_style="blue",
        padding=(1, 2),
    )


def _render_markup(markup: str) -> Text:
    return Text.from_markup(markup)


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


@app.command()
def add(
    pdf: Annotated[Path, typer.Argument(help="Path to the PDF to index.")],
    author: Annotated[str | None, typer.Option(help="Author, for citations.")] = None,
    title: Annotated[str | None, typer.Option(help="Title, for citations.")] = None,
    tier: Annotated[str, typer.Option(help=_TIER_HELP)] = DEFAULT_TIER,
    chunk_size: Annotated[int, typer.Option(help="Sentences per chunk.")] = DEFAULT_SIZE,
    overlap: Annotated[
        int, typer.Option(help="Sentences shared between chunks.")
    ] = DEFAULT_OVERLAP,
) -> None:
    """Index a PDF into your library."""
    columns = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        _PageColumn(),
        TimeElapsedColumn(),
    )
    try:
        with Progress(*columns, console=console, transient=True) as progress:
            task = progress.add_task(f"Embedding [bold]{pdf.name}[/bold]", total=None)

            def tick(done: int, total: int) -> None:
                progress.update(task, completed=done, total=total)

            result = index_add(
                pdf,
                author=author,
                title=title,
                tier=tier,
                chunk_size=chunk_size,
                overlap=overlap,
                progress=tick,
            )
    except FileNotFoundError as exc:
        _fail(str(exc))
    except NoTextLayerError as exc:
        _fail(str(exc))
    except ValueError as exc:
        _fail(str(exc))

    meta = result.meta
    if result.already_indexed:
        console.print(
            f"[yellow]Already indexed[/yellow] [bold]{meta.title}[/bold] "
            f"({meta.short_id}) at tier [bold]{tier}[/bold] — nothing to do."
        )
        return

    console.print(
        f"[green]Indexed[/green] [bold]{meta.title}[/bold] ({meta.short_id}) — "
        f"{meta.page_count} pages, {meta.chunk_count} chunks, tier [bold]{tier}[/bold]."
    )
    if result.warning:
        console.print(f"[yellow]Note[/yellow] {result.warning}")


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
            ", ".join(source.tiers) or "[dim]none[/dim]",
            source.added,
        )
    console.print(table)
    console.print(f"\n[dim]{len(sources)} source(s) in {home()}[/dim]")


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
) -> None:
    """Find page-cited passages that answer a query."""
    try:
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
