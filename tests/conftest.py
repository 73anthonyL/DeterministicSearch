"""Shared helpers for building tiny synthetic PDFs in tests."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pymupdf
import pytest

BODY_SIZE = 11.0
HEADING_SIZE = 20.0

# `insert_text` does not wrap: anything past the right margin is clipped and
# never makes it into the extracted text layer. Fixture prose is wrapped by hand
# so test pages behave like real typeset pages instead of losing their tails.
WRAP_COLUMNS = 88


def make_pdf(
    path: Path,
    pages: list[str],
    *,
    folios: list[str | None] | None = None,
    headings: dict[int, str] | None = None,
    toc: list[tuple[int, str, int]] | None = None,
    metadata: dict[str, str] | None = None,
) -> Path:
    """Write a small text PDF.

    `pages` is one body string per page. `folios` supplies the line printed at
    the bottom of each page, `headings` maps a 1-based page number to a
    large-font heading, and `toc` writes a real PDF outline.
    """
    doc = pymupdf.open()
    for index, body in enumerate(pages):
        page = doc.new_page()
        cursor = 72.0
        heading = (headings or {}).get(index + 1)
        if heading:
            page.insert_text((72, cursor), heading, fontsize=HEADING_SIZE)
            cursor += HEADING_SIZE * 2
        for line in body.splitlines() or [""]:
            for wrapped in textwrap.wrap(line, WRAP_COLUMNS) or [""]:
                page.insert_text((72, cursor), wrapped, fontsize=BODY_SIZE)
                cursor += BODY_SIZE * 1.5
        folio = (folios or [None] * len(pages))[index]
        if folio is not None:
            page.insert_text((72, 720), folio, fontsize=BODY_SIZE)
    if toc:
        doc.set_toc([list(entry) for entry in toc])
    if metadata:
        doc.set_metadata({**doc.metadata, **metadata})
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def pdf_factory(tmp_path: Path):
    """Return a factory that writes PDFs into the test's tmp_path."""

    def factory(name: str = "book.pdf", **kwargs) -> Path:
        return make_pdf(tmp_path / name, **kwargs)

    return factory
