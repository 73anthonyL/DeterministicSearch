"""Typeset the bundled sample PDF from the public-domain text next to it.

    uv run python scripts/make_sample.py            # writes src/dsearch/data/sample.pdf

The point is not a pretty book but a *realistic* one: front matter with
roman folios before the arabic ones start, a real outline, running heads
that alternate between the book title and the chapter title, a folio on
every page, and words hyphenated at the right margin. Those are the things
the extractor has to get right on a real book, so the sample exercises them.
The output is byte-for-byte reproducible (fixed metadata dates, no random
document id), so a regenerated file shows up as unchanged in git.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
DEFAULT_TEXT = HERE.parent / "src" / "dsearch" / "data" / "souls_of_black_folk.txt"
DEFAULT_OUT = DEFAULT_TEXT.with_name("sample.pdf")

PAGE_W, PAGE_H = 612.0, 792.0  # US letter

# Base-14 fonts through TextWriter: MuPDF's own font files, which carry the
# typographic characters (em dashes, curly quotes, ellipses) that the plain
# insert_text path, limited to Latin-1, silently turns into a middle dot.
FONTS = {name: pymupdf.Font(name) for name in ("tiro", "tibo", "tiit")}
MARGIN_X, TOP, BOTTOM = 90.0, 96.0, 700.0
BODY_FONT, BODY_SIZE, LEADING = "tiro", 11.0, 15.0
HEAD_FONT, HEAD_SIZE = "tibo", 18.0
SMALL_FONT, SMALL_SIZE = "tiit", 9.0  # running heads
FOLIO_Y = 736.0
HEAD_Y = 60.0
PARA_GAP = 7.0
VERSE_INDENT = 36.0
MIN_HYPHEN_WORD = 9  # letters; shorter words are pushed to the next line whole

TITLE = "The Souls of Black Folk"
AUTHOR = "Du Bois, W. E. B."
AUTHOR_DISPLAY = "W. E. B. Du Bois"
FRONT_MATTER_PAGES = 2  # title page + provenance page, before the Forethought


@dataclass
class Section:
    title: str
    paragraphs: list[tuple[str, bool]] = field(default_factory=list)  # (text, is_verse)
    front_matter: bool = False


def parse(text: str) -> list[Section]:
    """Sections from the `## heading` markup; indented lines are verse."""
    sections: list[Section] = []
    current: Section | None = None
    for block in re.split(r"\n\s*\n", text.strip()):
        if block.startswith("# "):
            continue  # the book title and provenance line
        if block.startswith("## "):
            title = block[3:].strip()
            current = Section(title=title, front_matter=title.lower().startswith("the forethought"))
            sections.append(current)
            continue
        if current is None:
            continue  # the provenance paragraph before the first heading
        verse = all(line.startswith("    ") for line in block.splitlines())
        current.paragraphs.append((block if verse else " ".join(block.split()), verse))
    return sections


def width(text: str, font: str = BODY_FONT, size: float = BODY_SIZE) -> float:
    # pymupdf's stubs say `int`; the function takes any number.
    return FONTS[font].text_length(text, fontsize=size)  # pyright: ignore[reportArgumentType]


class PageWriter:
    """One TextWriter per page, flushed once: a font is embedded once, not per text run."""

    def __init__(self, page: pymupdf.Page) -> None:
        self.page = page
        self.writer = pymupdf.TextWriter(page.rect)

    def put(self, x: float, y: float, text: str, font: str, size: float) -> None:
        self.writer.append((x, y), text, font=FONTS[font], fontsize=size)  # pyright: ignore[reportArgumentType]

    @property
    def number(self) -> int:
        """1-based page number."""
        return (self.page.number or 0) + 1

    def flush(self) -> None:
        self.writer.write_text(self.page)


def put(page: PageWriter, x: float, y: float, text: str, font: str, size: float) -> None:
    """Write `text` at (x, y) with the font's own glyphs, so every character survives."""
    page.put(x, y, text, font, size)


def hyphenate(word: str, room: float) -> tuple[str, str] | None:
    """Split `word` so that the first part plus a hyphen fits in `room`, at a vowel boundary.

    A crude rule, but it only has to produce the kind of break a typesetter
    would: never within the first or last three letters, and after a vowel.
    """
    letters = re.sub(r"[^A-Za-z]", "", word)
    if len(letters) < MIN_HYPHEN_WORD or not word.isalpha():
        return None
    best = None
    for cut in range(3, len(word) - 3):
        if word[cut - 1].lower() in "aeiouy" and word[cut].lower() not in "aeiouy":
            head = word[:cut] + "-"
            if width(head) <= room:
                best = (head, word[cut:])
    return best


def break_lines(paragraph: str, max_width: float) -> list[str]:
    """Greedy line breaking by measured width, hyphenating long words at the margin."""
    lines: list[str] = []
    current: list[str] = []
    words = paragraph.split()
    i = 0
    while i < len(words):
        word = words[i]
        trial = " ".join([*current, word])
        if width(trial) <= max_width:
            current.append(word)
            i += 1
            continue
        room = max_width - width(" ".join(current)) - width(" ")
        split = hyphenate(word, room) if current else None
        if split:
            current.append(split[0])
            words[i] = split[1]
        lines.append(" ".join(current))
        current = []
    if current:
        lines.append(" ".join(current))
    return lines


def roman(n: int) -> str:
    out, table = "", [(10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    for value, glyph in table:
        while n >= value:
            out, n = out + glyph, n - value
    return out


class Typesetter:
    def __init__(self) -> None:
        self.doc = pymupdf.open()
        self.page: PageWriter | None = None
        self.y = TOP
        self.section: Section | None = None
        self.toc: list[list] = []
        self.arabic_start: int | None = None  # pdf page where folio 1 falls

    # -- pages --------------------------------------------------------------
    @property
    def current(self) -> PageWriter:
        """The page being set; every content method runs after `new_page`."""
        if self.page is None:
            raise RuntimeError("no page started")
        return self.page

    def new_page(self) -> PageWriter:
        if self.page is not None:
            self.page.flush()
        self.page = PageWriter(self.doc.new_page(width=PAGE_W, height=PAGE_H))
        self.y = TOP
        self.furniture(self.page.number)
        return self.page

    def furniture(self, number: int) -> None:
        """Running head and folio. Front matter: roman folio, no head."""
        if self.arabic_start is None:
            folio = roman(number - FRONT_MATTER_PAGES) if number > FRONT_MATTER_PAGES else ""
        else:
            folio = str(number - self.arabic_start + 1)
            # Verso pages carry the book title, recto pages the chapter.
            head = (
                TITLE
                if folio and int(folio) % 2 == 0
                else (self.section.title if self.section else "")
            )
            head = head.upper()
            x = (
                MARGIN_X
                if int(folio) % 2 == 0
                else PAGE_W - MARGIN_X - width(head, SMALL_FONT, SMALL_SIZE)
            )
            put(self.current, x, HEAD_Y, head, SMALL_FONT, SMALL_SIZE)
        if folio:
            x = (PAGE_W - width(folio, BODY_FONT, BODY_SIZE)) / 2
            put(self.current, x, FOLIO_Y, folio, BODY_FONT, BODY_SIZE)

    def ensure_room(self, needed: float) -> None:
        if self.page is None or self.y + needed > BOTTOM:
            self.new_page()

    # -- content ------------------------------------------------------------
    def title_page(self) -> None:
        page = self.new_page()
        put(page, MARGIN_X, 260, TITLE, HEAD_FONT, 28)
        put(page, MARGIN_X, 300, "Essays and Sketches", SMALL_FONT, 14)
        put(page, MARGIN_X, 360, AUTHOR_DISPLAY, BODY_FONT, 14)
        put(page, MARGIN_X, 640, "Chicago: A. C. McClurg & Co., 1903", BODY_FONT, 10)
        page = self.new_page()
        note = [
            "This sample is typeset from the Project Gutenberg text of The Souls of",
            "Black Folk (ebook #408), which is in the public domain. It contains the",
            "Forethought and the first three chapters. It ships with dsearch so that",
            "`dsearch demo` has something real to search, and so that the extractor is",
            "tested against front matter, an outline, running heads, printed folios,",
            "and hyphenation at the margin. Regenerate it with scripts/make_sample.py.",
        ]
        for n, line in enumerate(note):
            put(page, MARGIN_X, 420 + n * LEADING, line, BODY_FONT, 10)

    def heading(self, section: Section) -> None:
        self.section = section
        self.new_page()
        if not section.front_matter and self.arabic_start is None:
            # The first chapter opens the arabic sequence: redraw this page's furniture.
            self.arabic_start = self.current.number
            self.doc.delete_page(self.current.number - 1)
            self.page = PageWriter(self.doc.new_page(width=PAGE_W, height=PAGE_H))
            self.furniture(self.page.number)
        self.y = TOP + 60
        number, _, rest = section.title.partition(". ")
        if rest:
            spaced = " ".join(number)  # "I I I" as some typesetters letter-space numerals
            put(self.current, MARGIN_X, self.y, spaced, SMALL_FONT, 12)
            self.y += 28
            title = rest
        else:
            title = section.title
        put(self.current, MARGIN_X, self.y, title, HEAD_FONT, HEAD_SIZE)
        self.toc.append([1, section.title, self.current.number])
        self.y += 40

    def paragraph(self, text: str, verse: bool) -> None:
        if verse:
            lines = [line[4:] for line in text.splitlines()]
            for line in lines:
                self.ensure_room(LEADING)
                indent = VERSE_INDENT + (len(line) - len(line.lstrip(" "))) * 3
                put(self.current, MARGIN_X + indent, self.y, line.strip(), SMALL_FONT, 10.5)
                self.y += LEADING
            self.y += PARA_GAP
            return
        lines = break_lines(text, PAGE_W - 2 * MARGIN_X)
        for n, line in enumerate(lines):
            self.ensure_room(LEADING)
            x = MARGIN_X + (18 if n == 0 else 0)
            put(self.current, x, self.y, line, BODY_FONT, BODY_SIZE)
            self.y += LEADING
        self.y += PARA_GAP

    def finish(self, out: Path) -> None:
        if self.page is not None:
            self.page.flush()
        self.doc.set_toc(self.toc)
        self.doc.set_metadata(
            {
                "title": TITLE,
                "author": AUTHOR,
                "subject": "Sample for dsearch; public-domain text from Project Gutenberg #408",
                "creator": "scripts/make_sample.py",
                "producer": "dsearch",
                "creationDate": "D:20261001000000Z",
                "modDate": "D:20261001000000Z",
            }
        )
        self.doc.save(out, garbage=4, deflate=True, no_new_id=True)
        self.doc.close()


def build(text_path: Path, out: Path) -> int:
    sections = parse(text_path.read_text(encoding="utf-8"))
    ts = Typesetter()
    ts.title_page()
    for section in sections:
        ts.heading(section)
        for text, verse in section.paragraphs:
            ts.paragraph(text, verse)
    count = ts.doc.page_count
    ts.finish(out)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--text", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    pages = build(args.text, args.out)
    print(f"wrote {args.out} ({pages} pages, {args.out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
