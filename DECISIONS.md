# Decisions

Choices the spec left open, and the reasoning behind each, so they can be
defended or reversed on purpose.

## Packaging and environment

- **Build backend: hatchling.** The spec allowed hatchling or setuptools.
  Hatchling needs no `setup.py`/`MANIFEST.in` and reads the `src/` layout from
  three lines of config.
- **Python 3.12 for the development venv.** The floor is 3.11 as specified. The
  venv pins 3.12 because `torch` (pulled in by `sentence-transformers`) has no
  wheels for 3.14, which is the default `python3` on this machine.
- **`import pymupdf`, not `import fitz`.** PyMuPDF now emits a deprecation
  warning for the `fitz` alias.

## Extraction

- **Folios are searched in a header/footer *band*, not just the first and last
  line.** In the sample chapters the running head and the page number are set on
  the same band, so the folio is extracted as the second line
  (`"i n t r o d u c t i o n"`, then `"3"`). Scanning one line found 15 of 29
  folios in chapter 1; scanning a 3-line band finds 29 of 29. On pages too short
  to have a real band (<= 6 lines) only the outermost lines are trusted, so a
  stray integer in body text is not mistaken for a page number.
- **The footer band is checked before the header band.** The bottom of the page
  is the more common home for a folio; in a book that prints both, they agree.
- **Roman-numeral folios resolve to `None`.** Front-matter numbers are real, but
  mixing `p. xiv` and `p. 14` in one library would make citations ambiguous.
  Front matter is rarely the evidence a student quotes.
- **A page needs 20 characters to count as having a text layer.** Scanned pages
  are seldom exactly empty — running heads and artifacts leak a few characters
  through. The scan threshold itself (>80% of pages) is from the spec.
- **Chapter titles: PDF outline first, font size second.** The outline is
  authoritative when present. The font heuristic calls a line a heading when it
  is >= 1.25x the modal span size, is <= 90 characters, and is not itself a page
  number. Both approaches carry the last-seen title forward across pages.
