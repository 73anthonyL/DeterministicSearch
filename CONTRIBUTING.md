# Contributing

Thanks for looking at dsearch. The tool is small enough that one afternoon is
enough to read all of it; this page is the short version of how it is built.

## Set up

```bash
git clone https://github.com/73anthonyL/DeterministicSearch.git && cd DeterministicSearch
uv sync --all-extras          # Python 3.11+; installs the locked dev tools too
uv run pre-commit install     # ruff, pyright, codespell, and a large-file guard on every commit
```

## Check your work

```bash
uv run pytest                 # offline, a few seconds; coverage is printed
uv run ruff check . && uv run ruff format --check .
uv run pyright
```

CI runs exactly these on Linux, macOS, and Windows across Python 3.11 to 3.13,
then builds the wheel and installs it into an empty environment.

## Commits and pull requests

- Branch from `main`; one change per pull request.
- Pull requests are squash-merged, and the **PR title becomes the commit
  message**, so it must be a conventional commit:
  `feat(search): ...`, `fix(extract): ...`, `docs: ...`, `test: ...`,
  `refactor: ...`, `build: ...`, `ci: ...`, `chore: ...`. A check enforces it.
- Add a line under `Unreleased` in `CHANGELOG.md` for anything a user would
  notice.
- A choice the code cannot explain by itself goes in `DECISIONS.md`, with the
  reason, so it can be defended or reversed on purpose.
- Never commit a copyrighted PDF or anything from `~/.dsearch`. A pre-commit
  hook refuses files over 1 MB.
- No AI-attribution trailers on commits.

## Reporting an extraction problem

If a PDF comes out with mangled text, missing page numbers, or wrong chapter
titles, open an issue with the PDF's producer (`python -c "import pymupdf;
print(pymupdf.open('file.pdf').metadata)"`), the `pymupdf` version, and if
possible a single redacted page that shows the problem.
