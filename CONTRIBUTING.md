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

## Issues and labels

Open an issue from one of the templates: **Bug report**, **Extraction
problem** (a PDF that comes out mangled; the template asks for the details
that make it fixable), or **Feature request**. Security problems go through
the private form linked in `SECURITY.md`, never a public issue.

Labels: `type: bug | feature | docs | chore` says what kind of change;
`area: extract | chunk | index | search | cite | terms | cli | ui | ci` says
where. `good first issue` marks something self-contained for a newcomer.
Release notes are grouped by these labels (`.github/release.yml`).

## Releases

Maintainer only. Bump the version in `pyproject.toml`, move the `Unreleased`
section of `CHANGELOG.md` under the new version, merge that as
`chore(release): x.y.z`, and push a `vx.y.z` tag. The release workflow
builds, publishes, and writes the GitHub Release.
