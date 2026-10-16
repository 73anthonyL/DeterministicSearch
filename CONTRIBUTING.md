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

Maintainer only. The package is published as `deterministic-search`; the
command and the import stay `dsearch`. Publishing uses PyPI's trusted
publishing: the release workflow proves its identity to the index with an
OIDC token, and no API token exists anywhere.

1. **Rehearse.** Set a pre-release version (`uv version 0.2.0rc1`), merge it,
   and run the *Release* workflow by hand with target `testpypi`. It builds,
   uploads to TestPyPI, installs the upload into an empty environment, and
   runs `dsearch demo`. An index never accepts the same version twice, so a
   failed rehearsal means `rc2`, not a retry.
2. **Release PR.** `uv version x.y.z`; move the `Unreleased` section of
   `CHANGELOG.md` under `## [x.y.z] - YYYY-MM-DD`; update `version` and
   `date-released` in `CITATION.cff`. Merge as `chore(release): x.y.z`.
3. **Tag.** `git switch main && git pull --ff-only && git tag -a vx.y.z -m "vx.y.z"
   && git push origin vx.y.z`. The workflow checks that the tag and
   `pyproject.toml` agree, publishes to TestPyPI, and stops at the `pypi`
   environment for approval.
4. **Approve.** Review the TestPyPI job's output, then approve the `pypi`
   deployment under the workflow run. The workflow publishes to PyPI, creates
   the GitHub Release with notes from the commits since the previous tag
   (`cliff.toml`), and attaches the wheel and sdist.
5. **Confirm.** `pipx install "deterministic-search[ui]==x.y.z"` on a clean
   machine; `dsearch demo`.
