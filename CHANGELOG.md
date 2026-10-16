# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `dsearch demo` indexes a bundled public-domain sample (the Forethought and
  first three chapters of Du Bois, *The Souls of Black Folk*, typeset with
  front matter, an outline, running heads, folios, and hyphenation) and runs
  one search, so a fresh install has something to try in two minutes.
- `dsearch doctor` prints versions, the library path and size, the model
  cache and which tiers are downloaded, the compute device, and whether the
  web UI is installed; `--json` for bug reports.
- `--json` on `list`, `search`, and `terms` prints machine-readable output.
  A search result carries the passage, its page, its context, and the MLA,
  in-text, and quoted-line citations.
- `-v/--verbose` (or `DSEARCH_LOG=DEBUG`) logs model loads, corpus sizes,
  and embedding throughput to stderr; `-q/--quiet` drops informational notes.
- Shell completion: `dsearch --install-completion`.
- `dsearch ui` opens the web UI from any install, including a wheel; it needs
  the new `ui` extra (`pip install 'dsearch[ui]'`).
- `py.typed`, so type checkers read the package's annotations.

### Changed

- Continuous integration now runs on Linux, macOS, and Windows across Python
  3.11 to 3.13, type-checks with pyright, checks spelling, enforces a coverage
  floor, and builds and smoke-tests the wheel. Actions are pinned to commits
  and updated by Dependabot. Pull request titles must be conventional commits.
- A pre-commit configuration runs the same checks locally; see
  `CONTRIBUTING.md`.
- Community files: a code of conduct, a security policy with private
  reporting, issue templates (bug, extraction problem, feature), a pull
  request template, `CODEOWNERS`, release-notes categories, and a
  `CITATION.cff`.
- Streamlit is now optional. The core install no longer pulls in pandas and
  pyarrow for people who only use the command line.
- The version is declared once, in `pyproject.toml`, and read from package
  metadata at runtime.
- Development dependencies moved from the `dev` extra to a PEP 735 dependency
  group; `uv sync` installs them, and `uv.lock` is committed. On Linux, uv
  installs CPU-only torch from the PyTorch index.

### Fixed

- A running head that changes with each chapter (so it repeats on only a few
  pages of a whole book) is now recognised when it opens or closes at least
  three pages, and no longer leaks into quoted passages.
- Progress bars and advisories from the Hugging Face and torch libraries no
  longer print during a model load unless you have configured them yourself.
- A model that cannot be downloaded (no network, or `HF_HUB_OFFLINE` set
  before the first run) is reported in one line naming the cause and the
  cache location, instead of a traceback.
- `--tier` is validated as a choice, and `--k`, `--top`, and `--chunk-size`
  reject values below one, before any work starts.
- `search` says which sources it skipped for lacking vectors at the tier.
- Informational notes go to stderr, so piped output holds only results.
- The web UI memoises search results on the query and the library's
  modification time, so repeating a search is instant.
- Searching, embedding, or listing terms with `--source <title>` now selects
  every file that shares that title (a book added as one PDF per chapter),
  instead of failing as ambiguous. `remove` and `edit` still require a single
  match. The web UI's "Search in" filter names each file and filters by id.
- The web UI escapes source text, titles, and authors before rendering, so a
  `<` or `&` on the page is shown as written rather than read as HTML.
- Error messages escape filenames and user input, so a path like
  `[draft].pdf` prints as typed rather than being read as console markup.
- The web UI no longer re-runs the last search every time a widget changes;
  results are kept until "Search" is clicked again.
- Removing a source in the web UI now asks for confirmation.

## [0.1.0] - 2026-10-05

The one-day build, plus key terms and continuous integration.

### Added

- `dsearch add`, `list`, `edit`, `remove`, `search`, `embed`, and `terms`.
- Per-page extraction with printed folios and chapter titles (`extract`),
  page-bounded sentence chunks (`chunk`), a hash-keyed library with per-tier
  vectors (`index`), hybrid dense + BM25 retrieval fused by reciprocal rank
  (`search`), MLA citations (`cite`), per-chapter key terms (`terms`).
- A Streamlit web UI over the same functions.
- GitHub Actions running ruff and pytest on pushes and pull requests.

[Unreleased]: https://github.com/73anthonyL/DeterministicSearch/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/73anthonyL/DeterministicSearch/releases/tag/v0.1.0
