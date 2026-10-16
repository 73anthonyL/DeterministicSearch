# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

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
