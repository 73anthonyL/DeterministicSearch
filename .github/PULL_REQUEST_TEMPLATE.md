## Summary

<!-- What changes, and why. One paragraph is usually enough. -->

## Type

<!-- The title is squash-merged as the commit, so it must be a conventional commit:
     feat(scope): ... | fix(scope): ... | docs: ... | test: ... | refactor: ... | build: ... | ci: ... | chore: ... -->

## Checklist

- [ ] Tests added or updated, and `uv run pytest` passes
- [ ] `uv run ruff check .`, `uv run ruff format --check .`, and `uv run pyright` are clean
- [ ] `CHANGELOG.md` has a line under `Unreleased` for anything a user would notice
- [ ] `DECISIONS.md` records any choice the code cannot explain by itself
- [ ] No PDFs, indexes, or anything from `~/.dsearch` are included
- [ ] For UI changes: a screenshot
