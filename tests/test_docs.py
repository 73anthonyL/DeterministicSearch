"""The README must describe the CLI that exists."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from dsearch.cli import app

README = Path(__file__).resolve().parents[1] / "README.md"


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


def registered_commands() -> list[str]:
    """Command names as Typer registers them: explicit, or derived from the function."""
    names = []
    for cmd in app.registered_commands:
        callback = cmd.callback
        derived = callback.__name__.replace("_", "-") if callback else ""
        names.append(cmd.name or derived)
    return names


class TestReadme:
    def test_every_command_is_documented(self, readme: str):
        commands = registered_commands()
        assert len(commands) >= 10
        help_block = readme[readme.index("$ dsearch --help") :]
        help_block = help_block[: help_block.index("```")]
        for name in commands:
            pattern = rf"^│ {name}\b"
            assert re.search(pattern, help_block, re.MULTILINE), f"{name} missing from --help"
        for name in commands:
            if name in {"ui", "demo", "doctor"}:
                continue  # covered by their own sections under other headings
            assert f"### `{name}`" in readme, f"no usage section for {name}"

    def test_no_personal_paths(self, readme: str):
        assert "/Users/" not in readme

    def test_links_to_the_docs_that_exist(self, readme: str):
        root = README.parent
        pattern = r"\]\((docs/[^)]+|CONTRIBUTING\.md|DECISIONS\.md|CHANGELOG\.md)\)"
        for link in re.findall(pattern, readme):
            assert (root / link).is_file(), link
