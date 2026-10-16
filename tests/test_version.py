"""The version is declared once, in pyproject.toml, and read from package metadata."""

import tomllib
from pathlib import Path

import dsearch


def test_version_matches_pyproject():
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert dsearch.__version__ == pyproject["project"]["version"]
