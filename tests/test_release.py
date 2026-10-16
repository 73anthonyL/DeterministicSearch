"""The release pipeline's moving parts must agree with each other.

A publish that reaches the index with the wrong name or from the wrong
environment is not caught by any other test, so the workflow, the trusted
publisher configuration it implies, and pyproject.toml are checked together.
"""

import re
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
DISTRIBUTION = "deterministic-search"


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load((ROOT / ".github" / "workflows" / "release.yml").read_text())


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_distribution_name_is_the_published_one(pyproject, workflow):
    assert pyproject["project"]["name"] == DISTRIBUTION
    assert workflow["env"]["PACKAGE"] == DISTRIBUTION
    for job in ("testpypi", "pypi"):
        assert workflow["jobs"][job]["environment"]["url"].endswith(f"/p/{DISTRIBUTION}")


def test_command_and_import_are_still_dsearch(pyproject):
    assert pyproject["project"]["scripts"] == {"dsearch": "dsearch.cli:main"}
    assert (ROOT / "src" / "dsearch" / "__init__.py").is_file()


def test_publishing_jobs_use_their_named_environment_and_oidc_only(workflow):
    """Trusted publishing is keyed on the environment name; a token would be a secret."""
    for job in ("testpypi", "pypi"):
        spec = workflow["jobs"][job]
        assert spec["environment"]["name"] == job
        assert spec["permissions"] == {"id-token": "write"}
        publish = [s for s in spec["steps"] if "pypa/gh-action-pypi-publish" in s.get("uses", "")]
        assert len(publish) == 1
        assert "password" not in publish[0].get("with", {})
    # The token is read-only by default; only the jobs above and the release
    # step widen it.
    assert workflow["permissions"] == {"contents": "read"}


def test_pypi_only_after_testpypi_and_only_from_a_tag(workflow):
    assert "testpypi" in workflow["jobs"]["pypi"]["needs"]
    assert workflow["jobs"]["pypi"]["if"] == "github.event_name == 'push'"
    triggers = workflow.get("on", workflow.get(True))  # YAML 1.1 reads a bare `on` as a boolean
    assert triggers["push"]["tags"] == ["v*"]
    assert triggers["workflow_dispatch"]["inputs"]["target"]["options"] == ["testpypi"]


def test_every_action_is_pinned_to_a_commit(workflow):
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            if uses := step.get("uses"):
                assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", uses), uses


def test_cliff_config_parses_and_skips_release_commits():
    config = tomllib.loads((ROOT / "cliff.toml").read_text())
    skipped = [p["message"] for p in config["git"]["commit_parsers"] if p.get("skip")]
    assert any("release" in pattern for pattern in skipped)
    assert config["git"]["tag_pattern"] == "v[0-9].*"
