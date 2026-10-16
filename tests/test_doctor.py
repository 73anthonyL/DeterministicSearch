"""`dsearch doctor` reports facts; these check the facts are computed right."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dsearch import doctor, index

if TYPE_CHECKING:
    from pathlib import Path


class TestModelCache:
    def test_bare_names_resolve_under_sentence_transformers(self):
        assert doctor.model_cache_folder("all-MiniLM-L6-v2") == (
            "models--sentence-transformers--all-MiniLM-L6-v2"
        )

    def test_qualified_names_keep_their_organisation(self):
        assert (
            doctor.model_cache_folder("BAAI/bge-base-en-v1.5") == "models--BAAI--bge-base-en-v1.5"
        )

    def test_cached_models_checks_the_folders(self, tmp_path: Path):
        (tmp_path / doctor.model_cache_folder(index.TIERS["fast"])).mkdir()
        cached = doctor.cached_models(tmp_path)
        assert cached == {"fast": True, "balanced": False, "best": False}


class TestSizes:
    def test_directory_size_sums_files(self, tmp_path: Path):
        (tmp_path / "a").write_bytes(b"x" * 10)
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b").write_bytes(b"y" * 5)
        assert doctor.directory_size(tmp_path) == 15

    def test_missing_directory_is_zero(self, tmp_path: Path):
        assert doctor.directory_size(tmp_path / "nope") == 0

    def test_format_bytes(self):
        assert doctor.format_bytes(512) == "512 B"
        assert doctor.format_bytes(2048) == "2.0 KB"
        assert doctor.format_bytes(3 * 1024**2) == "3.0 MB"
        assert doctor.format_bytes(5 * 1024**3) == "5.0 GB"


class TestDiagnostics:
    def test_reports_an_empty_library_without_creating_it(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("DSEARCH_HOME", str(tmp_path / "lib"))
        info = doctor.diagnostics(probe_device=False)
        assert info["library"]["exists"] is False
        assert info["library"]["sources"] == 0
        assert not (tmp_path / "lib").exists()
        assert set(info["models"]["tiers"]) == set(index.TIERS)
        assert info["models"]["device"] == "not probed"
        assert info["packages"]["pymupdf"]

    def test_offline_flag_reads_the_environment(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("DSEARCH_HOME", str(tmp_path))
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        assert doctor.diagnostics(probe_device=False)["models"]["offline"] is True

    def test_device_is_one_of_the_known_names(self):
        assert doctor.compute_device().split(" ")[0] in {"cuda", "mps", "cpu", "unavailable"}
