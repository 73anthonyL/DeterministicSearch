"""Logging is off by default and switched on by DSEARCH_LOG or -v."""

from __future__ import annotations

import logging

from rich.console import Console

from dsearch import log


class TestLevel:
    def test_default_is_warning(self, monkeypatch):
        monkeypatch.delenv(log.ENV_VAR, raising=False)
        assert log.level_from_env() == logging.WARNING

    def test_env_var_sets_level(self, monkeypatch):
        monkeypatch.setenv(log.ENV_VAR, "debug")
        assert log.level_from_env() == logging.DEBUG

    def test_explicit_beats_env(self, monkeypatch):
        monkeypatch.setenv(log.ENV_VAR, "DEBUG")
        assert log.level_from_env("ERROR") == logging.ERROR

    def test_unknown_name_falls_back(self, monkeypatch):
        monkeypatch.setenv(log.ENV_VAR, "LOUD")
        assert log.level_from_env() == logging.WARNING


class TestConfigure:
    def test_installs_one_handler_even_when_called_twice(self):
        logger = logging.getLogger("dsearch")
        log.configure("DEBUG", console=Console(file=open("/dev/null", "w"), force_terminal=False))  # noqa: SIM115
        log.configure("DEBUG", console=Console(file=open("/dev/null", "w"), force_terminal=False))  # noqa: SIM115
        ours = [h for h in logger.handlers if getattr(h, "_dsearch_handler", False)]
        assert len(ours) == 1
        assert logger.level == logging.DEBUG

    def test_debug_lines_reach_the_console(self):
        from io import StringIO

        sink = StringIO()
        log.configure("DEBUG", console=Console(file=sink, force_terminal=False, width=200))
        logging.getLogger("dsearch.index").debug("loaded fast model in 0.1s")
        assert "loaded fast model" in sink.getvalue()

    def test_warning_level_hides_debug(self):
        from io import StringIO

        sink = StringIO()
        log.configure("WARNING", console=Console(file=sink, force_terminal=False))
        logging.getLogger("dsearch.index").debug("hidden")
        assert sink.getvalue() == ""
