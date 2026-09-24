"""Tests for the Streamlit wrapper's pure formatting helpers.

The Streamlit widgets themselves are exercised by actually running the app in
the end-to-end verification; what is unit-tested here is the passage markup,
which is where a bug would silently misrepresent the evidence.
"""

from __future__ import annotations

from dsearch import app
from dsearch.chunk import Chunk
from dsearch.index import AddResult, SourceMeta
from dsearch.search import Result

SENTENCES = [
    "The clinic dismissed his knee pain.",
    "No one examined the knee.",
    "He went back to the rows.",
]


def make_result(best: int = 1, before: str | None = None, after: str | None = None) -> Result:
    def make_chunk(chunk_id: int, sentences: list[str]) -> Chunk:
        return Chunk(
            chunk_id=chunk_id,
            source_id="abc",
            text=" ".join(sentences),
            sentences=sentences,
            pdf_page=59,
            printed_page=47,
            chapter="Chapter 3",
        )

    meta = SourceMeta(
        source_id="abc",
        filename="ffbb.pdf",
        title="Fresh Fruit, Broken Bodies",
        author="Seth Holmes",
        added="2026-09-24",
        page_count=100,
    )
    return Result(
        chunk=make_chunk(1, SENTENCES),
        score=0.5,
        best_sentence=best,
        meta=meta,
        before=make_chunk(0, [before]) if before else None,
        after=make_chunk(2, [after]) if after else None,
    )


class TestTruncate:
    def test_leaves_short_text_alone(self):
        assert app._truncate("short", 20) == "short"

    def test_adds_an_ellipsis_when_cutting(self):
        out = app._truncate("x" * 100, 20)
        assert len(out) == 20
        assert out.endswith("…")


class TestPassageHtml:
    def test_bolds_only_the_matching_sentence(self):
        html = app._passage_html(make_result(best=1))
        assert "<strong>No one examined the knee.</strong>" in html
        assert "<strong>The clinic dismissed his knee pain.</strong>" not in html

    def test_dims_the_other_sentences(self):
        html = app._passage_html(make_result(best=1))
        assert html.count("color:") >= 2

    def test_includes_context_when_present(self):
        html = app._passage_html(make_result(before="Before text.", after="After text."))
        assert "Before text." in html
        assert "After text." in html

    def test_omits_context_when_absent(self):
        html = app._passage_html(make_result())
        assert "Before text." not in html

    def test_every_sentence_appears_verbatim(self):
        # The passage shown must be exactly the source's sentences.
        html = app._passage_html(make_result())
        for sentence in SENTENCES:
            assert sentence in html


class TestModuleShape:
    def test_importing_does_not_run_the_app(self):
        # Guarded by __name__ == "__main__", so importing is side-effect free.
        assert callable(app.main)

    def test_tier_notes_cover_every_tier(self):
        from dsearch.index import TIERS

        assert set(app.TIER_NOTES) == set(TIERS)


class TestUploadNaming:
    """An upload must keep its own filename in the library."""

    def test_temp_path_uses_the_upload_name(self, tmp_path, monkeypatch):
        # The library records `path.name`, so writing the upload under a
        # generated temp name would store "tmpjzua04hj.pdf" and make the source
        # unrecognisable in `dsearch list` and unusable with `dsearch remove`.
        import dsearch.app as app_module

        captured: dict[str, object] = {}

        def fake_add(path, **kwargs):
            captured["name"] = path.name
            captured["exists"] = path.is_file()
            raise ValueError("stop here — naming is all this test checks")

        monkeypatch.setattr(app_module, "add", fake_add)
        monkeypatch.setattr(app_module.st, "error", lambda *a, **kw: None)

        assert app_module._index_one(FakeUpload("Chapter2.pdf"), "", "", "fast", 3, 1, None) is None
        assert captured["name"] == "Chapter2.pdf"
        assert captured["exists"] is True


class TestMultiUpload:
    def test_indexes_each_upload_in_order_with_combined_progress(self, monkeypatch):
        import dsearch.app as app_module

        seen: list[str] = []
        texts: list[str] = []

        def fake_add(path, *, progress, **kwargs):
            seen.append(path.name)
            progress(2, 4)  # Halfway through this book.
            meta = SourceMeta("id", path.name, path.stem, "", "2026-09-24", 4, chunk_count=9)
            return AddResult(meta=meta, already_indexed=False, embedded_tier="fast")

        class Bar:
            def progress(self, fraction, text=""):
                texts.append(f"{fraction:.3f} {text}")

            def empty(self):
                pass

        monkeypatch.setattr(app_module, "add", fake_add)
        monkeypatch.setattr(app_module.st, "progress", lambda *a, **kw: Bar())
        monkeypatch.setattr(app_module.st, "success", lambda *a, **kw: None)
        monkeypatch.setattr(app_module.st, "info", lambda *a, **kw: None)

        uploads = [FakeUpload("a.pdf"), FakeUpload("b.pdf"), FakeUpload("c.pdf")]
        app_module._index_uploads(uploads, "", "", "fast", 3, 1)

        assert seen == ["a.pdf", "b.pdf", "c.pdf"]
        # Book 2 of 3, half done, is 1.5/3 of the way overall.
        assert "0.500 Book 2 of 3 — embedding page 2 of 4" in texts

    def test_author_and_title_are_only_passed_for_a_single_upload(self, monkeypatch):
        # The sidebar blanks them for several files; a single file still gets them.
        import dsearch.app as app_module

        kwargs_seen: list[dict] = []

        def fake_add(path, **kwargs):
            kwargs_seen.append(kwargs)
            raise ValueError("stop")

        monkeypatch.setattr(app_module, "add", fake_add)
        monkeypatch.setattr(app_module.st, "error", lambda *a, **kw: None)
        app_module._index_one(FakeUpload("a.pdf"), "Seth Holmes", "", "fast", 3, 1, None)
        assert kwargs_seen[0]["author"] == "Seth Holmes"
        assert kwargs_seen[0]["title"] is None  # Falls through to PDF metadata, then filename.


class TestLibraryTable:
    def _sources(self):
        return [
            SourceMeta("a" * 64, "a.pdf", "Chapter 4", "", "2026-09-24", 23, tiers=["fast"]),
            SourceMeta("b" * 64, "b.pdf", "Chapter 5", "", "2026-09-24", 44, tiers=["fast"]),
        ]

    def test_rows_show_title_author_pages_and_tiers(self):
        rows = app._library_rows(self._sources())
        assert rows[1] == {
            "Title": "Chapter 5",
            "Author": "",
            "Pages": 44,
            "Tiers": "fast",
            "ID": "bbbbbbbb",
        }

    def test_edits_go_through_the_same_edit_function_as_the_cli(self, monkeypatch):
        import dsearch.app as app_module

        calls: list[tuple] = []
        monkeypatch.setattr(
            app_module,
            "edit",
            lambda source_id, *, author, title: calls.append((source_id, author, title)),
        )
        ids = [m.source_id for m in self._sources()]
        app_module._apply_table_edits(
            {1: {"Author": "Seth Holmes", "Title": "Fresh Fruit, Broken Bodies"}, 0: {"Pages": 9}},
            ids,
        )
        assert calls == [("b" * 64, "Seth Holmes", "Fresh Fruit, Broken Bodies")]


class FakeUpload:
    def __init__(self, name: str):
        self.name = name

    def getbuffer(self):
        return b"%PDF-1.4 fake"


class TestTierGapBox:
    def _gap(self, alternatives):
        from dsearch.index import TierGap

        missing = [
            SourceMeta("a" * 64, "a.pdf", "Chapter 4", "", "2026-09-24", 23),
            SourceMeta("b" * 64, "b.pdf", "Chapter 5", "", "2026-09-24", 44),
        ]
        return TierGap(
            tier="balanced",
            missing=missing,
            pages=67,
            seconds=120.0,
            rate_measured=False,
            alternatives=alternatives,
        )

    def test_names_sources_pages_estimate_and_alternative(self):
        text = app._gap_message(self._gap(["fast"]), total=5)
        assert "2 of 5 source(s) have no *balanced* vectors" in text
        assert "**Chapter 4** (23 pages)" in text and "**Chapter 5** (44 pages)" in text
        assert "67 pages in all" in text
        assert "about 2 min" in text
        assert "default estimate" in text
        assert "*fast*, which every source already has" in text

    def test_no_alternative_when_none_is_shared(self):
        assert "already has" not in app._gap_message(self._gap([]), total=2)

    def test_search_waits_until_embed_now_is_clicked(self, monkeypatch):
        import dsearch.app as app_module

        embedded: list[str] = []
        monkeypatch.setattr(app_module, "list_sources", lambda: self._gap([]).missing)
        monkeypatch.setattr(app_module, "tier_gap", lambda tier, scoped: self._gap(["fast"]))
        monkeypatch.setattr(app_module.st, "warning", lambda *a, **kw: None)
        monkeypatch.setattr(app_module, "_embed_missing", lambda gap: embedded.append(gap.tier))

        monkeypatch.setattr(app_module.st, "button", lambda *a, **kw: False)
        assert app_module._ensure_tier("balanced", None) is False
        assert embedded == []

        monkeypatch.setattr(app_module.st, "button", lambda *a, **kw: True)
        assert app_module._ensure_tier("balanced", None) is True
        assert embedded == ["balanced"]

    def test_tiers_summary_lists_each_source(self, monkeypatch):
        import dsearch.app as app_module

        monkeypatch.setattr(app_module, "available_tiers", lambda sid: ["fast", "best"])
        text = app_module._tiers_summary(self._gap([]).missing)
        assert "**Chapter 4**: fast, best" in text
        assert "**Chapter 5**: fast, best" in text
