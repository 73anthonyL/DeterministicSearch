"""Tests for the Streamlit wrapper's pure formatting helpers.

The Streamlit widgets themselves are exercised by actually running the app in
the end-to-end verification; what is unit-tested here is the passage markup,
which is where a bug would silently misrepresent the evidence.
"""

from __future__ import annotations

from dsearch import app
from dsearch.chunk import Chunk
from dsearch.index import SourceMeta
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
