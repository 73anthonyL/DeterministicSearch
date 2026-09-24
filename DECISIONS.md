# Decisions

Choices the spec left open, and the reasoning behind each, so they can be
defended or reversed on purpose.

## Packaging and environment

- **Build backend: hatchling.** The spec allowed hatchling or setuptools.
  Hatchling needs no `setup.py`/`MANIFEST.in` and reads the `src/` layout from
  three lines of config.
- **Python 3.12 for the development venv.** The floor is 3.11 as specified. The
  venv pins 3.12 because `torch` (pulled in by `sentence-transformers`) has no
  wheels for 3.14, which is the default `python3` on this machine.
- **`import pymupdf`, not `import fitz`.** PyMuPDF now emits a deprecation
  warning for the `fitz` alias.

## Extraction

- **Folios are searched in a header/footer *band*, not just the first and last
  line.** In the sample chapters the running head and the page number are set on
  the same band, so the folio is extracted as the second line
  (`"i n t r o d u c t i o n"`, then `"3"`). Scanning one line found 15 of 29
  folios in chapter 1; scanning a 3-line band finds 29 of 29. On pages too short
  to have a real band (<= 6 lines) only the outermost lines are trusted, so a
  stray integer in body text is not mistaken for a page number.
- **The footer band is checked before the header band.** The bottom of the page
  is the more common home for a folio; in a book that prints both, they agree.
- **Roman-numeral folios resolve to `None`.** Front-matter numbers are real, but
  mixing `p. xiv` and `p. 14` in one library would make citations ambiguous.
  Front matter is rarely the evidence a student quotes.
- **A page needs 20 characters to count as having a text layer.** Scanned pages
  are seldom exactly empty — running heads and artifacts leak a few characters
  through. The scan threshold itself (>80% of pages) is from the spec.
- **Ligature glyphs are expanded during extraction.** The sample PDFs encode
  "fi" and "fl" as private-use codepoints (U+F0DE, U+F0DF), so without this the
  tool retrieves and quotes "I will pay the  ne" where the page plainly reads
  "I will pay the fine" — it would be showing the reader something the source
  does not say. The extractor also emits a spurious space after the glyph
  ("of" + fi + " ce"), which is dropped when a letter follows.
  The U+FBxx Unicode ligature block is always safe to expand. The two
  private-use slots carry no Unicode meaning and were read off this typesetting,
  where they reconstruct "five", "office", "influence" and "rifles" across all
  400+ occurrences; another publisher's font could use those slots differently,
  so no other private-use character is guessed at — unmapped ones are left
  visible so the problem is seen rather than silently papered over.
  This lives in extraction, not chunking, because it is *decoding* the page
  correctly rather than editing it.
- **Chapter titles: PDF outline first, font size second.** The outline is
  authoritative when present. The font heuristic calls a line a heading when it
  is >= 1.25x the modal span size, is <= 90 characters, and is not itself a page
  number. Both approaches carry the last-seen title forward across pages.

## Chunking

- **Chunks never span a page break.** The product promise is a passage a reader
  can cite to an exact page, so windowing restarts on each page. The costs are a
  short chunk at the foot of every page and a sentence broken by a page break
  being chunked as two fragments. Letting windows cross pages would produce
  slightly better passages but would make `p. 47` a guess.
- **`chunk_id` is a 0-based integer index within its source, and doubles as the
  row index into `vectors_<tier>.npy`.** A chunk is identified globally by
  (`source_id`, `chunk_id`). This keeps the vector/metadata join implicit, which
  the planned 2D embedding map depends on.
- **`chunk()` takes a keyword-only `source_id`.** The spec's signature omits it
  but `Chunk` carries it, so it has to be threaded in; keyword-only keeps the
  documented positional signature `chunk(pages, size, overlap)` intact.
- **Text is cleaned before splitting: de-hyphenation, running-head removal, bare
  folio removal, and reflow.** PyMuPDF preserves the typesetter's hard line
  breaks, so without this a quoted sentence reads
  `"my compan- ions"` or carries `i n t r o d u c t i o n 3` spliced onto its
  front. `Page.text` itself stays a faithful copy of the PDF; the cleaning lives
  in `chunk` so that extraction never invents or drops content.
- **Running heads are detected by repetition across pages, at a 30% threshold.**
  Books alternate the head between verso and recto, so each variant appears on
  only ~48% of pages (measured in the samples); the next most repeated line
  appears on ~5%. 30% separates them with room to spare. Detection is skipped
  for sources under 4 pages, where repetition proves nothing.
- **Sentence splitting is a regex with an abbreviation list, per the spec's "no
  NLTK download".** Known limitation, covered by a test: a sentence that
  genuinely *ends* in an abbreviation ("He worked for the co. They left.") is
  joined to the next one. Over-joining is safer than over-splitting here,
  because a split mid-sentence would truncate quoted evidence.
- **Roman-numeral and bare-integer lines are dropped as folios during cleaning**,
  matching the extraction rule.

## Library and indexing

- **`meta.json` carries more than the five fields the spec names.** Alongside
  author, title, filename, added date, and page count it stores `source_id`,
  `chunk_count`, `chunk_size`, `overlap`, and the list of embedded `tiers`.
  `list` needs the counts, search needs to know which tiers exist, and recording
  the chunking parameters means a stored index can be interpreted later without
  guessing what settings produced it.
- **Re-adding a file at a *new* tier reuses the stored chunks and embeds only
  the missing vectors.** The spec says an identical hash is skipped silently; it
  is read here as skipping redundant *work*, not as refusing to upgrade quality.
  Re-adding at a tier that is already present remains a true no-op.
- **Embeddings are L2-normalised at write time**, so cosine similarity at query
  time is a plain dot product and needs no per-query renormalisation.
- **`sentence_transformers` is imported lazily inside `load_model`.** It pulls in
  torch, which costs seconds of startup. `dsearch --help`, `list`, and `remove`
  must not pay that.
- **`library.json` and `meta.json` are written to a temp file and renamed.** An
  add interrupted with Ctrl-C leaves the previous library intact rather than a
  half-written file. A corrupt `library.json` is treated as empty rather than
  crashing every command.
- **`remove` accepts an id prefix, a filename, a filename stem, or a title**, and
  raises on an ambiguous match instead of guessing — it is about to delete
  something.
- **The >10 source warning is attached to the `AddResult`**, not printed from the
  library layer, so the CLI and the Streamlit app each render it their own way.

## Search

- **The two rankers treat a zero score differently, and this is deliberate.** A
  BM25 score of zero means not one query term appears in the chunk — no evidence
  — so that chunk earns no rank and contributes nothing to the fusion. A low
  cosine is still a meaningful ordering, so the dense ranker ranks every chunk.
  An earlier version let any rank-1 chunk through regardless of score, which let
  a zero-scoring chunk tie and then beat a genuine keyword-only match.
- **RRF constant 60 and a 200-candidate pool.** 60 is the value from the
  original RRF paper. The pool is much wider than any realistic `k` so that a
  passage found by only one of the two rankers still reaches the fusion.
- **Fusion is over ranks, not scores**, because an unbounded BM25 score and a
  cosine in [-1, 1] cannot be added meaningfully.
- **`Result` also carries `meta`**, the source's `SourceMeta`. The spec lists
  chunk, score, best-sentence index, and neighbours; citation and display both
  need the title and author, and threading the metadata through avoids a library
  lookup per rendered result.
- **The best-matching sentence is computed only for the k returned chunks**, by
  embedding their sentences at query time. Storing per-sentence vectors for the
  whole library would multiply index size several-fold to save milliseconds on a
  handful of results.
- **A long query's chunk score is the max over its sentence embeddings, not the
  mean.** Averaging a 100-word paragraph dilutes the one sentence that matters;
  max makes the paragraph behave like a set of separate questions, which is what
  a student pasting their own thinking actually wants.
- **A source with no vectors at the requested tier is skipped and named in
  `Corpus.skipped`, not fatal.** One book not yet upgraded to `best` should not
  block searching the rest of the library.
- **Context neighbours stop at a source boundary**, so the passage above a
  result is never from a different book.

## Citation

- **The spec's citation line is produced exactly**
  (`Holmes, Seth. *Fresh Fruit, Broken Bodies*. p. 47.`), with the title
  italicised using markdown asterisks so `rich` and Streamlit both render it.
- **An in-text parenthetical (`(Holmes 47)`) is offered alongside the full
  reference.** The full line belongs in a works-cited list; the parenthetical is
  what actually goes next to the quotation in the body of an essay, and it is
  the thing a student pastes most often.
- **MLA 9 terminal punctuation is handled per kind.** A final period is dropped
  from the quotation and reappears after the parenthetical; a question mark or
  exclamation point is part of the quoted words and stays inside the quotation
  marks, with a period after the parenthetical.
- **Unknown elements are omitted, never invented.** A source with no author
  begins its citation with the title; a page with no printed folio cites
  `PDF p. 59`, explicitly labelled so it is not mistaken for a printed page.
- **Three or more authors collapse to "et al."**, per MLA 9; two are joined with
  "and" and only the first is inverted.

## CLI

- **Errors exit 1 with a one-line message, never a traceback.** A missing file,
  a scanned PDF, an unknown tier, or an ambiguous source name are all ordinary
  user mistakes and are reported as such, on stderr.
- **Neighbouring-chunk context is truncated to 180 characters.** The full
  neighbour is available on the `Result`; at `--k 20` printing all of it buries
  the matches. The chunk's own sentences are never truncated — that is the
  evidence.
- **Progress is reported in pages, with an "reading pages…" state** before the
  page count is known (extraction finishes after the bar is already drawn).
- **`--version` is a top-level eager option.** Not in the spec, but it is the
  first thing anyone cloning the repo checks.
- **The in-text parenthetical is shown beside the full MLA line** in each panel
  footer, since that is the form that goes into the essay body.

## Testing

- **Test PDFs are generated, never real books**, per the spec. `insert_text`
  does not wrap, and anything past the right margin is silently clipped out of
  the text layer, so fixture prose is hand-wrapped at 88 columns. Without this
  every fixture page lost its tail and the fixtures quietly misrepresented what
  a real page looks like.
- **Embeddings are stubbed with a deterministic bag-of-words vector** in the
  index, search, and CLI tests, so the suite runs offline in seconds and ranking
  assertions are stable. The real models are exercised in the end-to-end run.

## Streamlit app

- **`app.py` contains no retrieval logic.** It calls exactly the functions the
  CLI calls, so the two front ends can never drift apart in ranking.
- **The entry point is guarded by `if __name__ == "__main__"`.** Streamlit runs
  a script with `__name__ == "__main__"`, so this changes nothing about
  `streamlit run` while making the module importable for unit tests.
- **The upload is written to a temp file and deleted in a `finally`.** Indexing
  needs a path for pymupdf; the library keys on the SHA-256 of the bytes, so the
  temp path is irrelevant to caching and re-uploading the same file is free.
- **Citations are shown through `st.code`,** which renders a one-click copy
  button — the "copyable citation block" the spec asks for.
- **Library management (tier, source filter, add, remove) lives in the sidebar**
  so the main column is only query and evidence.
- **Every Streamlit widget carries an explicit `key`.** Without one, Streamlit
  identifies a widget by its position in the tree, and this sidebar changes
  shape the moment the library stops being empty (the "Search in" selector
  appears). Keys make the author, title, and tier survive that reshuffle.
- **An upload is written into a temp *directory* under its own filename**, not
  to a generated temp name. The library records `path.name`, so the alternative
  stored `tmpjzua04hj.pdf` — meaningless in `dsearch list` and unusable as an
  argument to `dsearch remove`.

## Patch round 1

### Extraction artifacts

- **Superscript spans are dropped by MuPDF's own flag, then a regex catches
  the rest.** The flag is authoritative when present (every endnote marker in
  the sample chapters carries it); the regex is the belt to its braces for PDFs
  whose generator did not mark them. The regex requires a *lowercase* letter
  before the digits so that "B12", "F16" and "COVID19" survive, and requires
  whitespace or end of line after them so that "the 1990s" does. The one known
  false positive is a tight "p.35" with no space, which typeset books avoid.
  Counts are logged at DEBUG per page rather than printed, so an audit is
  `DSEARCH_LOG=DEBUG`-style opt-in and a clean add stays quiet.
- **Page text is rebuilt from spans in the same block/line layout as
  `get_text("text")`.** Verified byte-identical on all 43 pages of the sample
  chapter when no spans are dropped, so folio and running-head detection see
  exactly what they saw before.
- **Letter-spacing collapses only for runs of three or more single characters.**
  "F I V E" and "T H R E E" are typography; "A" and "I" in a title such as
  "A Day I Remember" are words, and a two-token run cannot be told apart from
  one. A single-letter word that directly follows a spaced number ("F I V E A
  Day") would be swallowed into it — accepted as rare.
- **Endnote and title cleaning live in `extract`, not `chunk`.** The earlier
  rule was that `Page.text` is a faithful copy of the PDF. A superscript marker
  is not part of the sentence on the page any more than the running head is, so
  removing it is still decoding, not editing — the same argument as ligatures.
- **`index_version` is an integer, with its history in a comment next to the
  constant.** A source written before versioning has no field and reads as 0,
  which is older than every real version, so it is rebuilt exactly once.
- **The PDF's absolute path is stored in `meta.json` so a stale source can be
  rebuilt from `search`, where no path is given.** The alternative — copying
  the PDF into `~/.dsearch` — would double the disk footprint of every book for
  a rebuild that happens once per format bump. If the file has moved (or was a
  Streamlit upload, whose temp file is gone), the search says so and runs on
  the old index rather than failing; `dsearch add` on the file fixes it.
- **A rebuild re-embeds every tier the source already had, not only the one
  requested.** Old vectors index old chunks, so leaving them would silently
  corrupt searches at the other tiers; deleting them would silently take away
  quality the user had paid for. Re-embedding is the only option that is
  neither.
- **Author and title survive a rebuild.** They may have been set by hand and
  are metadata about the book, not about the text pipeline.

### Citation fallback

- **The "Chapter5. p. 153." line was the filename fallback, not the chapter
  field.** `cite.py` never read `chunk.chapter`; `Chapter5.pdf` carried no PDF
  metadata, so its stem became the title. The fix is threefold: the fallback
  now produces "Chapter 5" (readable, and visibly a placeholder), `dsearch edit`
  sets the real author and title, and a test asserts the chapter can never
  appear in a citation.
- **Resolution lives in `cite.py` and is applied once, at add time.** `index`
  calls `resolve_author`/`resolve_title` when a source is created, and
  `meta.json` stores the outcome. Re-resolving on every search would mean
  re-opening the PDF for its metadata at query time; storing the result means
  `edit` is a plain overwrite. `cite` imports `index`/`search` types only
  under `TYPE_CHECKING` so that `index` can import `cite` without a cycle.
- **No filename fallback for the author.** A filename is at best a title.
  Putting "Chapter 5" in the author slot of a works-cited entry would be worse
  than leaving it blank, which MLA permits.
- **Filename title-casing raises only the first letter of each word** and
  splits `_`, `-`, and letter/digit seams. `str.title()` would produce "Don'T"
  and lowercase an acronym.
- **`citation_block` is now reference first, quotation second, and the
  quotation keeps its in-text parenthetical.** The spec fixes the order; the
  parenthetical is retained because it is the form that goes into the essay
  body and dropping it would remove the most-pasted line. The CLI footer
  prints the same two lines.
- **`edit` refuses a call with neither flag** rather than silently succeeding,
  so a mistyped `--tittle` is caught.
- **`edit` keeps the source's position in `dsearch list`.** `save_meta`
  replaces the row in place; a rename should not look like a re-add.
