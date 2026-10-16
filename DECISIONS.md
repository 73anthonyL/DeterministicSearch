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
  opt-in (`DSEARCH_LOG=DEBUG` or `dsearch -v`, wired up in v0.2) and a clean
  add stays quiet.
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

### Multi-PDF add

- **Globs are expanded by dsearch as well as by the shell.** zsh expands
  `samples/*.pdf` before the program sees it, but a quoted pattern, and every
  Windows shell, hands it through verbatim. A pattern that matches nothing is
  an error, not a silent no-op, because a typo in a path should be visible.
- **One bad file does not stop the batch.** A scanned PDF among five is
  reported on stderr and the other four are indexed; the exit code is 1 at the
  end so a script can tell. Stopping at the first failure would make the user
  re-run the command minus one file, which is exactly the tedium the batch is
  meant to remove.
- **`--author`/`--title` with several files is refused, not applied to all.**
  Five chapters of one book would benefit from applying it to all, but five
  different books would be silently mislabelled, and the refusal message names
  `dsearch edit` as the fix.
- **The Streamlit progress bar is one bar, not two.** Streamlit stacks widgets
  vertically and a second bar during a five-book upload would push the page
  around; the single bar's fraction is `(books done + this book's fraction) /
  books`, with the "Book 2 of 5 — page 10 of 44" text carrying the detail.
- **The library table is `st.data_editor` fed a list of dicts, not a
  DataFrame,** so the app has no pandas import of its own. Edits arrive as
  `{row: {column: value}}` in session state and are pushed through
  `index.edit` — the same function `dsearch edit` calls — so the two front
  ends cannot drift.
- **The table sits in a collapsed expander above the query.** It is
  reference material, not the task; results stay the first thing on screen.
- **Author and title inputs disappear when more than one file is selected**,
  rather than being disabled with a tooltip, because a disabled field that
  still shows a typed value invites the belief that it will be applied.

### Lazy per-tier embedding

- **The tier check reads vector files from disk, not `meta.tiers`.** A file
  that is missing is a gap whatever the metadata says; `dsearch list` uses the
  same on-disk answer so the two can never disagree.
- **The throughput rate is measured inside `embed_texts`, after the model is
  loaded.** The first run of a tier includes a model download that can dwarf
  the embedding itself; starting the clock after `load_model` keeps that out
  of the number. Runs under 5 pages are not recorded because a tiny file
  gives a noisy rate, and the spec's "first run" is honoured literally: a
  recorded rate is never overwritten.
- **Default rates are pessimistic** (3 / 0.6 / 0.5 pages per second for fast /
  balanced / best). An estimate that runs long is a pleasant surprise; one
  that runs short is a broken promise. The prompt says which kind it is.
- **Declining the prompt exits 0 and runs no search.** Saying no is a choice,
  not an error, and searching the subset that *does* have the tier would
  quietly answer a different question than the one asked. Non-interactive
  stdin (a pipe, a script) counts as no, so nothing long ever starts
  unattended.
- **The alternative named in the prompt is the first tier every in-scope
  source has, in tier order.** Naming all of them would make the prompt a
  paragraph; the first is enough to get a result without waiting.
- **In Streamlit, the search that triggered the warning is kept in session
  state.** Clicking "Embed now" reruns the script, which would otherwise
  forget the query; the pending search resumes once the tier is present, and
  stays on screen through later table edits.
- **`dsearch embed` runs the stale-index upgrade first**, like `search`. A
  pre-warm that embedded old chunks would have to be redone on the next
  search.
- **`--source` on `embed` is repeatable** (`--source a --source b`) rather than
  comma-separated, which is Typer's convention and copes with titles that
  contain commas.

## Key terms

Decided with the author before building: key terms before the embedding map;
lexical candidates re-ranked by embeddings; chapters compared within sources
that share a title and author; each term shown with its count and densest
page. What follows is what that left open.

### Scoring

- **The lexical score is not textbook TF-IDF.** That was the plan and the first
  thing tried. With five chapters nearly every word occurs in all of them, so
  IDF is close to constant and the ranking collapses to frequency: "people",
  "work", and "Triqui" led every chapter. The score used is the term's rate in
  the chapter times the log of the ratio to its rate in the rest of the book
  (the term's contribution to the KL divergence between the two). On the
  sample book it puts "checkers", "crew", and "berries" at the top of chapter
  3. A term no more frequent than elsewhere scores zero and is never shown.
- **The re-rank compares candidates with the chapter's centroid *minus the
  book's*, not the centroid itself.** Chapters of one book share most of their
  meaning, so similarity to the plain centroid measured how typical a term is
  of the book and promoted "migrant" and "immigrants" everywhere. The
  difference vector points at what is particular to the chapter. A one-chapter
  book has no difference to take and uses the plain centroid.
- **The two rankings are fused by reciprocal rank**, with the function search
  already uses (`_rrf`, made public as `rrf`). A weighted sum would need a
  weight, and a weight needs a justification.
- **A term must occur 3 times in a chapter to be a candidate**, and the best 50
  candidates go to the re-rank. Below 3, one sentence can make a word look
  maximally distinctive.
- **A one-chapter book is ranked by frequency, and the output says so.** With
  nothing to contrast against there is no distinctiveness to measure; silence
  would imply there was.

### Counting

- **Words are counted from each sentence once.** Chunks overlap by
  `meta.overlap` sentences within a page, so the repeated leading sentences of
  every chunk after the first on a page are skipped. Counting `chunk.text`
  would inflate any term in a shared sentence.
- **Phrases are two words and never cross punctuation.** Sentences are split
  at commas, dashes, quotes, and brackets before pairing words, so "workers,
  farm" is not a phrase. Three-word phrases were left out: in the sample book
  they were nearly all a two-word phrase plus a neighbour.
- **Regular plurals are counted with their singular** when the book uses both
  ("patient"/"patients", "berry"/"berries"). Nothing else is stemmed; a
  stemmer would also merge words that are not the same. Known limitation:
  "-es" plurals ("bosses") and irregular ones are not folded.
- **A term is shown in its most common spelling in the book,** so "Macario"
  and "Border Patrol" keep their capitals. Phrases are tallied as written
  rather than word by word, which would have printed "border Patrol".
- **Stopwords are a built-in list of about 170 words.** It only has to keep
  function words out of the candidates; the scoring already demotes words
  common to every chapter. No dependency was added.

### Pruning

- **A word is dropped for a phrase when 60% of its occurrences are inside that
  phrase** ("Miguel" for "San Miguel"), or when a phrase ranked above it
  contains it. "border" survives beside "Border Patrol" because most of its
  uses are on its own.
- **One word appears in at most two of a chapter's terms.** Without the cap,
  chapter 2 listed "embodied", "embodied anthropology", and "embodied
  experiences".

### Grouping and scope

- **Book matching is exact apart from case and surrounding space.** A typo in
  one source's title splits it into its own book. Fuzzy matching could merge
  two editions by mistake; `dsearch edit` is the fix, and the README says so.
- **Pages with no chapter title are grouped by source, under the filename.**
  Pooling every untitled page in a book would merge unrelated front matter.
- **`--source` narrows what is shown, not what is compared.** The rest of the
  book is still read, because a chapter's terms are only distinctive relative
  to its siblings.
- **`terms` never embeds.** A book missing the requested tier is ranked
  lexically with a note naming `dsearch embed`. Search asks before embedding
  because it cannot answer without vectors; terms can, so it does not ask.
- **`terms` does not rebuild a stale index either**, unlike `search` and
  `embed`: a rebuild re-embeds. It prints a note naming the source instead.
- **Nothing is cached.** Counting the sample library takes 0.2 s. The re-rank
  takes longer, almost all of it loading the model, which a cache of terms
  would not avoid on the first run and `--no-rerank` avoids on every run.
- **`Term.score` is the reciprocal of the final rank**, so scores from
  lexical-only and re-ranked runs are on the same scale.

### Front ends

- **The CLI prints one table per book with the chapter named on its first
  row,** rather than a panel per chapter, so five chapters fit on one screen.
- **In Streamlit the terms are computed on a button press** and kept in
  session state with the tier that produced them. Computing on every rerun
  would load the model each time a slider moved; keeping the tier means
  switching tiers hides a ranking that no longer matches.
- **The Streamlit section ignores the sidebar's source filter** and shows every
  book. The filter passes a *title*, which is ambiguous for a book split over
  several PDFs.

### Tests

- **The stub embedders in the CLI and index tests hash tokens with
  `zlib.crc32`, not `hash()`.** The built-in is salted per process, so about
  one run in thirty ranked a different page first and
  `TestEdit::test_updates_the_citation` failed. Found while running the suite
  for this feature; fixed in its own commit.


## Continuous integration

- **One GitHub Actions job: `ruff check`, `ruff format --check`, `pytest`**, on
  pushes to `main` and on every pull request. The suite stubs the embedder and
  generates its own PDFs, so the job downloads no models and needs no secrets.
- **CI runs Python 3.11, the supported floor.** The development venv is 3.12,
  so the floor is the version nothing else exercises.
- **CPU-only `torch` is installed before the package.** On Linux the default
  wheel bundles CUDA libraries the tests never use; the CPU index keeps the
  install small.

## Source filters and rendering (v0.2 fixes)

- **A title shared by several sources names the whole book for `search`,
  `embed`, and `terms`; `remove` and `edit` still demand one match.** The
  README tells readers to add a book as one PDF per chapter under one title,
  and the first thing they then do is filter a search to that title — which
  raised "matches several sources". `resolve_sources` returns every match when
  all share a title and stays ambiguous otherwise; `resolve_source` is
  unchanged for the two commands that are about to delete or rewrite
  something.
- **The web UI's source filter lists "title — filename" and returns the
  source id.** Two chapters with the same title were indistinguishable in the
  dropdown, and the title it returned was ambiguous. The id resolves exactly.
- **Source text is HTML-escaped in the app and markup-escaped in CLI errors.**
  A passage is shown through `unsafe_allow_html` so that it can be dimmed and
  bolded; without escaping, a `<` on the page became markup and the quote no
  longer matched the source. Rich reads `[...]` as style tags, so a filename
  or user input printed raw could lose characters.
- **The app keeps the last results in session state and searches only on a
  click.** Streamlit reruns the script on every widget change; the pending
  request survived reruns (so "Embed now" could resume it) but so did the
  search itself, which reloaded the corpus and re-embedded on each slider
  move. A new click drops the cached results.
- **The app's ✕ opens a confirmation popover.** Removing an index is cheap to
  redo but slow to rebuild, and the button sat beside the library list.
- **The Streamlit script is tested with `AppTest`**, not only its helpers. The
  source-filter bug lived in widget wiring that no helper test could reach.

## Packaging (v0.2)

- **The version lives once, in `pyproject.toml`, and `__version__` reads it
  from package metadata.** The alternative, `hatch-vcs` deriving it from the
  git tag, needs full history in every checkout and gives dev builds names
  like `0.2.0.dev3+g1a2b3c`. A static field means `uv version --bump` edits
  one line and the release workflow can assert that the tag matches it. A
  checkout that was never installed reports `0+unknown` rather than lying.
- **Streamlit is an extra (`dsearch[ui]`), not a core dependency.** It brings
  pandas and pyarrow, which the CLI never imports; a reader who only wants
  `dsearch search` should not install a data-frame stack for it.
- **`dsearch ui` launches Streamlit in a subprocess on the packaged `app.py`.**
  `streamlit run src/dsearch/app.py` only works from a clone; a wheel has no
  `src/`. The script is located with `importlib.resources`, so the command
  works from any install. A subprocess rather than Streamlit's bootstrap
  function: that function is private, its signature has changed between
  releases, and a child process keeps Ctrl-C and the event loop Streamlit's.
- **uv, with `uv.lock` committed and a PEP 735 `dev` dependency group.** The
  lock pins what CI and every clone install, so a failing test cannot be a
  version drift; the published wheel is unaffected, since metadata carries
  only the floors. A dependency *group* rather than a `dev` extra keeps
  development tools out of the package's published metadata.
- **CPU-only torch on Linux via `[tool.uv.sources]`.** The default Linux wheel
  bundles CUDA libraries; the CPU index cuts a fresh install by gigabytes and
  replaces the hand-written `pip install torch --index-url` step in CI. macOS
  and Windows wheels on PyPI are already CPU-only, so the source is scoped to
  Linux with a marker.
- **No upper bounds on runtime dependencies.** Upper bounds on a library make
  it the thing that breaks every other resolver; reproducibility comes from
  the lockfile instead. A bound is added only after a real break, with its
  reason here.
- **Python 3.11 to 3.13 are declared; 3.14 is not yet.** The lock resolves
  for 3.14, but nothing exercises it until the CI matrix does.

## Quality gates and CI (v0.2)

- **pyright, not mypy.** It is what the editor already runs (Pylance), so a
  CI failure is something the author saw while typing; it infers more without
  annotations; and it is markedly faster. The configuration concepts carry
  over one for one. `standard` mode now; `strict` can be switched on
  module by module.
- **Ruff rule families added: RUF, PT, N, C4, PIE, RET, TC, S, D.** `D` uses
  the Google convention but does not require a docstring to exist (`D1xx`
  ignored): the code already documents what matters, and a forced docstring
  on every helper is noise. `ANN` is left out because pyright covers
  annotations with better messages. Typographic quotes and dashes are
  allowed as "confusables" because the extractor's whole job is to handle
  them. Tests skip `S101` (assert) and `D`.
- **Coverage floor of 90%, two points under the first measurement.** A floor
  catches a change that silently drops a module from the suite; it is not a
  target. It is raised when coverage rises.
- **Three OSes times three Pythons in the test matrix, `fail-fast: false`.**
  Windows and macOS are where path handling and console encoding break, and
  nothing else exercises the 3.11 floor. All cells run so one failure shows
  its real extent. `PYTHONUTF8=1` on every cell: Rich draws box characters
  that the default Windows code page cannot encode.
- **Separate `lint`, `typecheck`, `test`, and `build` jobs.** A ruff failure
  should not wait for nine test runs, and the ruleset names the jobs it
  requires; one combined job would make every failure look the same.
- **The `build` job installs the wheel into an empty environment and runs
  it.** The suite runs from the checkout, so a file missing from the package
  (`app.py`, `py.typed`) would pass every test and fail every user.
- **Actions pinned to commit SHAs with the version in a comment.** A tag can
  be moved; a SHA cannot. Dependabot bumps the SHA and the comment together.
- **Least-privilege tokens and cancelled superseded runs.** `permissions:
  contents: read` on every workflow that needs nothing more; `concurrency`
  cancels the run a new push makes obsolete.
- **Pull request titles are checked for conventional-commit form.** With
  squash merges the title *is* the commit on `main`, and the changelog is
  generated from those commits.
- **Dependabot groups minor and patch bumps into one weekly PR** so updates
  arrive as one review, not fifteen.
- **`.editorconfig` and a pre-commit hook set** mirror CI locally. The
  large-file hook (1 MB) is the mechanism behind "no copyrighted PDFs"; the
  `uv-lock` hook keeps `uv.lock` in step so `--locked` in CI never fails on
  drift.

## CLI hardening (v0.2)

- **Notes go to stderr; results go to stdout.** `--json` has to be pipeable,
  and a "Note: skipped X" line in the middle of a JSON array is not. Errors
  were already on stderr; moving the notes makes the rule simple: stdout is
  data, stderr is commentary.
- **`--json` emits with the standard library, not Rich.** Rich wraps long
  lines to the terminal width; a wrapped JSON document is a broken one.
- **A search result's JSON includes all three citation forms.** A script
  that consumes results almost always wants the citation too, and composing
  it from `cite` is the CLI's job, not the caller's.
- **`Tier` is a `StrEnum` so Typer validates and completes it.** A string
  checked at runtime produced a custom error after the command had started;
  an enum produces Click's usage error before it does. The pipeline keeps
  taking plain strings (`tier.value`), so nothing below the CLI changed.
- **`ModelUnavailableError` wraps `OSError` from `SentenceTransformer`.** The
  hub's offline and connection failures are `OSError` subclasses; the message
  names the likely cause (no network, or `HF_HUB_OFFLINE` before the first
  run) and points at `dsearch doctor` for the cache path. One line, exit 1.
- **`doctor` probes the device by importing torch, nothing else does.** The
  rest of `doctor` reads metadata and directories; the import is the slow
  line and is wrapped so a broken binary is reported, not fatal.
- **Logging is a Rich handler on the `dsearch` logger only**, installed by
  the CLI callback or `DSEARCH_LOG`; library code logs and never configures.
  Reconfiguring replaces the handler, so tests and the app cannot stack
  duplicates.
- **The web UI memoises `search` with `st.cache_data`, keyed on the query
  and the library file's mtime.** The session-state cache from v0.2's bug
  fixes stops reruns from re-searching; this stops a *repeated* search from
  recomputing. The model itself is already cached per process in
  `index._MODEL_CACHE`, so no `st.cache_resource` wrapper is needed.
