# dsearch

**Local-first semantic evidence search for academic PDFs.**

Add a book, ask a question in plain language, and get back real page-cited
passages with an MLA citation you can paste into an essay.

`dsearch` never generates text. Every word it shows you was already in your PDF.
It retrieves spans, highlights the sentence that actually matched, and tells you
the page it came from. That is the whole point: **evidence that cannot be
fabricated.**

Everything runs on your machine. The embedding model is downloaded once and
cached; after that, searching needs no network at all.

---

## Install

Three commands, from a clean clone:

```bash
git clone https://github.com/73anthonyL/DeterministicSearch.git && cd DeterministicSearch
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Python 3.11 or newer. The first `dsearch add` downloads the embedding model
(about 90 MB for the default `fast` tier); everything after that is offline.

```
$ dsearch --help

 Usage: dsearch [OPTIONS] COMMAND [ARGS]...

 Local-first semantic evidence search for academic PDFs. Retrieval only.

╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ add     Index a PDF into your library.                                       │
│ list    Show every source in your library.                                   │
│ remove  Delete a source from your library.                                   │
│ search  Find page-cited passages that answer a query.                        │
╰──────────────────────────────────────────────────────────────────────────────╯
```

---

## Usage

### `add` — index one or more PDFs

```
$ dsearch add samples/Chapter3.pdf --author "Seth Holmes" \
    --title "Fresh Fruit, Broken Bodies"

Indexed Fresh Fruit, Broken Bodies (ea63ed8d) — 43 pages, 407 chunks, tier fast.
```

A progress bar counts through the pages while they are embedded. Sources are
keyed by the SHA-256 of the file's bytes, so adding the same PDF twice — even
under a different name — costs nothing:

```
$ dsearch add samples/Chapter3.pdf

Already indexed Fresh Fruit, Broken Bodies (ea63ed8d) at tier fast — skipped.
```

Several files, or a glob, go in one call. They are processed in order with a
"book 2 of 5" bar above the page bar; anything already indexed is skipped with
a note. With more than one file the author and title come from each PDF's own
metadata (or its filename) — fix them afterwards with `edit`:

```
$ dsearch add samples/Chapter1.pdf samples/Chapter2.pdf

Indexed Chapter 1 (eb815263) — 29 pages, 249 chunks, tier fast.
Indexed Chapter 2 (7b4b8faf) — 15 pages, 107 chunks, tier fast.
```

Options: `--author`, `--title` (single file only), `--tier fast|balanced|best`,
`--chunk-size`, `--overlap`.

### `edit` — fix the citation metadata

```
$ dsearch edit eb815263 --author "Seth Holmes" --title "Fresh Fruit, Broken Bodies"

Updated (eb815263) — Fresh Fruit, Broken Bodies by Seth Holmes.
```

Rewrites `meta.json` only; chunks and vectors are untouched. Citations resolve
author and title from the `add`/`edit` flags, then the PDF's metadata, then a
title-cased filename (`Chapter5.pdf` → "Chapter 5") — never the chapter.

### `list` — see your library

```
$ dsearch list

ID        Title                       Author       Pages  Chunks  Tiers           Added
eb815263  Fresh Fruit, Broken Bodies  Seth Holmes     29     249  fast, balanced  2026-09-24
7b4b8faf  Fresh Fruit, Broken Bodies  Seth Holmes     15     107  fast, balanced  2026-09-24
ea63ed8d  Fresh Fruit, Broken Bodies  Seth Holmes     43     407  fast            2026-09-24

3 source(s) in /Users/anthony/.dsearch
```

The Tiers column is read from the vector files on disk, so it always says
which tiers a search can use without waiting.

### `search` — find evidence

Context is dimmed, the sentence that actually matched is bold, and the footer is
a citation you can copy.

```
$ dsearch search "Why do migrant workers cross the border despite the danger?" --k 1

╭─ 1. Chapter 1: Introduction  ·  O N E Introduction  ·  p. 25 (PDF 25) ───────────────────────────╮
│                                                                                                  │
│  I wonder how I will pay the fine. I wonder how my Triqui friends are doing and how it would     │
│  feel to know you had to attempt the long trek again. “ i s i t w o r t h r i s k i n g…         │
│  In much of the mainstream media, migrant workers are seen as deserving their fates, even        │
│  untimely deaths, because they are understood to have chosen voluntarily to cross the border     │
│  for their own economic gain. However, as pointed out above, my Triqui companions explain that   │
│  they are forced to cross the border. In addition, the distinction between economic and          │
│  political migration is often blurry in the context of international policies enforcing          │
│  neoliberal free markets as well as active military repression of indigenous people who seek     │
│  collective socioeconomic improvement in southern Mexico.                                        │
│  In addition, the distinction between economic and political migration is often blurry in the    │
│  context of international policies enforcing neoliberal free markets as well as active…          │
│                                                                                                  │
│  Holmes, Seth. *Chapter 1: Introduction*. p. 25.  in-text: (Holmes 25)                           │
│                                                                                                  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

A bare proper noun works too — this is where the keyword half of the search earns
its keep:

```
$ dsearch search "Triqui" --k 1

╭─ 1. Chapter 3: Segregation on the Farm  ·  T H R E E Segregation on the Farm  ·  p. 75 (PDF 31) ─╮
│                                                                                                  │
│  I wanted to come here to make money, but no. I don’t even make enough to send to Oaxaca to my   │
│  mom who is taking care of my son. Sometimes the strawberry goes poorly, your back hur…          │
│  I am sorry; I don’t speak Spanish well. Pure Triqui. [Chuckles] Pure Triqui.                    │
│  [Chuckles] Pure Triqui. It’s very difficult here. The farm camp manager doesn’t want to give a  │
│  room to a single woman.                                                                         │
│                                                                                                  │
│  Holmes, Seth. *Chapter 3: Segregation on the Farm*. p. 75.  in-text: (Holmes 75)                │
│                                                                                                  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

So does a whole paragraph of your own half-formed thinking. Paste the messy
version — a long query is split into sentences and scored one at a time, so the
one idea that matters is not averaged away:

```
$ dsearch search "I keep coming back to the idea that the suffering of the farmworkers in
this book is not accidental or natural but is produced and then made to look normal. The
bodies that break down are the bodies at the bottom of the ethnic hierarchy on the farm,
and the people at the top explain that away by saying the workers are simply suited to
that kind of labor. I want to understand how the book connects the physical pain of
picking, the housing conditions in the camps, and the medical system that fails to
recognize any of it as an injury caused by work rather than by the worker himself." --k 1

╭─ 1. Chapter 2: We Are Field Workers  ·  T W O “We Are Field Workers”  ·  p. 31 (PDF 2) ──────────╮
│                                                                                                  │
│  Ultimately, I hope that my field research and writing will work toward ameliorating the social  │
│  suffering inherent to migrant labor in North America. Broadly, this book explores et…           │
│  The exploration begins by uncovering the structure of farm labor, describing how agricultural   │
│  work in the United States is segregated according to an ethnicity-citizenship hierarchy. The    │
│  book then shows ethnographically that this pecking order produces correlated suffering and      │
│  illness, particularly among undocumented, indigenous Mexican pickers. Yet it becomes clear      │
│  that this injurious hierarchy is neither willed nor planned by the farm executives and          │
│  managers; rather, it is produced by larger social structures.                                   │
│  Yet it becomes clear that this injurious hierarchy is neither willed nor planned by the farm    │
│  executives and managers; rather, it is produced by larger social structures. Of note,…          │
│                                                                                                  │
│  Holmes, Seth. *Chapter 2: We Are Field Workers*. p. 31.  in-text: (Holmes 31)                   │
│                                                                                                  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

`--k 20` is the "search deeper" path. `--source` limits a search to one text:

```bash
dsearch search "housing in the camps" --k 20 --source "Chapter 3: Segregation on the Farm"
```

### `embed` — pre-warm a quality tier

Chunks do not depend on the tier; only `vectors_<tier>.npy` does. Searching at
a tier some sources lack never re-extracts or re-chunks — it offers to embed
just the missing vectors, after saying what that costs:

```
$ dsearch search "Triqui" --k 1 --tier balanced

Note 5 of 5 source(s) have no balanced vectors: Fresh Fruit, Broken Bodies (29 pages), … — 154 pages in all.
       Embedding them would take about 4 min (default estimate until measured) and only adds
balanced vectors — nothing else is touched.
Embed balanced vectors for 5 source(s) now? (or run with `--tier fast`, which all sources already have) [y/N]: n
Nothing embedded; search not run.
```

`--yes` skips the question. The estimate uses a per-tier rate measured on the
first real run and kept in `~/.dsearch/config.json`; until then it is a
deliberately pessimistic default. To do the work ahead of time:

```
$ dsearch embed --tier balanced

Embedded Fresh Fruit, Broken Bodies (eb815263) at tier balanced.
…
```

`--source` limits it (repeatable).

### `remove` — drop a source

```
$ dsearch remove ea63ed8d

Removed Chapter 3: Segregation on the Farm (ea63ed8d).
```

Accepts an id prefix, a filename, or a title.

---

## The web UI

```bash
streamlit run src/dsearch/app.py
```

Drag one or more PDFs onto the sidebar, pick a quality tier, and watch the
"book 2 of 5 — page 10 of 44" counter during indexing. A library table above
the query box lets you edit author and title in place (the same code path as
`dsearch edit`). Searching at a tier some sources lack shows the cost and an
"Embed now" button rather than starting a long job silently; the sidebar lists
which tiers each source already has. The query box takes paragraphs. Each
result is the same card you get in the terminal, with a one-click copyable
citation block. It calls exactly the same functions the CLI does, so the two
can never disagree about ranking.

---

## How it works

1. **Extract** (`extract.py`) — one `Page` per PDF page via pymupdf. The printed
   folio is read out of the header/footer band, so citations use the number
   printed on the paper rather than the PDF's own page count. Chapter titles come
   from the PDF outline, or failing that from font-size heading detection.
   Ligature glyphs are expanded, so a page reading "pay the fine" is never quoted
   as "pay the  ne", and superscript endnote markers are dropped, so
   "competency.35" is quoted as "competency." A PDF with no text layer raises a
   clear error telling you to OCR it.

2. **Chunk** (`chunk.py`) — text is de-hyphenated, running heads are dropped, and
   prose is split into sentences with a regex. Sentences are then windowed into
   overlapping chunks (3 sentences, 1 shared, by default). **Chunks never cross a
   page break**, so every passage cites one exact page.

3. **Index** (`index.py`) — each source lives in `~/.dsearch/sources/<sha256>/`
   as `meta.json`, `chunks.jsonl`, and one `vectors_<tier>.npy` per quality tier.
   Row *i* of the vector matrix is chunk *i*. Set `DSEARCH_HOME` to move the
   library.

4. **Search** (`search.py`) — two rankers run over the same chunks. A dense
   ranker embeds the query and takes cosine similarity; BM25 scores literal
   keyword overlap. Their **rankings** are fused with reciprocal rank fusion,
   because an unbounded BM25 score and a cosine in [-1, 1] cannot be added
   meaningfully. This is why both "Triqui" and a vague paragraph work: keywords
   catch the proper noun, embeddings catch the paraphrase. A query over ~200
   words is embedded sentence by sentence, and each chunk keeps its *best*
   sentence match rather than an average.

5. **Cite** (`cite.py`) — an MLA reference and an in-text parenthetical, built
   only from what is actually known. Missing elements are dropped, never invented.

### Quality tiers

| Tier | Model | Trade-off |
|---|---|---|
| `fast` (default) | `all-MiniLM-L6-v2` | Quickest, smallest download |
| `balanced` | `all-mpnet-base-v2` | Better recall, slower |
| `best` | `BAAI/bge-base-en-v1.5` | Highest quality, slowest |

Switching tiers reuses the stored chunks and embeds only the missing vectors,
so upgrading quality never re-extracts the PDF and never removes another
tier's vectors.

### Index versions

`meta.json` records the `index_version` that built each source. When a
release changes extraction or chunking (v2 drops superscript endnote markers
and cleans letter-spaced chapter titles), the next `add` or `search` rebuilds
any older source from the path it was added from, saying so in one line. If
the PDF has moved, the search says that instead and `dsearch add <pdf>`
rebuilds it.

---

## Development

```bash
pip install -e ".[dev]"
pytest          # 335 tests, no network required
ruff check .
```

The test suite generates its own tiny PDFs and stubs out the embedding model, so
it runs offline in a couple of seconds.

---

## Roadmap

- **Per-chapter key-term extraction** — surface the terms that distinguish each
  chapter, reading the stored chunk metadata directly.
- **A 2D embedding map** — project the stored chunk vectors down to two
  dimensions and colour them by chapter, to see how a book's arguments cluster.

Both read the existing `chunks.jsonl` and `vectors_<tier>.npy`; the `Chunk`
schema carries `chapter` and a `chunk_id` that doubles as the vector row index
specifically so neither feature needs a re-index.

---

## Design notes

Every decision the spec did not settle is written down, with its reasoning, in
[DECISIONS.md](DECISIONS.md).

## License

MIT — see [LICENSE](LICENSE).
