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

Python 3.11, 3.12, or 3.13. Until the first PyPI release, install from the
repository; the `ui` extra adds the web interface:

```bash
pipx install "dsearch[ui] @ git+https://github.com/73anthonyL/DeterministicSearch"
# or:  uv tool install "dsearch[ui] @ git+https://github.com/73anthonyL/DeterministicSearch"
```

From a clone, for development:

```bash
git clone https://github.com/73anthonyL/DeterministicSearch.git && cd DeterministicSearch
uv sync --all-extras          # creates .venv with the exact locked versions
uv run dsearch --help         # or: source .venv/bin/activate
```

The first `dsearch add` downloads the embedding model (about 90 MB for the
default `fast` tier); everything after that is offline.

## Try it in two minutes

A public-domain book ships inside the package: the Forethought and first three
chapters of W. E. B. Du Bois, *The Souls of Black Folk* (1903), typeset with
front matter, an outline, running heads, and printed page numbers like a real
book. `dsearch demo` indexes it and runs one search:

```
$ dsearch demo

Sample: dsearch-demo/sample.pdf (The Souls of Black Folk, Du Bois, W. E. B.)
The first run downloads the embedding model (about 90 MB).

Indexed The Souls of Black Folk (c9dfcfb6) — 31 pages, 259 chunks, tier fast.

Searching for: "What does it feel like to be seen as a problem?"

╭─ 1. The Souls of Black Folk  ·  I. Of Our Spiritual Strivings  ·  p. 1 (PDF 4) ──────────────────╮
│                                                                                                  │
│  All, nevertheless, flutter round it. They approach me in a half-hesitant sort of way, eye me    │
│  curiously or compassionately, and then, instead of saying directly, How does it feel t…         │
│  At these I smile, or am interested, or reduce the boiling to a simmer, as the occasion may      │
│  require. To the real question, How does it feel to be a problem? I answer seldom a word.        │
│  I answer seldom a word. And yet, being a problem is a strange experience,—peculiar even for     │
│  one who has never been anything else, save perhaps in babyhood and in Europe. It is in…         │
│                                                                                                  │
│  Du Bois, W. E. B. *The Souls of Black Folk*. p. 1.                                              │
│  “To the real question, How does it feel to be a problem?” (Du Bois 1).                          │
│                                                                                                  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
…
Next: ask your own question with dsearch search "...", add a book with dsearch add your.pdf
--author ... --title ..., or open the web UI with dsearch ui.
```

Then ask it something of your own:

```
$ dsearch search "Booker T. Washington's programme of industrial education" -k 1

╭─ 1. The Souls of Black Folk  ·  III. Of Mr. Booker T. Washington and Others  ·  p. 20 (PDF 23) ──╮
│                                                                                                  │
│  Easily the most striking thing in the history of the American Negro since 1876 is the           │
│  ascendancy of Mr. Booker T. Washington. It began at the time when war memories and ideals wer…  │
│  Mr. Washington came, with a simple definite programme, at the psychological moment when the     │
│  nation was a little ashamed of having bestowed so much sentiment on Negroes, and was            │
│  concentrating its energies on Dollars. His programme of industrial education, conciliation of   │
│  the South, and submission and silence as to civil and political rights, was not wholly          │
│  original; …                                                                                     │
│                                                                                                  │
│  Du Bois, W. E. B. *The Souls of Black Folk*. p. 20.                                             │
│  “Mr. Washington came, with a simple definite programme, at the psychological moment when the    │
│  nation was a little ashamed of having bestowed so much sentiment on Negroes, and was            │
│  concentrating its energies on Dollars” (Du Bois 20).                                            │
│                                                                                                  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

The page numbers are the ones printed on the page (`p. 20`), not the PDF's
own count (`PDF 23`); the front matter has roman numerals and is cited by PDF
page. The provenance of the sample is in `src/dsearch/data/README.md`.

Installing with plain `pip` on Linux pulls the default `torch` wheel, which
bundles CUDA libraries this tool never uses. To keep it small:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "dsearch[ui] @ git+https://github.com/73anthonyL/DeterministicSearch"
```

`uv` does this automatically from the project's configuration.

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

### `terms` — what each chapter is about

The words and phrases a chapter uses far more than the rest of its book, each
with how often it occurs and the page where it is densest — a place to start
reading, or a query to hand to `search`:

```
$ dsearch terms --top 4

Fresh Fruit, Broken Bodies — Seth Holmes  (re-ranked at tier fast)
Chapter                             Term                   Count  Densest page
ONE Introduction                    coyote                    29  p. 14 (PDF 14)
                                    border                    75  p. 9 (PDF 9)
                                    Border Patrol             20  p. 22 (PDF 22)
                                    bus                       29  p. 2 (PDF 2)

TWO “We Are Field Workers”          embodied anthropology      5  p. 30 (PDF 1)
                                    body                      31  p. 34 (PDF 5)
                                    social                    28  p. 31 (PDF 2)
                                    experiences               21  p. 34 (PDF 5)

THREE Segregation on the Farm       farm                     145  p. 66 (PDF 22)
                                    pickers                   86  p. 73 (PDF 29)
                                    berries                   63  p. 69 (PDF 25)
                                    strawberry                39  p. 59 (PDF 15)

FOUR “How the Poor Suffer”          violence                  56  p. 109 (PDF 22)
                                    headaches                 20  p. 98 (PDF 11)
                                    pain                      30  p. 93 (PDF 6)
                                    suffering                 24  p. 95 (PDF 8)

FIVE “Doctors Don’t Know Anything”  patients                 114  p. 115 (PDF 5)
                                    physician                 87  p. 123 (PDF 13)
                                    medical                   67  p. 116 (PDF 6)
                                    clinicians                34  p. 141 (PDF 31)
```

Sources that share a title and author are treated as one book, so a book added
as one PDF per chapter is compared chapter against chapter. If a chapter lands
in a book of its own, its title or author differs from its siblings';
`dsearch edit` fixes that.

| Option | Effect |
|---|---|
| `--top 10` | Terms per chapter. |
| `--source <id>` | Show only that source's chapters. They are still compared with the rest of the book. |
| `--tier fast` | Which stored vectors re-rank the terms. A tier the book lacks is never embedded for this; the terms are ranked by word counts and a note says so. |
| `--no-rerank` | Rank by word counts only. Instant, and loads no model. |

```
$ dsearch terms --source Chapter5.pdf --top 3 --no-rerank

Fresh Fruit, Broken Bodies — Seth Holmes  (ranked by word counts)
Chapter                             Term       Count  Densest page
FIVE “Doctors Don’t Know Anything”  patients     114  p. 115 (PDF 5)
                                    physician     87  p. 123 (PDF 13)
                                    clinic        83  p. 127 (PDF 17)
```

### `doctor` — what is installed, where

```
$ dsearch doctor

dsearch      0.2.0
python       3.12.4  (/Users/you/.venv/bin/python)
platform     macOS-15.0-arm64-arm-64bit
library      /Users/you/.dsearch  —  5 source(s), 12.4 MB
model cache  /Users/you/.cache/huggingface/hub
  fast       all-MiniLM-L6-v2  —  cached
  balanced   all-mpnet-base-v2  —  cached
  best       BAAI/bge-base-en-v1.5  —  not downloaded
offline      no
device       mps
pymupdf      1.28.2
…
```

Paste it into a bug report. `--json` gives the same as data.

### Machine-readable output and logging

`list`, `search`, and `terms` take `--json`. A search result carries the
passage, its page, its neighbours, and the MLA, in-text, and quoted-line
citations:

```bash
dsearch search "housing in the camps" --json | jq '.[0].citation'
```

Informational notes go to stderr, so stdout is always just the data.
`dsearch -v <command>` (or `DSEARCH_LOG=DEBUG`) logs model loads, corpus
sizes, and embedding throughput; `-q` drops the notes.

### `remove` — drop a source

```
$ dsearch remove ea63ed8d

Removed Chapter 3: Segregation on the Farm (ea63ed8d).
```

Accepts an id prefix, a filename, or a title.

---

## The web UI

```bash
dsearch ui              # needs the `ui` extra; opens a browser tab
dsearch ui --headless --port 8600
```

Drag one or more PDFs onto the sidebar, pick a quality tier, and watch the
"book 2 of 5 — page 10 of 44" counter during indexing. A library table above
the query box lets you edit author and title in place (the same code path as
`dsearch edit`). Searching at a tier some sources lack shows the cost and an
"Embed now" button rather than starting a long job silently; the sidebar lists
which tiers each source already has. The query box takes paragraphs. Each
result is the same card you get in the terminal, with a one-click copyable
citation block. "Key terms by chapter", above the query box, shows the same
table as `dsearch terms` for every book in the library. The app calls exactly
the same functions the CLI does, so the two can never disagree about ranking.

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

6. **Terms** (`terms.py`) — reads the stored chunks and vectors; nothing is
   re-indexed. Words and two-word phrases are counted from each sentence once
   (chunks overlap, so counting chunk text would count shared sentences twice).
   A candidate is scored by its rate in the chapter weighted by how many times
   higher that is than its rate in the rest of the book. The best 50 are then
   embedded and ordered by similarity to the direction from the book's average
   vector to the chapter's, and the two orderings are fused by reciprocal rank,
   as in search. A term is always a span of the source, never a generated label.

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
uv sync --all-extras       # dev tools come from the `dev` dependency group
uv run pytest              # 445 tests, no network required
uv run ruff check .
uv run ruff format --check .
```

`uv.lock` is committed so every clone and CI run resolves the same versions;
`uv sync --locked` fails rather than silently updating it.

The test suite generates its own tiny PDFs and stubs out the embedding model, so
it runs offline in a couple of seconds.

---

## Roadmap

- ~~**Per-chapter key-term extraction**~~ — shipped as `dsearch terms`.
- **A 2D embedding map** — project the stored chunk vectors down to two
  dimensions and colour them by chapter, to see how a book's arguments cluster.
  Each cluster can be labelled with its chapter's key terms.

The map reads the existing `chunks.jsonl` and `vectors_<tier>.npy`; the `Chunk`
schema carries `chapter` and a `chunk_id` that doubles as the vector row index
specifically so it needs no re-index, as key terms did not.

---

## Design notes

Every decision the spec did not settle is written down, with its reasoning, in
[DECISIONS.md](DECISIONS.md).

## License

MIT — see [LICENSE](LICENSE).
