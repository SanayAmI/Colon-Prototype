# PMEST-Net: Facet-Disentangled Metadata Vector Space Architecture

An end-to-end framework applying Dr. S. R. Ranganathan's **PMEST** library
classification theory to neural metadata extraction, producing keyword
embeddings that are disentangled by facet instead of blended into one
opaque vector.

> **Status: MVP / zero-shot prototype.** Stages 1–4 of the architecture
> are implemented and runnable end to end. Stage 5 (supervised training)
> is not implemented — see [Status](#status-whats-real-vs-placeholder)
> for the exact boundary between what's learned, what's zero-shot, and
> what's a deterministic placeholder.

---

## Table of Contents

- [Why this exists](#why-this-exists)
- [The PMEST facets](#the-pmest-facets)
- [How it works](#how-it-works)
- [Status: what's real vs. placeholder](#status-whats-real-vs-placeholder)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [How to access / run it](#how-to-access--run-it)
  - [Try it interactively](#try-it-interactively)
  - [Classify a CSV of your own](#classify-a-csv-of-your-own)
  - [Run the pipeline stage-by-stage](#run-the-pipeline-stage-by-stage)
- [Known limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Citation & license](#citation--license)

---

## Why this exists

Standard embedding models (Sentence-BERT, OpenAI's text-embedding
models, etc.) map every piece of text into one monolithic vector space.
That causes **attribute entanglement**: a document's dominant topic
drowns out orthogonal attributes that are just as important for
retrieval — *when* something happened, *where* it happened, *what
process* it describes, versus *what it's fundamentally about*.

PMEST-Net instead classifies each extracted keyword into one of five
independent facets, then places it in a **disentangled vector space**
where each facet gets its own dedicated sub-region. A search for "trade
routes in the 1300s" can then weight the Time subspace heavily without
that similarity leaking into or being drowned out by the Personality
subspace — and vice versa.

## The PMEST facets

| Facet | Symbol | Definition | Example |
|---|---|---|---|
| **Personality** | P | Core subject, central entity, or main concept under investigation | *Graph Neural Networks*, *Trade routes* |
| **Matter** | M | Physical substances, materials, chemicals, hardware, datasets | *Graphene oxide*, *ImageNet* |
| **Energy** | E | Actions, methods, algorithms, processes, transformations | *Classification*, *Optimization*, *Synthesis* |
| **Space** | S | Geographic, spatial, or architectural locations | *Sahara Desert*, *European Union* |
| **Time** | T | Dates, eras, epochs, intervals, frequencies | *2008 financial crisis*, *the 1300s* |

A single word can land in different facets depending on context — the
architecture's running example is "Glass," which is **Matter** as a raw
material, **Personality** as the subject of an optics paper, or
**Energy** when the topic is the *glass transition process*. Resolving
that ambiguity using document context is the whole point of Stage 3.

## How it works

The pipeline runs a document's metadata through four stages:

```
 Stage 1                Stage 2                 Stage 3                  Stage 4
 Dual-Channel   -->   Neural Span      -->    PMEST Facet       -->   Disentangled
 Encoding             Extraction &            Classification          Vector Space
                       Scoring
```

**Stage 1 — Ingestion & Structural Encoding.** Metadata is split into a
textual stream (Title, Abstract, Description) and a structural stream
(dates, coordinates, author, publisher). Textual tokens are embedded
with token + position + field-type + salience information, then run
through a dual-channel encoder: a bidirectional Transformer over all
tokens (Channel A), and a Graph Attention Network over a metadata
dependency graph — Author↔Institution, Publication↔Region, etc.
(Channel B). The two channels are fused into one token-level embedding
matrix, `X_fused`.

**Stage 2 — Candidate Extraction & Scoring.** KeyBERT proposes candidate
keyword spans, which are aligned back onto `X_fused`'s token
coordinates. Each span gets a pooled representation (boundary vectors +
soft-attention pool + length embedding), then two scoring heads
(Informativeness, Document Coverage) combine into a raw keyword score.
Dynamic thresholding + Non-Maximum Suppression prune overlapping,
low-value candidates down to the final keyword set.

**Stage 3 — Facet Disambiguation.** For each retained keyword, three
signals are assembled: the keyword's own pooled vector, a document-level
global vector, and the structural graph context of the field it came
from (Part 1). These feed a facet classifier that outputs a five-way
probability distribution over P/M/E/S/T (Part 2), from which a hard
(argmax) or soft (full distribution) assignment can be read (Part 3).

**Stage 4 — Disentangled Embedding.** The keyword's raw candidate vector
is projected into five facet-specific subspaces (160-D Personality,
96-D Matter, 128-D Energy, 64-D Space, 64-D Time — 512-D total), each
scaled by its Stage 3 probability before concatenation. A keyword that's
almost purely spatial ends up with an active Space subspace and every
other subspace scaled toward zero — geometric purity by construction,
not by post-hoc filtering.

## Status: what's real vs. placeholder

This is the section to read before trusting any number the pipeline
produces. Nothing here is hidden — the code's own docstrings say the
same thing, this just collects it in one place.

| Component | Status |
|---|---|
| Stage 1 (dual-channel encoder) | **Real, trained-architecture, untrained weights.** Runs correctly; weights are randomly initialized (no training loop exists yet), so encoder outputs are structurally correct but not yet optimized. |
| Stage 2 (span extraction + scoring) | **Real, same caveat as Stage 1** — the scoring heads' logic (informativeness, coverage, NMS) is fully implemented and verified, but their weights are untrained. |
| Stage 3 Part 1 (context assembly) | **Real.** Builds the three-signal fused vector exactly per spec, ready for a trained classifier once one exists. |
| Stage 3 Part 2/3 (facet classification) | **Zero-shot, not trained.** Uses a pretrained NLI model (`MoritzLaurer/deberta-v3-large-zeroshot-v2.0` by default) against natural-language facet descriptions, blended with a structural prior (field → facet_bias) and a centrality-based confidence sharpener. This is a genuine working classifier — see [Known Limitations](#known-limitations) for where its structural-bias path currently goes dark. |
| Stage 4 Part 1 (subspace layout) | **Real** — reads and validates the 512-D allocation from config. |
| Stage 4 Part 2 (subspace projection) | **Deterministic placeholder, not trained.** The spec calls for five trained linear projection matrices (`W_P...W_T`). Those don't exist without a training loop, so each facet instead claims a fixed, contiguous slice of the input vector. The *gating* (multiplying by Stage 3's probability) is real and does meaningful work; the *projection* itself carries no learned semantics yet. |
| Stage 5 (training: `L_kw`, `L_facet`, `L_ortho`, `L_triplet`) | **Not implemented.** `src/pipelines/train.py`, `src/losses/multi_task_loss.py`, and both files in `tests/` are empty stubs. Training requires labeled facet data (from Colon Classification, AGROVOC, MeSH, or hand annotation), which doesn't exist in this repo yet. |

## Repository structure

```
config/
  default_config.yaml     # all tunable settings: model names, dims, thresholds
  field_mappings.json      # per-field parsing rules + facet_bias hints
data/
  raw/                     # input records (sample_record.json, and any CSV you classify)
  processed/               # batch classification output + dedup state (see below)
  vocab/                   # cached corpus statistics (IDF)
src/
  preprocessing/           # Stage 1 Part 1-2: decomposition, tokenization, graph building
  models/
    encoders.py            # Stage 1 Part 3: dual-channel encoder
    span_extractor.py       # Stage 2: candidate extraction + scoring + NMS
    facet_classifier.py     # Stage 3: context assembly + zero-shot classification
    pmest_embedding.py      # Stage 4: subspace layout + projection + gating
  losses/                  # Stage 5 (stub)
  pipelines/               # train.py / infer.py (stubs)
  utils/                   # corpus stats, logging
scripts/
  check_stage*.py          # runnable smoke tests for each stage/part, printed + asserted
  interactive_demo.py       # CLI: type a title/abstract, see the full pipeline's output
  batch_classify_csv.py     # batch-classify a CSV of records, append results to a running CSV
tests/                     # currently empty stubs; verification lives in scripts/check_stage*.py instead
```

## Installation

```bash
git clone https://github.com/SanayAmI/Colon-Prototype.git
cd Colon-Prototype
pip install -r requirements.txt
```

Two models download from the HuggingFace Hub on first run (no action
needed beyond having network access): `bert-base-uncased` (Stage 1's
tokenizer) and `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` (Stage 3's
zero-shot classifier, ~870MB). Set `HF_TOKEN` in your environment for
higher rate limits if you hit throttling.

## How to access / run it

There's no server or API in this repo yet — everything runs as a local
script. Three ways in, depending on what you want:

### Try it interactively

Type a title/abstract and see the full Stage 1→4 breakdown for your own
input, without touching a CSV or writing code:

```bash
python scripts/interactive_demo.py
```

Models load once at startup (can take a while on first run), then it
loops accepting input until you type `quit`. For each keyword it
extracts, you'll see the facet probability distribution, the hard
assignment, and whether Stage 3's structural-bias signal was available
for that field.

### Classify a CSV of your own

To batch-process a dataset (e.g. book/paper metadata) into per-keyword
PMEST classifications:

```bash
python scripts/batch_classify_csv.py path/to/your_dataset.csv
```

- **Input**: a CSV with at minimum a `title` column (case-insensitive;
  `abstract`/`summary`/`description` and date/coordinate columns are
  picked up automatically if present — see `COLUMN_ALIASES` in the
  script to add your own column names).
- **Output**: appended to `data/processed/pmest_classifications.csv` —
  one row per extracted keyword, with all five facet probabilities, the
  hard assignment, and the full 512-D PMEST vector as separate numeric
  columns (`pmest_0` ... `pmest_511`).
- **Re-running is safe**: every row's dedup key (an ID/ISBN column if
  your CSV has one, otherwise a hash of title+abstract) is recorded in
  `data/processed/seen_keys.json`. A row that's already been classified
  — in this run or any previous one, from this file or a different one
  — is skipped, not reclassified and not duplicated in the output.

### Run the pipeline stage-by-stage

Each stage has a standalone, runnable check script that builds
synthetic or sample data, runs that stage's code, and prints + asserts
its output — useful for understanding one stage in isolation or
verifying a change didn't break anything upstream:

```bash
python scripts/check_stage1_part3.py     # dual-channel encoder
python scripts/check_stage2_part1.py     # candidate span extraction
python scripts/check_stage2_part2.py     # scoring heads + NMS
python scripts/check_stage3_part1.py     # facet context assembly
python scripts/check_stage3_part2_3.py   # zero-shot facet classification
python scripts/check_stage4_part1_2_3.py # disentangled vector projection
```

`scripts/check_stage1_part2.py` is deprecated (superseded by
`check_stage1_part3.py`) and kept only for reference.

## Known limitations

- **Stage 3's structural-bias path is currently dark.** `field_mappings.json`
  tags `Location_Meta`/`Geolocation` → `space` and `Temporal_Meta`/
  `Timestamp` → `time`, and Stage 3 Part 2 is built to blend that prior
  in — but only when the field's Stage 1 graph node carries real
  (non-zero) structural signal. Right now, `graph_builder.py` gives
  numeric/parsed fields (dates, coordinates) a zero-vector node, so the
  confidence gate never opens on the live pipeline. It's verified to
  work correctly on synthetic non-zero input (`check_stage3_part2_3.py`);
  it just isn't reachable yet from real metadata.
- **Corpus statistics (IDF) reflect whatever's in `data/raw/`, not
  necessarily your dataset.** If you batch-classify a large CSV, the
  Informativeness Head's rarity prior is only as good as the corpus it
  was built from — consider rebuilding `data/vocab/corpus_stats.json`
  from your own dataset before relying on informativeness scores.
- **KeyBERT's keyword sampling isn't seeded**, so re-running the same
  input can surface a different (though usually overlapping) set of top
  candidate phrases. This is expected variance, not a bug.
- **No trained weights anywhere in the pipeline** — see the status table
  above. Every number this pipeline produces today is either zero-shot
  (Stage 3's classification) or structurally-correct-but-untrained
  (everything else). Treat outputs as a working prototype of the
  *architecture*, not as production-quality embeddings.

## Roadmap

The path from this MVP to the fully trained architecture described in
the theoretical spec:

1. **Build a labeled facet dataset** — via an existing ontology (MeSH,
   AGROVOC), distant supervision from Colon Classification schemes, or
   hand annotation. This is the actual bottleneck; everything else below
   is comparatively mechanical once labels exist.
2. **Replace Stage 3's zero-shot classifier** with a trained "5-Way
   Gated Multi-Label Attention Layer" that consumes Stage 3 Part 1's
   fused context vector directly.
3. **Replace Stage 4's deterministic slicing** with real trained
   projection matrices `W_P...W_T`.
4. **Implement Stage 5's multi-task loss** — keyword extraction loss,
   facet classification loss, orthogonality loss (to keep the five
   subspaces genuinely disentangled), and a triplet loss over the
   512-D space — and wire up `src/pipelines/train.py`.
5. **Facet-filtered retrieval** — once real embeddings exist, build the
   query-time weighting described in the spec:
   `Sim(Q, D) = α·S_P + β·S_M + γ·S_E + δ·S_S + ε·S_T`.

## Citation & license

Released under the MIT License. If you use PMEST-Net in academic work,
please cite Dr. S. R. Ranganathan's original Colon Classification scheme
alongside this repository.
