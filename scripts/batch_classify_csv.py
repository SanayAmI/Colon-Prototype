"""
Batch classification pipeline: CSV of book/document metadata IN,
CSV of per-keyword PMEST classifications OUT (appended across runs).

Usage:
    python scripts/batch_classify_csv.py path/to/input.csv

Column mapping: auto-detects common column name variants (case-
insensitive) for Title, Abstract, Temporal_Meta (date), and
Location_Meta (coordinates). See COLUMN_ALIASES below -- if your CSV
uses different names, add them there rather than renaming your file.

Deduplication (row-level, per project decision -- NOT keyword-level,
since a keyword's facet classification is document-context-dependent
and must never be reused across different source books):
    - If the input CSV has a recognizable ID column (isbn, id,
      record_id, book_id -- case-insensitive), that column's value is
      the dedup key.
    - Otherwise, falls back to a normalized hash of Title (+ Abstract
      if present).
    - Every key ever processed is recorded in
      data/processed/seen_keys.json (created on first run). A row whose
      key is already in there is SKIPPED ENTIRELY -- not reclassified,
      not re-added to the output CSV -- regardless of which input CSV
      or which run it appears in.

Output: data/processed/pmest_classifications.csv, APPENDED to across
runs (header written only if the file doesn't exist yet). One row per
EXTRACTED KEYWORD (a book can produce several), including all 512
PMEST vector dimensions as separate columns (pmest_0 ... pmest_511).

Run: python scripts/batch_classify_csv.py your_dataset.csv
"""

import csv
import hashlib
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from src.preprocessing.cleaner import MetadataDecomposer
from src.preprocessing.tokenization import build_field_aware_tokenizer_and_embedding
from src.models.encoders import DualChannelMetadataEncoder
from src.models.span_extractor import (
    KeyBERTCandidateAligner,
    SpanCandidateBuilder,
    SalienceInformativenessScorer,
    dynamic_threshold_and_nms,
)
from src.models.facet_classifier import (
    FacetInputAssembler,
    ZeroShotFacetClassifier,
    assign_facets_batch,
    PMEST_FACETS,
)
from src.models.pmest_embedding import PMESTSubspaceLayout, PMESTSubspaceProjector
from src.utils.corpus_stats import build_corpus_stats


# ----------------------------------------------------------------------------
# Column mapping: your CSV's column names -> the pipeline's expected fields.
# Matching is case-insensitive and ignores surrounding whitespace. Add more
# aliases here if your dataset uses different names than these.
# ----------------------------------------------------------------------------
COLUMN_ALIASES = {
    "Title": ["title", "book_title", "name"],
    "Abstract": ["abstract", "summary", "description", "synopsis"],
    "Temporal_Meta": ["temporal_meta", "date_range", "publication_date", "date"],
    "Location_Meta": ["location_meta", "coordinates", "geolocation", "location"],
}
ID_COLUMN_ALIASES = ["isbn", "id", "record_id", "book_id", "uid"]

STATE_DIR = Path("data/processed")
SEEN_KEYS_PATH = STATE_DIR / "seen_keys.json"
OUTPUT_CSV_PATH = STATE_DIR / "pmest_classifications.csv"

OUTPUT_COLUMNS = (
    ["dedup_key", "dedup_key_source", "source_row_title", "phrase", "source_field"]
    + [f"{facet}_prob" for facet in PMEST_FACETS]
    + ["hard_facet", "hard_confidence"]
    + [f"pmest_{i}" for i in range(512)]
    + ["processed_at"]
)


def _normalize_header(header: str) -> str:
    return header.strip().lower()


def detect_column_mapping(fieldnames: list[str]) -> dict[str, str]:
    """Maps pipeline field names (Title, Abstract, ...) to this CSV's
    actual column names. Raises if Title (the only strictly required
    field -- nothing can be extracted without it) isn't found."""
    normalized_to_actual = {_normalize_header(h): h for h in fieldnames}

    mapping = {}
    for pipeline_field, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalized_to_actual:
                mapping[pipeline_field] = normalized_to_actual[alias]
                break

    if "Title" not in mapping:
        raise ValueError(
            f"Could not find a Title column in {fieldnames}. "
            f"Add your column name to COLUMN_ALIASES['Title'] in this script."
        )

    return mapping


def detect_id_column(fieldnames: list[str]) -> str | None:
    normalized_to_actual = {_normalize_header(h): h for h in fieldnames}
    for alias in ID_COLUMN_ALIASES:
        if alias in normalized_to_actual:
            return normalized_to_actual[alias]
    return None


def compute_dedup_key(row: dict, id_column: str | None, column_mapping: dict) -> tuple[str, str]:
    """Returns (key, source) where source is 'id_column:<name>' or
    'title_hash' -- recorded in the output for auditability."""
    if id_column is not None:
        id_value = (row.get(id_column) or "").strip()
        if id_value:
            return id_value, f"id_column:{id_column}"

    title = (row.get(column_mapping.get("Title", ""), "") or "").strip()
    abstract = (row.get(column_mapping.get("Abstract", ""), "") or "").strip()
    normalized = (title + "|" + abstract).strip().lower()
    key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return key, "title_hash"


def load_seen_keys() -> set[str]:
    if not SEEN_KEYS_PATH.exists():
        return set()
    with open(SEEN_KEYS_PATH, "r", encoding="utf-8") as f:
        return set(json.load(f))


def save_seen_keys(seen: set[str]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(SEEN_KEYS_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(seen), f, indent=2)


def row_to_record(row: dict, column_mapping: dict) -> dict:
    record = {}
    for pipeline_field, actual_column in column_mapping.items():
        value = (row.get(actual_column) or "").strip()
        if value:
            record[pipeline_field] = value
    return record


class BatchClassifier:
    """Loads every model ONCE, then classify_record() runs Stage 1 -> 4
    on one record. Same structure as PMESTNetDemoSession in
    interactive_demo.py -- kept as a separate class here rather than
    importing that one, since this one is batch-oriented (no stdin
    prompts) and the two scripts have different lifecycles."""

    def __init__(self, cfg: dict, hidden_dim: int = 768):
        self.hidden_dim = hidden_dim
        self.k_raw_dim = 3 * hidden_dim
        field_mappings_path = "config/field_mappings.json"

        print("Loading models (first run downloads ~1-2GB from HuggingFace Hub)...")

        self.decomposer = MetadataDecomposer(field_mappings_path)
        self.tokenizer, self.embedding_layer = build_field_aware_tokenizer_and_embedding(
            field_mappings_path=field_mappings_path,
            pretrained_tokenizer_name="bert-base-uncased",
            hidden_dim=hidden_dim,
            max_sequence_length=cfg["encoder"]["max_sequence_length"],
        )
        self.encoder = DualChannelMetadataEncoder(
            tokenizer=self.tokenizer,
            embedding_layer=self.embedding_layer,
            hidden_dim=hidden_dim,
            transformer_layers=2,
            transformer_heads=12,
            gat_layers=2,
            gat_heads=4,
            use_graph_channel=cfg["encoder"]["use_graph_channel"],
        )
        self.encoder.eval()

        kw_cfg = cfg["keyword_extraction"]
        self.aligner = KeyBERTCandidateAligner(
            keybert_model_name=kw_cfg["base_embedding_model"],
            stage1_tokenizer=self.tokenizer.tokenizer,
            ngram_range=tuple(kw_cfg["ngram_range"]),
            top_n=kw_cfg["top_n"],
            use_mmr=kw_cfg["use_mmr"],
            diversity=kw_cfg["diversity"],
            stopwords=kw_cfg["stopwords"],
        )

        self.span_builder = SpanCandidateBuilder(hidden_dim=hidden_dim, max_span_length=16)
        self.span_builder.eval()

        # NOTE: corpus stats are still built from data/raw only (the
        # original single sample record) -- this batch script does NOT
        # yet rebuild IDF stats from the dataset being classified. See
        # "known limitations" printed at the end of main().
        self.corpus_stats = build_corpus_stats("data/raw")
        self.scorer = SalienceInformativenessScorer(hidden_dim=hidden_dim)
        self.scorer.eval()

        self.assembler = FacetInputAssembler()
        self.classifier = ZeroShotFacetClassifier(cfg)

        self.layout = PMESTSubspaceLayout.from_config(cfg)
        self.projector = PMESTSubspaceProjector(
            layout=self.layout, k_raw_dim=self.k_raw_dim, use_trained_projections=False
        )
        self.projector.eval()

        print("Models loaded.\n")

    def classify_record(self, record: dict, record_id: str):
        decomposed = self.decomposer.decompose(record, record_id=record_id)

        with torch.no_grad():
            stage1_output = self.encoder(decomposed)
        x_fused = stage1_output.token_embeddings.squeeze(0)
        graph = self.encoder.graph_builder.build(decomposed)

        full_text = " ".join(f.text for f in decomposed.textual_stream)
        if not full_text.strip():
            return [], [], [], []

        aligned_spans = self.aligner.propose_and_align(full_text)
        with torch.no_grad():
            representations = self.span_builder.build_all(x_fused, aligned_spans)
        if not representations:
            return [], [], [], []

        with torch.no_grad():
            scored_spans = self.scorer.score_all(representations, x_fused, self.corpus_stats)
        retained = dynamic_threshold_and_nms(scored_spans)
        if not retained:
            return [], [], [], []

        with torch.no_grad():
            contexts = self.assembler.assemble(
                encoder_output=stage1_output, graph=graph, scored_spans=retained
            )
            distributions = self.classifier.classify_batch(contexts, decomposed)
        assignments = assign_facets_batch(distributions)

        phrases = [c.phrase for c in contexts]
        k_raw_batch = [c.fused_vector for c in contexts]
        with torch.no_grad():
            pmest_vectors = self.projector.forward_batch(phrases, k_raw_batch, distributions)

        return contexts, distributions, assignments, pmest_vectors


def build_output_rows(dedup_key, dedup_source, title, contexts, distributions, assignments, pmest_vectors):
    rows = []
    processed_at = datetime.now(timezone.utc).isoformat()

    for ctx, dist, assign, pv in zip(contexts, distributions, assignments, pmest_vectors):
        row = {
            "dedup_key": dedup_key,
            "dedup_key_source": dedup_source,
            "source_row_title": title,
            "phrase": ctx.phrase,
            "source_field": ctx.source_field or "",
            "hard_facet": assign.hard,
            "hard_confidence": f"{assign.hard_confidence:.6f}",
            "processed_at": processed_at,
        }
        probs = dist.as_dict()
        for facet in PMEST_FACETS:
            row[f"{facet}_prob"] = f"{probs[facet]:.6f}"

        vector = pv.vector.tolist()
        for i, value in enumerate(vector):
            row[f"pmest_{i}"] = f"{value:.6f}"

        rows.append(row)
    return rows


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/batch_classify_csv.py path/to/input.csv")
        sys.exit(1)

    input_csv_path = Path(sys.argv[1])
    if not input_csv_path.exists():
        print(f"Error: input file not found: {input_csv_path}")
        sys.exit(1)

    with open("config/default_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    with open(input_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            print("Error: input CSV has no header row.")
            sys.exit(1)

        column_mapping = detect_column_mapping(fieldnames)
        id_column = detect_id_column(fieldnames)
        print(f"Column mapping detected: {column_mapping}")
        print(f"ID column: {id_column if id_column else '(none found -- falling back to title/abstract hash)'}\n")

        rows = list(reader)

    seen_keys = load_seen_keys()
    print(f"Already-processed keys on record: {len(seen_keys)}\n")

    classifier = BatchClassifier(cfg)

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    output_exists = OUTPUT_CSV_PATH.exists()

    total_rows = len(rows)
    skipped_duplicate = 0
    skipped_no_keywords = 0
    processed = 0
    new_output_rows = []

    for i, row in enumerate(rows, start=1):
        dedup_key, dedup_source = compute_dedup_key(row, id_column, column_mapping)

        if dedup_key in seen_keys:
            skipped_duplicate += 1
            continue

        record = row_to_record(row, column_mapping)
        title = record.get("Title", "")

        if not title:
            print(f"  [{i}/{total_rows}] skipping row with empty Title")
            continue

        print(f"  [{i}/{total_rows}] classifying: '{title[:60]}'"
              f"{'...' if len(title) > 60 else ''}")

        contexts, distributions, assignments, pmest_vectors = classifier.classify_record(
            record, record_id=dedup_key
        )

        # Mark as seen REGARDLESS of whether keywords were extracted --
        # per project decision, a row that's been through the pipeline
        # once should not be reprocessed on a later run, even if it
        # happened to yield zero keywords (re-running it wouldn't
        # produce a different result, since nothing about the input
        # changed).
        seen_keys.add(dedup_key)

        if not contexts:
            skipped_no_keywords += 1
            continue

        new_output_rows.extend(
            build_output_rows(dedup_key, dedup_source, title, contexts, distributions, assignments, pmest_vectors)
        )
        processed += 1

    # ---- write output CSV: append, write header only if new ----
    if new_output_rows:
        with open(OUTPUT_CSV_PATH, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            if not output_exists:
                writer.writeheader()
            writer.writerows(new_output_rows)

    save_seen_keys(seen_keys)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  input rows total:            {total_rows}")
    print(f"  skipped (already processed): {skipped_duplicate}")
    print(f"  skipped (no keywords found): {skipped_no_keywords}")
    print(f"  newly processed:             {processed}")
    print(f"  keyword rows written:        {len(new_output_rows)}")
    print(f"  output file:                 {OUTPUT_CSV_PATH}")
    print(f"  seen-keys state file:        {SEEN_KEYS_PATH}")
    print("\nKNOWN LIMITATIONS:")
    print("  - corpus_stats (IDF priors) still come from data/raw's single")
    print("    sample record, NOT from this dataset -- informativeness scoring")
    print("    across a large real dataset may be weak/uninformative until")
    print("    corpus_stats is rebuilt from the dataset itself.")
    print("  - Stage 3's structural-bias gate will likely stay closed for")
    print("    every row unless your CSV's date/coordinate fields end up")
    print("    surviving Stage 2's NMS as extracted keywords (see")
    print("    facet_classifier.py's known limitation).")


if __name__ == "__main__":
    main()