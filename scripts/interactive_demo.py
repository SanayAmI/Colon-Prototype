"""
Interactive CLI gateway for the PMEST-Net MVP pipeline.

Lets a person type/paste a Title and Abstract (optionally a date and/or
coordinate string) and see the full pipeline's output for their own
input: Stage 2's extracted keywords, Stage 3's facet distributions +
hard assignments, and Stage 4's final PMEST vectors.

Models (bert-base-uncased for Stage 1's tokenizer, and Stage 3's
zero-shot NLI model per config/default_config.yaml) are loaded ONCE at
startup, then the script loops accepting input until you quit -- this
avoids re-paying the multi-second/multi-minute model load cost per
input (see check_stage4_part1_2.py's terminal output for how slow a
cold load can be).

KNOWN LIMITATION (same as every other check_stage*.py script): the
structural-bias gate in Stage 3 Part 2 (facet_classifier.py's
_structural_prior) currently never opens, because Location_Meta/
Temporal_Meta/etc. get zero-vector graph nodes from Stage 1's
MetadataGraphBuilder (they're numeric/parsed fields, not in
_TEXT_LIKE_STRUCTURAL_FIELDS). This CLI still lets you type a date/
coordinate so you can see that limitation for yourself -- it will
always print structural_vector_nonzero=False for those fields.

Run: python scripts/interactive_demo.py
"""

import sys
from pathlib import Path

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


class PMESTNetDemoSession:
    """Loads every model ONCE, then exposes a single classify_record()
    call that runs the full Stage 1 -> 4 pipeline on one input record.
    """

    def __init__(self, cfg: dict, hidden_dim: int = 768):
        self.cfg = cfg
        self.hidden_dim = hidden_dim
        self.k_raw_dim = 3 * hidden_dim

        field_mappings_path = "config/field_mappings.json"

        print("Loading models (this can take a while on first run, "
              "downloads ~1-2GB from HuggingFace Hub)...")

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

        # Corpus stats: only data/raw/sample_record.json exists today, so
        # this is a corpus of size 1 -- IDF priors will be weak/uninformative
        # until a real corpus is built (see the "database pull" plan).
        self.corpus_stats = build_corpus_stats("data/raw")
        self.scorer = SalienceInformativenessScorer(hidden_dim=hidden_dim)
        self.scorer.eval()

        self.assembler = FacetInputAssembler()
        self.classifier = ZeroShotFacetClassifier(cfg)

        self.layout = PMESTSubspaceLayout.from_config(cfg)
        self.projector = PMESTSubspaceProjector(
            layout=self.layout,
            k_raw_dim=self.k_raw_dim,
            use_trained_projections=False,
        )
        self.projector.eval()

        print("Models loaded. Ready.\n")

    def classify_record(self, record: dict, record_id: str = "cli_input"):
        """Runs Stage 1 -> 4 on one input record. Returns
        (contexts, distributions, assignments, pmest_vectors), any of
        which may be empty lists if nothing survived extraction/NMS."""
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


def prompt_for_record() -> dict | None:
    """Collects one record's fields from stdin. Returns None if the
    user wants to quit."""
    print("-" * 70)
    title = input("Title (or 'quit' to exit): ").strip()
    if title.lower() in ("quit", "exit", "q"):
        return None

    abstract = input("Abstract (optional, press Enter to skip): ").strip()
    date_str = input(
        "Date range, MM/DD/YYYY - MM/DD/YYYY (optional, press Enter to skip): "
    ).strip()
    coord_str = input(
        "Coordinates, e.g. '48.8566° N, 2.3522° E' (optional, press Enter to skip): "
    ).strip()

    record = {"Title": title}
    if abstract:
        record["Abstract"] = abstract
    if date_str:
        record["Temporal_Meta"] = date_str
    if coord_str:
        record["Location_Meta"] = coord_str

    return record


def print_results(contexts, distributions, assignments, pmest_vectors, layout):
    if not contexts:
        print("\n  No keywords survived extraction/NMS for this input. "
              "Try a longer Title/Abstract.\n")
        return

    print(f"\n  {len(contexts)} keyword(s) extracted:\n")

    for ctx, dist, assign, pv in zip(contexts, distributions, assignments, pmest_vectors):
        print(f"  --- '{ctx.phrase}' ---")
        print(f"    source_field: {ctx.source_field}")

        structural_is_real = not torch.allclose(
            ctx.structural_vector, torch.zeros_like(ctx.structural_vector)
        )
        print(f"    structural_vector_nonzero: {structural_is_real} "
              f"{'(structural bias could apply)' if structural_is_real else '(structural bias gate closed -- see known limitation)'}")

        print(f"    facet distribution:")
        for facet in PMEST_FACETS:
            marker = "  <-- hard" if facet == assign.hard else ""
            print(f"      {facet:12s}: {dist.as_dict()[facet]:.4f}{marker}")

        print(f"    PMEST vector: shape={tuple(pv.vector.shape)}, "
              f"hard-facet slice norm={pv.facet_slices[assign.hard].norm().item():.4f}")
        print()


def main():
    with open("config/default_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    session = PMESTNetDemoSession(cfg)

    print("Type a Title (and optionally Abstract/date/coordinates) to see")
    print("the full PMEST-Net pipeline's output. Type 'quit' to exit.\n")

    while True:
        record = prompt_for_record()
        if record is None:
            print("\nExiting.")
            break

        contexts, distributions, assignments, pmest_vectors = session.classify_record(record)
        print_results(contexts, distributions, assignments, pmest_vectors, session.layout)


if __name__ == "__main__":
    main()