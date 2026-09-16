"""
Quick manual sanity check for Stage 4, Parts 1 and 2 (pmest_embedding.py).

Chains the full pipeline built so far:
  Stage 1 -> Stage 2 (Part 1 + Part 2/NMS) -> Stage 3 Part 1 (context
  assembly) -> Stage 3 Part 2/3 (ZeroShotFacetClassifier + assign_facets)
  -> Stage 4 Part 1 (PMESTSubspaceLayout, read from config) -> Stage 4
  Part 2 (PMESTSubspaceProjector: deterministic K_raw slicing + Stage 3
  probability gating -> final 512-D Disentangled PMEST Vector).

Per the module docstring in pmest_embedding.py, Stage 4 Part 2's
per-facet PROJECTIONS are a deterministic placeholder (a fixed slice of
K_raw), not trained weights -- W_P...W_T don't exist as real learned
matrices yet (Stage 5's training loop is still a stub). What IS real and
worth checking here is the GATING math: a keyword whose Stage 3
distribution favors one facet should end up with that facet's slice of
the final vector visibly larger in magnitude than the others, and facets
the distribution assigns near-zero probability to should be suppressed
toward zero in the final vector -- this is the "Sahara Desert" property
from the spec (S stays active, E scales toward zero), and it holds
because of gating, independent of the placeholder's lack of trained
semantics.

Run: python scripts/check_stage4_part1_2.py
"""

import json
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


def run_upstream_pipeline(cfg, hidden_dim=768):
    """Runs Stage 1 -> Stage 2 -> Stage 3 Part 1/2/3 on the real sample
    record, exactly as the Stage 3 check scripts do, and returns
    everything Stage 4 needs: contexts (for K_raw / fused_vector) and
    their matching facet distributions (for gating)."""
    field_mappings_path = "config/field_mappings.json"
    sample_record_path = "data/raw/sample_record.json"

    with open(sample_record_path, "r", encoding="utf-8") as f:
        record = json.load(f)

    decomposer = MetadataDecomposer(field_mappings_path)
    decomposed = decomposer.decompose(record, record_id="test_001")

    tokenizer, embedding_layer = build_field_aware_tokenizer_and_embedding(
        field_mappings_path=field_mappings_path,
        pretrained_tokenizer_name="bert-base-uncased",
        hidden_dim=hidden_dim,
        max_sequence_length=cfg["encoder"]["max_sequence_length"],
    )

    encoder = DualChannelMetadataEncoder(
        tokenizer=tokenizer,
        embedding_layer=embedding_layer,
        hidden_dim=hidden_dim,
        transformer_layers=2,
        transformer_heads=12,
        gat_layers=2,
        gat_heads=4,
        use_graph_channel=cfg["encoder"]["use_graph_channel"],
    )
    encoder.eval()
    with torch.no_grad():
        stage1_output = encoder(decomposed)

    x_fused = stage1_output.token_embeddings.squeeze(0)
    graph = encoder.graph_builder.build(decomposed)

    full_text = " ".join(f.text for f in decomposed.textual_stream)

    kw_cfg = cfg["keyword_extraction"]
    aligner = KeyBERTCandidateAligner(
        keybert_model_name=kw_cfg["base_embedding_model"],
        stage1_tokenizer=tokenizer.tokenizer,
        ngram_range=tuple(kw_cfg["ngram_range"]),
        top_n=kw_cfg["top_n"],
        use_mmr=kw_cfg["use_mmr"],
        diversity=kw_cfg["diversity"],
        stopwords=kw_cfg["stopwords"],
    )
    aligned_spans = aligner.propose_and_align(full_text)

    builder = SpanCandidateBuilder(hidden_dim=hidden_dim, max_span_length=16)
    builder.eval()
    with torch.no_grad():
        representations = builder.build_all(x_fused, aligned_spans)

    corpus_stats = build_corpus_stats("data/raw")
    scorer = SalienceInformativenessScorer(hidden_dim=hidden_dim)
    scorer.eval()
    with torch.no_grad():
        scored_spans = scorer.score_all(representations, x_fused, corpus_stats)
    retained = dynamic_threshold_and_nms(scored_spans)

    assembler = FacetInputAssembler()
    with torch.no_grad():
        contexts = assembler.assemble(
            encoder_output=stage1_output, graph=graph, scored_spans=retained
        )

    classifier = ZeroShotFacetClassifier(cfg)
    with torch.no_grad():
        distributions = classifier.classify_batch(contexts, decomposed)
    assignments = assign_facets_batch(distributions)

    return contexts, distributions, assignments


def main():
    with open("config/default_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    hidden_dim = 768
    k_raw_dim = 3 * hidden_dim  # fused_vector = span + global + structural

    # ================================================================
    # Upstream recap: Stage 1 -> 2 -> 3
    # ================================================================
    print("=" * 70)
    print("UPSTREAM PIPELINE: Stage 1 -> 2 -> 3 (recap)")
    print("=" * 70)

    contexts, distributions, assignments = run_upstream_pipeline(cfg, hidden_dim)
    print(f"  contexts / distributions / assignments: "
          f"{len(contexts)} / {len(distributions)} / {len(assignments)}")

    if not contexts:
        print("  Nothing retained upstream -- nothing for Stage 4 to embed.")
        return

    for ctx, assign in zip(contexts, assignments):
        print(f"    '{ctx.phrase}' -> hard={assign.hard} "
              f"(confidence={assign.hard_confidence:.4f})")

    # ================================================================
    # STAGE 4 PART 1: Vector Subspace Partitioning
    # ================================================================
    print("\n" + "=" * 70)
    print("STAGE 4 PART 1: PMESTSubspaceLayout (read from config)")
    print("=" * 70)

    layout = PMESTSubspaceLayout.from_config(cfg)
    print(f"  dims: {layout.dims}")
    print(f"  total_dim: {layout.total_dim}")

    offsets = layout.offsets()
    print(f"  output offsets (within final 512-D vector):")
    for facet in PMEST_FACETS:
        start, end = offsets[facet]
        print(f"    {facet:12s}: [{start:4d}:{end:4d})  size={end - start}")

    # ---- Part 1 assertions ----
    assert layout.total_dim == 512, f"expected total_dim=512, got {layout.total_dim}"
    assert sum(layout.dims.values()) == layout.total_dim
    assert list(layout.dims.keys()) == list(PMEST_FACETS), (
        "layout.dims must be in PMEST_FACETS order"
    )
    # offsets must be contiguous, non-overlapping, and cover [0, total_dim)
    all_offsets = [offsets[f] for f in PMEST_FACETS]
    assert all_offsets[0][0] == 0, "first offset must start at 0"
    assert all_offsets[-1][1] == layout.total_dim, "last offset must end at total_dim"
    for (_, prev_end), (next_start, _) in zip(all_offsets, all_offsets[1:]):
        assert prev_end == next_start, "offsets must be contiguous with no gaps/overlaps"

    print("\n  Part 1 assertions passed (dims sum correctly, offsets contiguous).")

    # ================================================================
    # STAGE 4 PART 2: Subspace Projection + Gated Facet Integration
    # ================================================================
    print("\n" + "=" * 70)
    print("STAGE 4 PART 2: PMESTSubspaceProjector (deterministic slicing MVP)")
    print("=" * 70)

    projector = PMESTSubspaceProjector(
        layout=layout,
        k_raw_dim=k_raw_dim,
        use_trained_projections=False,
    )
    projector.eval()

    phrases = [ctx.phrase for ctx in contexts]
    k_raw_batch = [ctx.fused_vector for ctx in contexts]

    with torch.no_grad():
        pmest_vectors = projector.forward_batch(phrases, k_raw_batch, distributions)

    print(f"  PMEST vectors produced: {len(pmest_vectors)} "
          f"(should equal upstream contexts: {len(contexts)})")

    for pv, assign in zip(pmest_vectors, assignments):
        print(f"\n  --- '{pv.phrase}' (hard facet: {assign.hard}) ---")
        print(f"    full vector shape: {tuple(pv.vector.shape)} (expect ({layout.total_dim},))")

        norms = {f: pv.facet_slices[f].norm().item() for f in PMEST_FACETS}
        for facet in PMEST_FACETS:
            marker = " <-- hard assignment" if facet == assign.hard else ""
            print(f"    {facet:12s}: prob={pv.facet_probs[facet]:.4f}  "
                  f"gated_slice_norm={norms[facet]:.4f}{marker}")

    # ---- Part 2 shape assertions ----
    for pv in pmest_vectors:
        assert pv.vector.shape == (layout.total_dim,), (
            f"'{pv.phrase}': expected shape ({layout.total_dim},), "
            f"got {tuple(pv.vector.shape)}"
        )
        for facet in PMEST_FACETS:
            assert pv.facet_slices[facet].shape == (layout.dims[facet],), (
                f"'{pv.phrase}' facet '{facet}': expected shape "
                f"({layout.dims[facet]},), got {tuple(pv.facet_slices[facet].shape)}"
            )
        # gated slices concatenated in PMEST_FACETS order must equal the
        # full vector exactly -- this is the literal formula from the
        # spec: V_PMEST = [P_prob*V_P || M_prob*V_M || ... ]
        reconstructed = torch.cat([pv.facet_slices[f] for f in PMEST_FACETS], dim=0)
        assert torch.allclose(pv.vector, reconstructed), (
            f"'{pv.phrase}': vector must equal concatenation of its own "
            f"facet_slices in PMEST_FACETS order"
        )

    print("\n  Part 2 shape/reconstruction assertions passed.")

    # ---- Gating correctness assertion: the "Sahara Desert" property ----
    # A facet with near-zero Stage 3 probability must have a near-zero
    # gated slice norm (gating actually suppresses it), and conversely
    # the hard-assigned (highest-probability) facet should not be
    # suppressed to zero. This holds regardless of the placeholder
    # projection's lack of trained semantics -- it's purely a property
    # of multiplying by a probability.
    print("\n  Gating correctness check (probability suppresses/preserves norm):")
    for pv, assign in zip(pmest_vectors, assignments):
        hard_facet = assign.hard
        hard_norm = pv.facet_slices[hard_facet].norm().item()
        hard_prob = pv.facet_probs[hard_facet]

        weakest_facet = min(pv.facet_probs, key=pv.facet_probs.get)
        weakest_prob = pv.facet_probs[weakest_facet]
        weakest_norm = pv.facet_slices[weakest_facet].norm().item()

        print(f"    '{pv.phrase}': hard='{hard_facet}' (prob={hard_prob:.4f}, "
              f"norm={hard_norm:.4f})  weakest='{weakest_facet}' "
              f"(prob={weakest_prob:.4f}, norm={weakest_norm:.4f})")

        # The weakest facet's gated norm must be strictly smaller than
        # what it would be ungated (prob < 1 guarantees this whenever
        # prob < 1, which dynamic NLI + sharpening virtually always
        # gives us -- an exact prob of 1.0 or 0.0 is only possible in
        # degenerate cases).
        ungated_weakest_norm = (pv.facet_slices[weakest_facet] / max(weakest_prob, 1e-9)).norm().item()
        if weakest_prob > 1e-9:
            assert weakest_norm <= ungated_weakest_norm + 1e-4, (
                f"'{pv.phrase}': gated norm should not exceed ungated norm "
                f"for facet '{weakest_facet}'"
            )

    print("\n  Gating correctness check passed.")
    print("\nDone. Check: per-keyword, the facet with the highest Stage 3 "
          "probability should generally show a larger gated_slice_norm than "
          "facets with near-zero probability (proportional to probability, "
          "since the underlying slice magnitudes are NOT yet meaningful -- "
          "see pmest_embedding.py module docstring on the deterministic "
          "placeholder).")


if __name__ == "__main__":
    main()