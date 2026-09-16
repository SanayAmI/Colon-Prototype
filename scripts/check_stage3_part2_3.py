"""
Quick manual sanity check for Stage 3, Parts 2 and 3 (facet_classifier.py).

Chains the full pipeline built so far:
  Stage 1 -> Stage 2 (Part 1 + Part 2/NMS) -> Stage 3 Part 1 (context
  assembly) -> Stage 3 Part 2 (ZeroShotFacetClassifier) -> Stage 3
  Part 3 (assign_facets: soft distribution -> hard assignment)

This script has a SECOND job beyond the usual smoke test: Part 2's
structural-bias confidence gate (see facet_classifier.py's
_structural_prior) only opens when a keyword's source field BOTH (a) has
a facet_bias set in field_mappings.json, AND (b) got a real non-zero
graph node from Stage 1's MetadataGraphBuilder. As of this writing,
fields with a facet_bias (Location_Meta, Geolocation, Temporal_Meta,
Timestamp) are all numeric/parsed fields that graph_builder.py currently
gives a ZERO-vector node (see _TEXT_LIKE_STRUCTURAL_FIELDS) -- so on the
real sample record, the gate is expected to stay closed for every single
keyword. This script prints that explicitly per keyword so the
limitation is visible, then separately exercises the blending math on a
synthetic non-zero context so the "gate open" code path still has real
coverage even though the live pipeline can't reach it yet.

Run: python scripts/check_stage3_part2_3.py
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
    FacetInputContext,
    ZeroShotFacetClassifier,
    assign_facets_batch,
    PMEST_FACETS,
)
from src.utils.corpus_stats import build_corpus_stats


def run_live_pipeline(cfg, hidden_dim=768):
    """Runs Stage 1 -> Stage 2 -> Stage 3 Part 1 on the real sample
    record, exactly as check_stage3_part1.py does, and returns everything
    Part 2 needs (contexts + the DecomposedMetadata for facet_bias lookup)."""
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

    return contexts, decomposed


def main():
    with open("config/default_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    hidden_dim = 768

    # ================================================================
    # PART A: live pipeline on the real sample record
    # ================================================================
    print("=" * 70)
    print("LIVE PIPELINE: Stage 1 -> 2 -> 3 Part 1 (recap)")
    print("=" * 70)

    contexts, decomposed = run_live_pipeline(cfg, hidden_dim)
    print(f"  contexts assembled: {len(contexts)}")
    if not contexts:
        print("  No contexts to classify -- nothing further to check.")
        return

    print("\n" + "=" * 70)
    print("STAGE 3 PART 2: ZeroShotFacetClassifier (live contexts)")
    print("=" * 70)

    classifier = ZeroShotFacetClassifier(cfg)

    print("\n  Per-keyword structural-gate status (see script docstring):")
    gate_ever_opened = False
    for ctx in contexts:
        struct_attr = decomposed.get_structural_field(ctx.source_field) if ctx.source_field else None
        text_attr = decomposed.get_textual_field(ctx.source_field) if ctx.source_field else None
        facet_bias = (struct_attr.facet_bias if struct_attr else None) or \
                     (text_attr.facet_bias if text_attr else None)
        structural_is_real = not torch.allclose(
            ctx.structural_vector, torch.zeros_like(ctx.structural_vector)
        )
        gate_open = facet_bias is not None and structural_is_real
        gate_ever_opened = gate_ever_opened or gate_open
        print(f"    '{ctx.phrase}': field={ctx.source_field}, "
              f"facet_bias={facet_bias}, structural_vector_nonzero={structural_is_real} "
              f"-> gate={'OPEN' if gate_open else 'closed'}")

    if not gate_ever_opened:
        print("\n  [EXPECTED, see docstring] Structural bias gate stayed CLOSED for "
              "every keyword in the live pipeline -- facet_bias fields "
              "(Temporal_Meta/Location_Meta/etc.) currently get zero-vector "
              "graph nodes, so ZeroShotFacetClassifier fell back to pure NLI + "
              "centrality sharpening for all of them. This is a known Stage 1 "
              "limitation, not a Part 2/3 bug.")

    print("\n  Distributions + hard assignments:")
    with torch.no_grad():
        distributions = classifier.classify_batch(contexts, decomposed)
    assignments = assign_facets_batch(distributions)

    for ctx, dist, assign in zip(contexts, distributions, assignments):
        print(f"\n  --- '{ctx.phrase}' ---")
        for facet in PMEST_FACETS:
            print(f"    {facet:12s}: {dist.as_dict()[facet]:.4f}")
        print(f"    hard assignment: {assign.hard} (confidence={assign.hard_confidence:.4f})")

    # ---- invariant assertions on the live results ----
    for dist in distributions:
        total = sum(dist.as_dict().values())
        assert abs(total - 1.0) < 1e-4, f"probabilities must sum to 1.0, got {total}"

    for dist, assign in zip(distributions, assignments):
        expected_hard = max(dist.as_dict(), key=dist.as_dict().get)
        assert assign.hard == expected_hard, "hard assignment must equal argmax of soft distribution"
        assert assign.hard_confidence == dist.as_dict()[expected_hard]

    print("\n  All live-pipeline invariants passed (sums to 1.0, hard == argmax).")

    # ================================================================
    # PART B: synthetic gate-open case (proves the blending math itself,
    # since the live pipeline currently can't reach this branch)
    # ================================================================
    print("\n" + "=" * 70)
    print("SYNTHETIC CHECK: structural-bias gate OPEN (Temporal_Meta, non-zero vector)")
    print("=" * 70)

    torch.manual_seed(0)
    synthetic_ctx = FacetInputContext(
        phrase="July and August 2019",
        span_vector=torch.randn(hidden_dim),
        global_vector=torch.randn(hidden_dim),
        structural_vector=torch.randn(hidden_dim),  # deliberately NON-zero
        source_field="Temporal_Meta",               # facet_bias = "time"
        fused_vector=torch.zeros(hidden_dim * 3),    # unused by Part 2
    )

    with torch.no_grad():
        nli_only_scores = classifier._nli_scores(synthetic_ctx.phrase)
    prior_vec, confidence = classifier._structural_prior(synthetic_ctx, decomposed)

    print(f"  facet_bias on Temporal_Meta: {prior_vec} (should be one-hot on 'time')")
    print(f"  confidence (should be 1.0, structural_vector is non-zero): {confidence}")
    assert confidence == 1.0, "confidence must be 1.0 when structural_vector is non-zero"
    assert prior_vec == [0.0, 0.0, 0.0, 0.0, 1.0], "prior must be one-hot on 'time' (index 4)"

    with torch.no_grad():
        blended_dist = classifier.classify(synthetic_ctx, decomposed, structural_bias_weight=0.4)

    print(f"\n  NLI-only scores (pre-blend, PMEST order): {[round(s, 4) for s in nli_only_scores]}")
    print(f"  Blended distribution (40% weight toward 'time' prior):")
    for facet in PMEST_FACETS:
        print(f"    {facet:12s}: {blended_dist.as_dict()[facet]:.4f}")

    # The blend should have pulled probability mass toward "time" relative
    # to the pure NLI score, proving the gate-open branch actually changes
    # the output when it fires.
    time_idx = PMEST_FACETS.index("time")
    nli_only_normalized = [s / sum(nli_only_scores) for s in nli_only_scores]
    assert blended_dist.time >= nli_only_normalized[time_idx], (
        "blending with an open gate should not DECREASE the biased facet's "
        "probability relative to pure NLI"
    )
    print("\n  Confirmed: blended 'time' probability >= pure-NLI 'time' probability.")
    print("  Synthetic gate-open branch is exercised and behaves correctly.")

    print("\nDone.")


if __name__ == "__main__":
    main()