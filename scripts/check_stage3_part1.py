"""
Quick manual sanity check for Stage 3, Part 1 (facet_classifier.py).

Chains the full pipeline built so far:
  Stage 1 (DualChannelMetadataEncoder -> X_fused + MetadataGraph)
  -> Stage 2 Part 1 (KeyBERT candidate proposal + span representations)
  -> Stage 2 Part 2 (Informativeness/Coverage scoring + dynamic
     threshold + NMS -> final retained keyword candidates)
  -> Stage 3 Part 1 (FacetInputAssembler: fuses each retained keyword's
     span vector + document's global vector + its originating field's
     structural graph vector into one FacetInputContext, ready for the
     eventual Stage 3 Part 2 classifier to consume)

Run: python scripts/check_stage3_part1.py
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
from src.models.facet_classifier import FacetInputAssembler
from src.utils.corpus_stats import build_corpus_stats


def main():
    field_mappings_path = "config/field_mappings.json"
    default_config_path = "config/default_config.yaml"
    sample_record_path = "data/raw/sample_record.json"

    with open(default_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    with open(sample_record_path, "r", encoding="utf-8") as f:
        record = json.load(f)

    # ---- Stage 1: Part 1 -> Part 2 -> Part 3 ----
    decomposer = MetadataDecomposer(field_mappings_path)
    decomposed = decomposer.decompose(record, record_id="test_001")

    hidden_dim = 768
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

    x_fused = stage1_output.token_embeddings.squeeze(0)  # (L, hidden_dim)

    # Need the raw MetadataGraph too (Stage 1 only returns the pooled
    # graph_node_embeddings on stage1_output, but Part 1 needs
    # node_field_names to map a span's source field -> its node row,
    # so we rebuild the same graph the encoder built internally).
    graph = encoder.graph_builder.build(decomposed)

    print("=" * 70)
    print("STAGE 1 OUTPUT (recap)")
    print("=" * 70)
    print(f"  X_fused shape: {tuple(x_fused.shape)}")
    print(f"  total tokens: {len(stage1_output.source_field_per_token)}")
    print(f"  graph nodes: {graph.num_nodes}  fields: {graph.node_field_names}")

    # ---- Reconstruct the SAME concatenated text Channel A tokenized ----
    full_text = " ".join(f.text for f in decomposed.textual_stream)

    # ---- Stage 2 Part 1: KeyBERT candidate proposal + span representations ----
    print("\n" + "=" * 70)
    print("STAGE 2 PART 1: candidate proposal + span representations (recap)")
    print("=" * 70)

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
    ok_spans = [s for s in aligned_spans if s.alignment_ok]
    print(f"  candidates aligned: {len(ok_spans)} / {len(aligned_spans)} proposed")

    builder = SpanCandidateBuilder(hidden_dim=hidden_dim, max_span_length=16)
    builder.eval()
    with torch.no_grad():
        representations = builder.build_all(x_fused, aligned_spans)
    print(f"  span representations built: {len(representations)}")

    # ---- Stage 2 Part 2: scoring + dynamic threshold + NMS ----
    print("\n" + "=" * 70)
    print("STAGE 2 PART 2: informativeness/coverage scoring + NMS (recap)")
    print("=" * 70)

    corpus_stats = build_corpus_stats("data/raw")
    scorer = SalienceInformativenessScorer(hidden_dim=hidden_dim)
    scorer.eval()
    with torch.no_grad():
        scored_spans = scorer.score_all(representations, x_fused, corpus_stats)

    retained = dynamic_threshold_and_nms(scored_spans)
    print(f"  candidates scored: {len(scored_spans)}")
    print(f"  candidates retained after threshold + NMS: {len(retained)}")
    for s in retained:
        print(f"    '{s.representation.span.phrase}' "
              f"(raw_keyword_score={s.raw_keyword_score:.4f})")

    if not retained:
        print("\n  No candidates survived NMS -- nothing to assemble for Stage 3. "
              "Try a different sample record or looser NMS params.")
        return

    # ---- Stage 3 Part 1: assemble fused facet-input context ----
    print("\n" + "=" * 70)
    print("STAGE 3 PART 1: FacetInputAssembler")
    print("=" * 70)

    assembler = FacetInputAssembler()
    with torch.no_grad():
        contexts = assembler.assemble(
            encoder_output=stage1_output,
            graph=graph,
            scored_spans=retained,
        )

    print(f"  contexts assembled: {len(contexts)} "
          f"(should equal 'candidates retained' above)")

    for ctx in contexts:
        print(f"\n  --- '{ctx.phrase}' (source_field={ctx.source_field}) ---")
        print(f"    span_vector shape:       {tuple(ctx.span_vector.shape)}       (expect {hidden_dim},)")
        print(f"    global_vector shape:     {tuple(ctx.global_vector.shape)}     (expect {hidden_dim},)")
        print(f"    structural_vector shape: {tuple(ctx.structural_vector.shape)} (expect {hidden_dim},)")
        print(f"    fused_vector shape:      {tuple(ctx.fused_vector.shape)}      (expect {3*hidden_dim},)")

        structural_is_zero = torch.allclose(ctx.structural_vector, torch.zeros(hidden_dim))
        print(f"    structural_vector is all-zero: {structural_is_zero} "
              f"({'expected -- span came from a non-graph field' if structural_is_zero else 'non-graph field but got a signal -- check mapping!' if ctx.source_field not in graph.node_field_names else 'field is in the graph, has real structural signal'})")

    # ---- Sanity assertions ----
    for ctx in contexts:
        assert ctx.span_vector.shape == (hidden_dim,)
        assert ctx.global_vector.shape == (hidden_dim,)
        assert ctx.structural_vector.shape == (hidden_dim,)
        assert ctx.fused_vector.shape == (3 * hidden_dim,)
        # global_vector must be IDENTICAL across all contexts from the same
        # document -- it's a document-level constant, not per-span.
        assert torch.allclose(ctx.global_vector, contexts[0].global_vector), (
            "global_vector should be the same for every span in one document"
        )
        # fused_vector must literally be the concatenation of the three parts
        reconstructed = torch.cat(
            [ctx.span_vector, ctx.global_vector, ctx.structural_vector], dim=0
        )
        assert torch.allclose(ctx.fused_vector, reconstructed), (
            "fused_vector must equal cat([span_vector, global_vector, structural_vector])"
        )

    print("\nAll shape and consistency assertions passed.")
    print("Done. Check: structural_vector is non-zero only for spans whose "
          "source_field is a structural (graph) field like Temporal_Meta / "
          "Location_Meta / Author, and zero for textual fields like Title/Abstract.")


if __name__ == "__main__":
    main()