"""PMEST facet classification model component (Stage 3, Part 1).

Part 1: Contextual Facet Attention input assembly.

This module does NOT classify anything. It builds the fused input vector
that Stage 3 Part 2 (the 5-Head Facet Classification Network) will
eventually consume. Per the architecture spec, that input is composed of
three ingredients per keyword candidate:

    1. The keyword's pooled span vector       (LOCAL SPAN CONTEXT)
    2. The global metadata embedding          (LOCAL SPAN CONTEXT side,
                                                document-level)
    3. The structural graph context           (METADATA STRUCTURE side)

Note on "global metadata embedding" vs "structural graph context":
the diagram groups (1)+(2) under "Local Span Context" and (3) under
"Metadata Structure", but the spec's prose (Contextual Facet Attention
Mechanism, item 2) separates the global embedding from the local span
vector. We follow the prose's three-way split here since it's more
precise, and keep the diagram's two-box grouping documented in
`FacetInputContext` for traceability.

This is intentionally spec-faithful rather than zero-shot-pragmatic: the
MVP's actual facet classifier (Stage 3 Part 2, not yet built) is a
zero-shot NLI model that classifies from the keyword's TEXT, and will NOT
consume this vector. This module exists so the fused vector is already
being produced and can be logged/cached now, ready for the eventual
trained 5-Way Gated Multi-Label Attention Layer to consume once it
exists -- see Stage 5 of the spec for its training objective.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import pipeline

import torch

from src.models.encoders import DualChannelEncoderOutput
from src.models.span_extractor import ScoredCandidateSpan
from src.preprocessing.graph_builder import MetadataGraph


@dataclass
class FacetInputContext:
    """The three-part fused representation for one keyword candidate,
    ready to be handed to Stage 3 Part 2's classifier once it exists.

    Shapes (hidden_dim = H, matches Stage 1's DualChannelMetadataEncoder):
        span_vector:        (H,)   -- from Stage 2's pooled_vector
        global_vector:      (H,)   -- document-level, mean-pooled x_fused
        structural_vector:  (H,)   -- this span's originating field's
                                       graph node embedding, pre-broadcast
                                       (i.e. NOT already mixed into
                                       span_vector -- see module docstring
                                       in graph_builder.py: x_fused already
                                       has structural context broadcast
                                       into every token, so this vector is
                                       a deliberately separate, "purer"
                                       structural signal pulled straight
                                       from graph_node_embeddings)
        source_field:       str    -- which field this span came from,
                                       kept for debugging / rule-based
                                       structural_bias hints later (e.g.
                                       "this field is Temporal_Meta" ->
                                       bias toward Time)
        fused_vector:       (3H,)  -- concatenation of the above three,
                                       the literal "combined representation"
                                       the spec says gets passed into the
                                       5-Head network
    """

    phrase: str
    span_vector: torch.Tensor
    global_vector: torch.Tensor
    structural_vector: torch.Tensor
    source_field: str | None
    fused_vector: torch.Tensor


class FacetInputAssembler:
    """Builds FacetInputContext objects for every retained Stage 2 keyword
    candidate belonging to one document/record.

    Usage:
        assembler = FacetInputAssembler()
        contexts = assembler.assemble(
            encoder_output=dual_channel_output,   # Stage 1 output
            graph=metadata_graph,                 # Stage 1's Channel B graph
            scored_spans=nms_filtered_candidates,  # Stage 2 output
        )
    """

    def _global_metadata_embedding(
        self, encoder_output: DualChannelEncoderOutput
    ) -> torch.Tensor:
        """Document-level embedding: mean-pool x_fused over real tokens
        only (respecting the attention mask, so padding doesn't dilute it).
        """
        token_embeddings = encoder_output.token_embeddings.squeeze(0)  # (seq_len, H)
        mask = encoder_output.attention_mask.squeeze(0).float()        # (seq_len,)

        mask_sum = mask.sum().clamp(min=1.0)
        masked_embeddings = token_embeddings * mask.unsqueeze(-1)
        return masked_embeddings.sum(dim=0) / mask_sum  # (H,)

    def _structural_vector_for_span(
        self,
        span_source_field: str | None,
        graph: MetadataGraph,
    ) -> torch.Tensor:
        """Looks up the graph node embedding for the field this span came
        from. Falls back to a zero vector if the span's field has no
        corresponding structural node (e.g. it came from a purely textual
        field like Abstract/Title that never entered the graph at all --
        the graph only contains structural_stream fields, see
        graph_builder.py's MetadataGraphBuilder.build()).
        """
        hidden_dim = graph.node_features.shape[1] if graph.num_nodes > 0 else 768

        if span_source_field is None or span_source_field not in graph.node_field_names:
            return torch.zeros(hidden_dim)

        node_idx = graph.node_field_names.index(span_source_field)
        return graph.node_features[node_idx]

    def assemble(
        self,
        encoder_output: DualChannelEncoderOutput,
        graph: MetadataGraph,
        scored_spans: list[ScoredCandidateSpan],
    ) -> list[FacetInputContext]:
        global_vector = self._global_metadata_embedding(encoder_output)

        contexts = []
        for scored in scored_spans:
            span = scored.representation.span
            pooled_vector = scored.representation.pooled_vector

            token_idx = span.token_start
            source_field = (
                encoder_output.source_field_per_token[token_idx]
                if 0 <= token_idx < len(encoder_output.source_field_per_token)
                else None
            )

            structural_vector = self._structural_vector_for_span(source_field, graph)

            fused_vector = torch.cat(
                [pooled_vector, global_vector, structural_vector], dim=0
            )  # (3H,)

            contexts.append(
                FacetInputContext(
                    phrase=span.phrase,
                    span_vector=pooled_vector,
                    global_vector=global_vector,
                    structural_vector=structural_vector,
                    source_field=source_field,
                    fused_vector=fused_vector,
                )
            )

        return contexts
    # ---------------------------------------------------------------------------
# Stage 3, Part 2 — Zero-Shot Facet Classification (MVP)
#
# Config-driven per config/default_config.yaml's `facet_classification`
# block: method, model, multi_label, hypothesis_template, and the five
# per-facet label descriptions are all read from there, not hardcoded here.
#
# IMPORTANT: per project decision, this MVP classifier does NOT ignore
# Stage 3 Part 1's FacetInputContext just because it's zero-shot. It
# genuinely consumes three signals out of that context:
#
#   1. context.phrase            -> the NLI premise text (unavoidable;
#                                    an NLI model has nothing else to
#                                    reason over).
#   2. context.source_field's    -> a structural PRIOR looked up from
#      facet_bias (field_mappings    field_mappings.json (e.g.
#      .json, via DecomposedMetadata) Temporal_Meta -> "time"), blended
#                                    into the NLI scores.
#   3. cosine(span_vector,       -> a confidence SHARPENING multiplier:
#      global_vector)               the more thematically central a span
#                                    is to its document, the more we
#                                    trust the NLI's existing read on it
#                                    (this is NOT facet-specific -- it
#                                    does not push toward Personality by
#                                    name, since hardcoding "central ->
#                                    Personality" would just be a
#                                    disguised rule-based shortcut rather
#                                    than something the classifier
#                                    actually determined).
#
# What Part 1's raw vectors (span_vector, global_vector, structural_vector,
# fused_vector) are NOT used for: none of their raw float components are
# fed into the NLI model. They have no trained meaning yet (Stage 4/5's
# projection matrices and losses don't exist), so treating their literal
# values as classifier input would be indistinguishable from noise. What
# IS used is derived, interpretable information already computed during
# Part 1's assembly (which field a span came from; whether that field had
# a real graph node; how similar two of Part 1's vectors are to each
# other) -- these are legitimate structural facts, not raw embeddings.
# ---------------------------------------------------------------------------

from src.preprocessing.cleaner import DecomposedMetadata


PMEST_FACETS: tuple[str, ...] = ("personality", "matter", "energy", "space", "time")


@dataclass
class FacetDistribution:
    """Soft PMEST facet assignment for a single extracted keyword.

    Mirrors Stage 3's Facet Distribution vector: five probabilities that
    sum to 1.0. `primary_facet` is the Stage 3 Part 3 Hard Assignment
    (argmax over the distribution) -- see HardAssignment / soft_and_hard
    below for the explicit Part 3 step.
    """

    phrase: str
    personality: float
    matter: float
    energy: float
    space: float
    time: float

    def as_dict(self) -> dict[str, float]:
        return {
            "personality": self.personality,
            "matter": self.matter,
            "energy": self.energy,
            "space": self.space,
            "time": self.time,
        }

    def as_vector(self) -> list[float]:
        """Facet probabilities in the fixed P, M, E, S, T order (matches
        Stage 4's subspace ordering, for whenever Stage 4 is built)."""
        d = self.as_dict()
        return [d[facet] for facet in PMEST_FACETS]


class ZeroShotFacetClassifier:
    """Stage 3 Part 2: assigns PMEST facet probability distributions.

    MVP stand-in for the spec's trained "5-Way Gated Multi-Label
    Attention Layer" (Stage 3, item 1). Has no learned weights of its
    own -- wraps a pretrained NLI zero-shot-classification pipeline, per
    config/default_config.yaml's `facet_classification` block, and blends
    its output with the structural prior and centrality signal already
    computed by Stage 3 Part 1's FacetInputAssembler.
    """

    def __init__(self, config: dict, device: int = -1):
        """
        Args:
            config: the FULL loaded default_config.yaml dict (or an
                equivalent). Only the `facet_classification` key is read.
            device: -1 for CPU, or a CUDA device index.
        """
        fc_cfg = config["facet_classification"]

        if fc_cfg["method"] != "zero_shot_nli":
            raise ValueError(
                f"ZeroShotFacetClassifier only supports method='zero_shot_nli', "
                f"got {fc_cfg['method']!r}. A trained classifier method would "
                f"need a different class -- see Stage 5 training objectives."
            )

        self._pipeline = pipeline(
            "zero-shot-classification",
            model=fc_cfg["model"],
            device=device,
        )
        self._hypothesis_template = fc_cfg["hypothesis_template"]
        self._multi_label = fc_cfg["multi_label"]

        # label key (e.g. "space") -> label description text (e.g.
        # "a geographic, spatial, or architectural location"), in the
        # fixed PMEST_FACETS order.
        self._label_descriptions: dict[str, str] = fc_cfg["labels"]
        missing = set(PMEST_FACETS) - set(self._label_descriptions)
        if missing:
            raise ValueError(
                f"config facet_classification.labels is missing entries for: "
                f"{missing}"
            )

    def _nli_scores(self, phrase: str) -> list[float]:
        """Runs the zero-shot pipeline and returns scores in fixed
        PMEST_FACETS order (the pipeline itself returns them sorted by
        score descending, so we re-align by label text)."""
        candidate_labels = [self._label_descriptions[f] for f in PMEST_FACETS]

        result = self._pipeline(
            phrase,
            candidate_labels=candidate_labels,
            hypothesis_template=self._hypothesis_template,
            multi_label=self._multi_label,
        )
        label_to_score = dict(zip(result["labels"], result["scores"]))
        return [label_to_score[self._label_descriptions[f]] for f in PMEST_FACETS]

    def _structural_prior(
        self,
        context: FacetInputContext,
        decomposed: DecomposedMetadata,
    ) -> tuple[list[float] | None, float]:
        """Looks up context.source_field's facet_bias and, if one exists,
        returns a one-hot-ish prior vector plus a confidence weight for
        how much to trust it.

        Confidence gating (per project decision): the structural_vector
        Part 1 attached to this context is zero exactly when the source
        field never became a graph node (e.g. it's a purely textual field
        like Title/Abstract -- see FacetInputAssembler._structural_vector_for_span).
        A zero structural_vector means Stage 1's Channel B contributed NO
        real signal for this span, so even if facet_bias happens to be
        set for that field, we shouldn't lean on it as if it were backed
        by learned structural context. Concretely:
            structural_vector is non-zero -> confidence = 1.0 (full trust)
            structural_vector is all-zero -> confidence = 0.0 (ignored)
        This is deliberately a hard gate, not a graded one: "the field had
        a real graph node" is itself a binary fact, so a continuous
        confidence score here would just be inventing precision that
        doesn't exist yet.

        Returns:
            (prior_vector, confidence) where prior_vector is None if
            source_field has no facet_bias mapping at all (e.g. Author,
            Publisher -- both structural=null in field_mappings.json).
        """
        if context.source_field is None:
            return None, 0.0

        struct_attr = decomposed.get_structural_field(context.source_field)
        text_attr = decomposed.get_textual_field(context.source_field)
        facet_bias = None
        if struct_attr is not None:
            facet_bias = struct_attr.facet_bias
        elif text_attr is not None:
            facet_bias = text_attr.facet_bias

        if facet_bias is None:
            return None, 0.0

        if facet_bias not in PMEST_FACETS:
            raise ValueError(
                f"facet_bias {facet_bias!r} on field {context.source_field!r} "
                f"is not one of {PMEST_FACETS}"
            )

        prior_vector = [1.0 if f == facet_bias else 0.0 for f in PMEST_FACETS]

        structural_signal_is_real = not torch.allclose(
            context.structural_vector, torch.zeros_like(context.structural_vector)
        )
        confidence = 1.0 if structural_signal_is_real else 0.0

        return prior_vector, confidence

    def _centrality_multiplier(self, context: FacetInputContext) -> float:
        """Cosine similarity between this span's vector and the
        document's global vector, remapped from [-1, 1] to a sharpening
        multiplier.

        This is a general confidence sharpener, not facet-specific: it
        does not push probability toward any particular facet by name.
        It only controls how much we trust/sharpen whatever the NLI
        model already output, on the premise that a span highly similar
        to its document's overall representation is less likely to be an
        incidental/noisy mention and more likely to be classified with
        real confidence either way.

        Remapping: cosine similarity is typically small-positive for
        unrelated-but-not-opposed vectors in high dimensions, so we
        rescale (cos + 1) / 2 into [0, 1], then use it to interpolate
        the sharpening exponent between 0.5 (flatten toward uniform,
        for peripheral spans) and 2.0 (sharpen the existing distribution,
        for central spans). An exponent of 1.0 (no change) sits at the
        midpoint, cos_sim == 0.
        """
        span = context.span_vector
        glob = context.global_vector
        cos_sim = torch.nn.functional.cosine_similarity(
            span.unsqueeze(0), glob.unsqueeze(0)
        ).item()

        normalized = (cos_sim + 1.0) / 2.0  # [0, 1]
        exponent = 0.5 + normalized * 1.5   # [0.5, 2.0]
        return exponent

    def classify(
        self,
        context: FacetInputContext,
        decomposed: DecomposedMetadata,
        structural_bias_weight: float = 0.4,
    ) -> FacetDistribution:
        """Classify one keyword's FacetInputContext into a PMEST facet
        distribution, genuinely using Part 1's context (not just the
        phrase string).

        Args:
            context: Part 1's FacetInputContext for this keyword.
            decomposed: the same DecomposedMetadata the context's document
                was built from (needed to look up facet_bias for
                context.source_field).
            structural_bias_weight: how much weight the structural prior
                receives when it exists AND is confidence-gated to 1.0,
                in [0, 1]. Has no effect when no prior exists or its
                confidence is 0 (see _structural_prior).

        Returns:
            A FacetDistribution with probabilities summing to 1.0.
        """
        nli_scores = self._nli_scores(context.phrase)

        # ---- centrality-based sharpening (signal 3) ----
        exponent = self._centrality_multiplier(context)
        sharpened = [max(s, 1e-9) ** exponent for s in nli_scores]
        total = sum(sharpened)
        probs = [s / total for s in sharpened]

        # ---- structural prior blending (signal 2) ----
        prior_vector, confidence = self._structural_prior(context, decomposed)
        effective_weight = structural_bias_weight * confidence

        if prior_vector is not None and effective_weight > 0.0:
            blended = [
                (1 - effective_weight) * p + effective_weight * b
                for p, b in zip(probs, prior_vector)
            ]
            blend_total = sum(blended)
            probs = [b / blend_total for b in blended]

        return FacetDistribution(
            phrase=context.phrase,
            personality=probs[0],
            matter=probs[1],
            energy=probs[2],
            space=probs[3],
            time=probs[4],
        )

    def classify_batch(
        self,
        contexts: list[FacetInputContext],
        decomposed: DecomposedMetadata,
        structural_bias_weight: float = 0.4,
    ) -> list[FacetDistribution]:
        """Classify all of Stage 3 Part 1's assembled contexts for one
        document. All contexts must come from the same document/
        DecomposedMetadata (their global_vector should already be
        identical, per Part 1's own consistency guarantee)."""
        return [
            self.classify(ctx, decomposed, structural_bias_weight)
            for ctx in contexts
        ]


# ---------------------------------------------------------------------------
# Stage 3, Part 3 — Soft vs. Hard Facet Assignment
# ---------------------------------------------------------------------------

@dataclass
class FacetAssignment:
    """Stage 3 Part 3's output for one keyword: both assignment modes
    exposed side by side, per spec section "Soft vs. Hard Facet
    Assignment" -- downstream code chooses whichever it needs rather
    than this module deciding for them.
    """

    phrase: str
    soft: FacetDistribution      # full probability vector, sums to 1.0
    hard: str                    # argmax facet name, e.g. "space"
    hard_confidence: float       # the winning facet's probability


def assign_facets(distribution: FacetDistribution) -> FacetAssignment:
    """Derives the Hard Assignment (single primary facet) from a Soft
    Assignment (full distribution), per spec: "A keyword can be assigned
    a single primary facet (Hard Assignment) or maintain a probability
    weight across multiple facets (Soft Assignment)."

    This is intentionally a pure function, not a method on
    FacetDistribution itself, so that Part 2's classifier stays
    decoupled from Part 3's assignment policy -- e.g. if a future
    version wants a confidence-thresholded hard assignment (falling back
    to "ambiguous" below some cutoff) instead of a plain argmax, that
    policy change lives here without touching Part 2 at all.
    """
    probs = distribution.as_dict()
    hard_facet = max(probs, key=probs.get)

    return FacetAssignment(
        phrase=distribution.phrase,
        soft=distribution,
        hard=hard_facet,
        hard_confidence=probs[hard_facet],
    )


def assign_facets_batch(
    distributions: list[FacetDistribution],
) -> list[FacetAssignment]:
    return [assign_facets(d) for d in distributions]