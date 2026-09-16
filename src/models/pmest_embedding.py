"""PMEST embedding model component (Stage 4).

Stage 4: Multi-Facet Vector Embedding Formulation (PMEST Vector Space).

Part 1: Vector Subspace Partitioning -- reads the dimension allocation
        (160/96/128/64/64 = 512) from config/default_config.yaml's
        `vector_space` block and exposes it as a typed, validated
        structure, so no other module hardcodes these numbers.

Part 2: Subspace Projection Networks + Gated Facet Integration -- maps
        a keyword's raw candidate vector K_raw into five facet subspaces
        and gates each by its Stage 3 facet probability, producing the
        final 512-D Disentangled PMEST Vector.

MVP NOTE ON W_P...W_T: per the architecture spec (Stage 4, item 2) these
are meant to be TRAINED linear projection matrices, and per Stage 5 they
are literally what the orthogonality loss (L_ortho) trains. No training
loop exists yet (see src/pipelines/train.py, still a stub), so real
learned nn.Linear layers here would produce numerically meaningless
(random-init) output -- no better than noise, and worse than noise in
one respect: it would LOOK like a trained result to anyone inspecting
values without reading this docstring.

Per project decision, this MVP instead uses a DETERMINISTIC placeholder:
each facet claims a fixed, contiguous, non-overlapping slice of K_raw
sized to that facet's dimension (see _build_facet_slices). This has no
learned parameters, is fully traceable (you can always name exactly
which input dimensions produced a given facet's output), and requires
no random seed to reproduce. It is explicitly NOT a claim that slicing
is semantically meaningful -- dimension i of K_raw was never trained to
carry "Personality-relevant" information any more than dimension i+200
was. It is a shape-correct, deterministic stand-in occupying the exact
seam where trained nn.Linear(K_raw_dim, facet_dim) layers plug in later
(see PMESTSubspaceProjector.use_trained_projections, currently always
False) -- swapping in real weights is a change to HOW each facet's
output is computed, not WHERE, since input/output shapes are identical
either way.

K_raw, in this pipeline, is Stage 3 Part 1's `fused_vector`
(FacetInputContext.fused_vector, 3*hidden_dim) -- NOT Stage 2's
combined_vector. Reasoning: Stage 3's own spec identifies global
document context and structural field context as necessary to
disambiguate a keyword's facet (the "Glass" example), and Stage 2's
combined_vector carries neither -- it describes the span in isolation.
Using fused_vector also means that once Stage 4 is trained end-to-end,
gradients through W_P...W_T flow back into fused_vector's components,
which in turn touches Stage 1's encoder weights -- letting the whole
pipeline be fine-tuned toward facet-relevant representations, which a
combined_vector-only design would not allow.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.facet_classifier import FacetDistribution, PMEST_FACETS


# ----------------------------------------------------------------------------
# Stage 4, Part 1 -- Vector Subspace Partitioning
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class PMESTSubspaceLayout:
    """Validated dimension allocation for the five PMEST facet subspaces,
    read from config/default_config.yaml's `vector_space` block rather
    than hardcoded, so a config change propagates automatically.

    Facet order is always PMEST_FACETS (personality, matter, energy,
    space, time) -- this fixes both the concatenation order in
    Stage 4 Part 2's output vector AND the slice order in the
    deterministic K_raw placeholder (see _build_facet_slices).
    """

    dims: dict[str, int]     # facet name -> dimension, in PMEST_FACETS order
    total_dim: int

    @classmethod
    def from_config(cls, config: dict) -> "PMESTSubspaceLayout":
        vs_cfg = config["vector_space"]
        dims = {facet: vs_cfg["facets"][facet] for facet in PMEST_FACETS}

        computed_total = sum(dims.values())
        if computed_total != vs_cfg["total_dim"]:
            raise ValueError(
                f"vector_space.facets sums to {computed_total}, but "
                f"vector_space.total_dim declares {vs_cfg['total_dim']}. "
                f"Fix config/default_config.yaml."
            )

        return cls(dims=dims, total_dim=computed_total)

    def offsets(self) -> dict[str, tuple[int, int]]:
        """Facet name -> (start, end) OUTPUT offsets within the final
        512-D concatenated PMEST vector, in PMEST_FACETS order. Distinct
        from the INPUT slice offsets into K_raw (see
        PMESTSubspaceProjector._build_facet_slices) -- these two offset
        sets are unrelated to each other and happen to both exist only
        because Part 2's placeholder projection is itself a slice.
        """
        result = {}
        cursor = 0
        for facet in PMEST_FACETS:
            size = self.dims[facet]
            result[facet] = (cursor, cursor + size)
            cursor += size
        return result


# ----------------------------------------------------------------------------
# Stage 4, Part 2 -- Subspace Projection Networks + Gated Facet Integration
# ----------------------------------------------------------------------------

@dataclass
class PMESTVector:
    """One keyword's final Stage 4 output: the 512-D Disentangled PMEST
    Vector, plus its five ungated per-facet slices for inspection (e.g.
    to verify a purely-spatial keyword's Energy slice is near-zero, per
    the spec's "Sahara Desert" example).
    """

    phrase: str
    vector: torch.Tensor                  # (total_dim,), e.g. (512,)
    facet_slices: dict[str, torch.Tensor]  # facet -> its GATED slice, pre-concat
    facet_probs: dict[str, float]          # the Stage 3 probabilities used to gate


class PMESTSubspaceProjector(nn.Module):
    """Stage 4 Part 2: projects K_raw into five facet subspaces and gates
    each by its Stage 3 facet probability.

    use_trained_projections is a permanent False for the MVP -- see
    module docstring. It exists as an explicit flag (rather than the
    trained path simply not existing yet) so that swapping in real
    nn.Linear layers later is a matter of implementing the True branch
    and flipping this flag, not restructuring this class's interface.
    """

    def __init__(
        self,
        layout: PMESTSubspaceLayout,
        k_raw_dim: int,
        use_trained_projections: bool = False,
    ):
        super().__init__()
        self.layout = layout
        self.k_raw_dim = k_raw_dim
        self.use_trained_projections = use_trained_projections

        if use_trained_projections:
            # Seam for the real, trained version (not implemented in the
            # MVP): one nn.Linear per facet, input=k_raw_dim, output=that
            # facet's dimension. These are exactly W_P...W_T from the
            # spec, and exactly what Stage 5's L_ortho would regularize.
            self.projections = nn.ModuleDict(
                {
                    facet: nn.Linear(k_raw_dim, layout.dims[facet])
                    for facet in PMEST_FACETS
                }
            )
        else:
            self.projections = None
            self._input_slices = self._build_facet_slices(layout, k_raw_dim)

    def _build_facet_slices(
        self, layout: PMESTSubspaceLayout, k_raw_dim: int
    ) -> dict[str, tuple[int, int]]:
        """Deterministic placeholder input mapping: each facet claims the
        next contiguous block of K_raw, sized to that facet's output
        dimension, in PMEST_FACETS order. Requires k_raw_dim >= 512 (the
        current K_raw, Stage 3's fused_vector at hidden_dim=768, is
        2304-D, so no padding/repeating is needed -- every facet gets a
        real, distinct slice with room to spare).
        """
        if k_raw_dim < layout.total_dim:
            raise ValueError(
                f"Deterministic slicing placeholder requires k_raw_dim "
                f"({k_raw_dim}) >= vector_space.total_dim "
                f"({layout.total_dim}). Got a K_raw too small to slice "
                f"without overlap or padding -- either use a larger "
                f"K_raw source or implement a padding/repeat strategy."
            )

        slices = {}
        cursor = 0
        for facet in PMEST_FACETS:
            size = layout.dims[facet]
            slices[facet] = (cursor, cursor + size)
            cursor += size
        return slices

    def _project_facet(self, k_raw: torch.Tensor, facet: str) -> torch.Tensor:
        """Returns facet's UNGATED subspace vector V_facet, shape
        (layout.dims[facet],)."""
        if self.use_trained_projections:
            return self.projections[facet](k_raw)

        start, end = self._input_slices[facet]
        return k_raw[start:end]

    def forward(
        self,
        phrase: str,
        k_raw: torch.Tensor,
        facet_distribution: FacetDistribution,
    ) -> PMESTVector:
        """
        Args:
            phrase: the keyword text (carried through for traceability).
            k_raw: Stage 3 Part 1's fused_vector for this keyword,
                shape (k_raw_dim,).
            facet_distribution: Stage 3 Part 2/3's soft facet assignment
                for this SAME keyword -- P_prob, M_prob, E_prob, S_prob,
                T_prob, used as the gating scalars.

        Returns:
            A PMESTVector with the final concatenated 512-D vector.
        """
        if k_raw.shape[-1] != self.k_raw_dim:
            raise ValueError(
                f"k_raw has dim {k_raw.shape[-1]}, but this projector was "
                f"built for k_raw_dim={self.k_raw_dim}."
            )

        probs = facet_distribution.as_dict()

        gated_slices: dict[str, torch.Tensor] = {}
        for facet in PMEST_FACETS:
            v_facet = self._project_facet(k_raw, facet)       # ungated
            gated_slices[facet] = probs[facet] * v_facet       # gated, per spec

        pmest_vector = torch.cat([gated_slices[f] for f in PMEST_FACETS], dim=0)

        expected_dim = self.layout.total_dim
        if pmest_vector.shape[0] != expected_dim:
            raise RuntimeError(
                f"Assembled PMEST vector has dim {pmest_vector.shape[0]}, "
                f"expected {expected_dim}. This indicates a bug in facet "
                f"slice/projection sizing."
            )

        return PMESTVector(
            phrase=phrase,
            vector=pmest_vector,
            facet_slices=gated_slices,
            facet_probs=probs,
        )

    def forward_batch(
        self,
        phrases: list[str],
        k_raw_batch: list[torch.Tensor],
        facet_distributions: list[FacetDistribution],
    ) -> list[PMESTVector]:
        if not (len(phrases) == len(k_raw_batch) == len(facet_distributions)):
            raise ValueError(
                "phrases, k_raw_batch, and facet_distributions must all be "
                "the same length."
            )
        return [
            self.forward(p, k, fd)
            for p, k, fd in zip(phrases, k_raw_batch, facet_distributions)
        ]