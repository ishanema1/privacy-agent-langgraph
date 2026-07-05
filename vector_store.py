"""
vector_store.py

Prior-assessment retrieval for the agentic privacy assessment pipeline.

Implements the "prior assessment search" step of the pipeline: a new
customer use case is embedded and matched against a store of past
assessments. A high-similarity match lets the agent reuse prior analysis
instead of re-running a full attacker-model evaluation from scratch.

Design notes (worth reading before you judge the FAISS-less approach):
  - At this corpus size (dozens to low hundreds of past assessments), a
    brute-force cosine similarity search over an in-memory numpy matrix
    is faster to reason about, easier to test, and has zero index-build
    overhead compared to an ANN index like FAISS. The `EmbeddingIndex`
    interface below is intentionally narrow (add / search) so it can be
    swapped for a FAISS or managed vector-DB backend without touching
    any calling code, if/when corpus size ever justifies it.
  - Embeddings are produced by a pluggable `Embedder`. The default tries
    to use a local sentence-transformers model; if that dependency isn't
    installed, it falls back to a lightweight hashing-based embedding so
    this module still runs end-to-end with zero external dependencies.
    The fallback is for demo/portability only — swap in the real model
    for anything beyond a smoke test.

This is a sanitized, standalone reference implementation for a public
portfolio case study. No real customer data, contracts, or internal
infrastructure details are used or represented here.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np

# Cosine similarity above which we treat a prior assessment as reusable,
# short-circuiting a fresh attacker-model run.
REUSE_SIMILARITY_THRESHOLD = 0.70


# --------------------------------------------------------------------------
# Embedding backend
# --------------------------------------------------------------------------

class Embedder(Protocol):
    """Anything that turns text into a fixed-length, L2-normalized vector."""

    def embed(self, text: str) -> np.ndarray: ...
    @property
    def dim(self) -> int: ...


class SentenceTransformerEmbedder:
    """Local, open embedding model. Requires `pip install sentence-transformers`."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # deferred import
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        vec = self._model.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype="float32")


class HashingEmbedder:
    """
    Dependency-free fallback embedder for demos and offline testing.

    Bag-of-words hashed into a fixed-size vector, L2-normalized so cosine
    similarity behaves the same way it would for a real embedding model.
    This is NOT a substitute for a trained embedding model in production —
    it exists so this file runs with zero external dependencies.
    """

    def __init__(self, dim: int = 256):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype="float32")
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for tok in tokens:
            idx = hash(tok) % self._dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


def default_embedder() -> Embedder:
    """Prefer a real sentence-transformer model; fall back gracefully."""
    try:
        return SentenceTransformerEmbedder()
    except ImportError:
        return HashingEmbedder()


# --------------------------------------------------------------------------
# Assessment records + store
# --------------------------------------------------------------------------

@dataclass
class Assessment:
    """A single past (or pending) privacy assessment."""

    id: str
    customer_ref: str        # anonymized reference, e.g. "cust_014" — never a real customer name
    use_case_summary: str    # short natural-language description of the data-sharing use case
    risk_level: str          # "low" | "medium" | "high"
    risk_score: Optional[float] = None  # numeric attacker-model score, if this was a scored assessment
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def to_metadata(self) -> dict:
        return {
            "id": self.id,
            "customer_ref": self.customer_ref,
            "use_case_summary": self.use_case_summary,
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
        }


class PriorAssessmentStore:
    """
    In-memory vector store of past privacy assessments.

    Used to implement the pipeline's routing decision: reuse a prior
    assessment when a sufficiently similar case already exists, otherwise
    signal that a fresh attacker-model evaluation is required.
    """

    def __init__(self, embedder: Optional[Embedder] = None):
        self._embedder = embedder or default_embedder()
        self._assessments: list[Assessment] = []
        self._matrix: Optional[np.ndarray] = None  # (n, dim) matrix of stacked embeddings

    def add_assessment(
        self, customer_ref: str, use_case_summary: str, risk_level: str, risk_score: Optional[float] = None
    ) -> Assessment:
        """Embed and index a completed assessment for future retrieval."""
        embedding = self._embedder.embed(use_case_summary)
        assessment = Assessment(
            id=str(uuid.uuid4()),
            customer_ref=customer_ref,
            use_case_summary=use_case_summary,
            risk_level=risk_level,
            risk_score=risk_score,
            embedding=embedding,
        )
        self._assessments.append(assessment)
        self._matrix = (
            embedding.reshape(1, -1)
            if self._matrix is None
            else np.vstack([self._matrix, embedding])
        )
        return assessment

    def search(self, use_case_summary: str, top_k: int = 3) -> list[tuple[Assessment, float]]:
        """
        Return the `top_k` most similar prior assessments for a new use case,
        as (assessment, cosine_similarity) pairs sorted by similarity descending.
        """
        if not self._assessments:
            return []

        query = self._embedder.embed(use_case_summary)
        # Embeddings are L2-normalized, so a dot product IS cosine similarity.
        scores = self._matrix @ query
        top_k = min(top_k, len(self._assessments))
        top_indices = np.argsort(-scores)[:top_k]

        return [(self._assessments[i], float(scores[i])) for i in top_indices]

    def find_reusable_assessment(self, use_case_summary: str) -> Optional[tuple[Assessment, float]]:
        """
        Implements the pipeline's step-3 routing decision: if the closest
        prior assessment clears REUSE_SIMILARITY_THRESHOLD, return it so the
        agent can skip a fresh attacker-model run. Otherwise return None.
        """
        results = self.search(use_case_summary, top_k=1)
        if not results:
            return None

        best_match, score = results[0]
        return (best_match, score) if score >= REUSE_SIMILARITY_THRESHOLD else None


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------

def _demo() -> None:
    """End-to-end demo using synthetic, non-confidential examples."""
    store = PriorAssessmentStore()

    store.add_assessment(
        customer_ref="cust_001",
        use_case_summary="Automotive OEM requesting anonymized vehicle trajectory data "
                          "for fleet route-optimization analytics, EU-only vehicles.",
        risk_level="medium",
    )
    store.add_assessment(
        customer_ref="cust_002",
        use_case_summary="Insurance provider requesting aggregated driving-behavior scores "
                          "for usage-based insurance pricing, no raw trajectories.",
        risk_level="low",
    )
    store.add_assessment(
        customer_ref="cust_003",
        use_case_summary="Industrial logistics company requesting anonymized delivery-route "
                          "data for supply-chain congestion modeling.",
        risk_level="medium",
    )

    new_use_case = (
        "Automotive manufacturer requesting anonymized trip trajectory data "
        "for European fleet vehicles to optimize routing."
    )

    print(f"embedder in use: {type(store._embedder).__name__}\n")

    match = store.find_reusable_assessment(new_use_case)
    if match:
        assessment, score = match
        print(f"Reusable prior assessment found (similarity={score:.3f}):")
        print(f"  customer_ref : {assessment.customer_ref}")
        print(f"  use case     : {assessment.use_case_summary}")
        print(f"  risk_level   : {assessment.risk_level}")
        print("  -> skipping attacker-model re-run, drafting from precedent")
    else:
        print("No sufficiently similar prior assessment found.")
        print("  -> triggering full attacker-model evaluation")

    print("\nTop 3 matches, for reference:")
    for assessment, score in store.search(new_use_case, top_k=3):
        print(f"  [{score:.3f}] {assessment.customer_ref}: {assessment.use_case_summary}")


if __name__ == "__main__":
    _demo()
