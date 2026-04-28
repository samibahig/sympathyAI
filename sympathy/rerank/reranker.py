"""
reranker.py — Sympathetic re-ranking of LLM candidate responses.

Instead of selecting the highest-likelihood output, SympathyAI selects
the response that best satisfies structural reasoning constraints:
coherence, guideline adherence, and causal alignment.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from sympathy.scoring.coherence import coherence_score
from sympathy.scoring.guidelines import guideline_score
from sympathy.scoring.causality import causality_score


@dataclass
class ScoredResponse:
    """A candidate response annotated with its component scores."""

    text: str
    coherence: float = 0.0
    guideline: float = 0.0
    causality: float = 0.0
    total: float = 0.0

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "scores": {
                "coherence": self.coherence,
                "guideline": self.guideline,
                "causality": self.causality,
                "total": self.total,
            },
        }


def sympathetic_score(
    response: str,
    keywords: List[str],
    symptoms: Optional[List[str]] = None,
    diagnosis_terms: Optional[List[str]] = None,
    w_coherence: float = 0.4,
    w_guideline: float = 0.4,
    w_causality: float = 0.2,
) -> ScoredResponse:
    """
    Compute a composite structure-aware score for a single response.

    Args:
        response:        Candidate LLM response.
        keywords:        Required guideline / clinical concept terms.
        symptoms:        Symptom terms from the case (for causality).
        diagnosis_terms: Diagnosis terms expected (for causality).
        w_coherence:     Weight for coherence component (default 0.4).
        w_guideline:     Weight for guideline component (default 0.4).
        w_causality:     Weight for causality component (default 0.2).

    Returns:
        ScoredResponse with all component scores populated.
    """
    c_score = coherence_score(response)
    g_score = guideline_score(response, keywords)
    ca_score = causality_score(
        response,
        symptoms or keywords,
        diagnosis_terms or keywords,
    )

    total = (
        w_coherence * c_score
        + w_guideline * g_score
        + w_causality * ca_score
    )

    return ScoredResponse(
        text=response,
        coherence=round(c_score, 4),
        guideline=round(g_score, 4),
        causality=round(ca_score, 4),
        total=round(total, 4),
    )


def rerank(
    responses: List[str],
    keywords: List[str],
    symptoms: Optional[List[str]] = None,
    diagnosis_terms: Optional[List[str]] = None,
    return_all: bool = False,
) -> ScoredResponse | List[ScoredResponse]:
    """
    Re-rank candidate responses by sympathetic score and return the best.

    Args:
        responses:       List of candidate LLM-generated responses.
        keywords:        Required clinical keyword set.
        symptoms:        Optional symptom list for causality scoring.
        diagnosis_terms: Optional diagnosis terms for causality scoring.
        return_all:      If True, return all responses ranked by score.

    Returns:
        Best ScoredResponse, or ranked list if return_all=True.

    Example:
        >>> responses = [
        ...     "This is unclear, possibly stress.",
        ...     "Symptoms suggest MDD; DSM-5 criteria are met.",
        ... ]
        >>> best = rerank(responses, keywords=["depression", "DSM", "criteria"])
        >>> best.text
        'Symptoms suggest MDD; DSM-5 criteria are met.'
    """
    if not responses:
        raise ValueError("responses list must not be empty.")

    scored = [
        sympathetic_score(r, keywords, symptoms, diagnosis_terms)
        for r in responses
    ]
    ranked = sorted(scored, key=lambda x: x.total, reverse=True)

    return ranked if return_all else ranked[0]
