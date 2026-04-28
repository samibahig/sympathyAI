"""
causality.py — Symptom-to-diagnosis causal alignment scoring.

A causally aligned response explicitly links presented symptoms to
a proposed diagnosis using recognisable causal language.
"""

from __future__ import annotations
from typing import List

CAUSAL_CONNECTORS = [
    "because", "therefore", "thus", "consequently", "suggests",
    "consistent with", "indicative of", "due to", "caused by",
    "supports", "confirms", "is consistent",
]


def causality_score(
    response: str,
    symptoms: List[str],
    diagnosis_terms: List[str],
) -> float:
    """
    Estimate how well a response causally links symptoms to a diagnosis.

    Scoring logic:
      - +0.4  if at least one symptom is mentioned.
      - +0.4  if at least one diagnosis term is mentioned.
      - +0.2  if a causal connector is present alongside both.

    Args:
        response:        The model-generated clinical response.
        symptoms:        Key symptom terms from the clinical case.
        diagnosis_terms: Expected diagnostic labels / concepts.

    Returns:
        Float in [0.0, 1.0].

    Example:
        >>> causality_score(
        ...     "Persistent sadness and insomnia suggest major depressive disorder.",
        ...     symptoms=["sadness", "insomnia"],
        ...     diagnosis_terms=["major depressive disorder"],
        ... )
        1.0
    """
    text = response.lower()

    symptom_hit = any(s.lower() in text for s in symptoms)
    diagnosis_hit = any(d.lower() in text for d in diagnosis_terms)
    connector_hit = any(c in text for c in CAUSAL_CONNECTORS)

    score = 0.0
    if symptom_hit:
        score += 0.4
    if diagnosis_hit:
        score += 0.4
    if connector_hit and symptom_hit and diagnosis_hit:
        score += 0.2

    return round(score, 4)
