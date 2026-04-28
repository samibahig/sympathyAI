"""
guidelines.py — Clinical guideline adherence scoring.

Measures whether a response mentions the domain concepts expected
for a given clinical case (e.g. DSM-5 criteria for depression,
troponin / ECG for suspected MI).
"""

from __future__ import annotations
from typing import List


def guideline_score(response: str, keywords: List[str]) -> float:
    """
    Keyword-coverage score in [0, 1].

    Checks how many of the expected clinical keywords appear in the response.
    Each missing keyword is penalised equally.

    Args:
        response: The model-generated clinical response.
        keywords: List of required clinical concepts / guideline terms.

    Returns:
        Fraction of keywords present: 0.0 (none) → 1.0 (all).

    Example:
        >>> guideline_score("DSM-5 criteria are met for depression.", ["depression", "DSM", "criteria"])
        1.0
        >>> guideline_score("Patient seems stressed.", ["depression", "DSM", "criteria"])
        0.0
    """
    if not keywords:
        return 1.0  # No guidelines to check → vacuously satisfied

    text = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text)
    return round(hits / len(keywords), 4)


# ---------------------------------------------------------------------------
# Predefined guideline keyword sets for common conditions
# ---------------------------------------------------------------------------

GUIDELINE_KEYWORDS = {
    "major_depressive_disorder": [
        "depression", "DSM", "criteria", "anhedonia", "depressed mood",
        "two weeks", "impairment",
    ],
    "myocardial_infarction": [
        "myocardial infarction", "ECG", "troponin", "chest pain",
        "ST elevation", "STEMI", "NSTEMI", "cardiac",
    ],
    "pneumonia": [
        "pneumonia", "infection", "antibiotics", "consolidation",
        "chest X-ray", "fever", "cough",
    ],
    "anxiety_disorder": [
        "anxiety", "GAD", "DSM", "worry", "six months", "impairment",
    ],
}


def get_guideline_keywords(condition: str) -> List[str]:
    """
    Return the keyword set for a known condition.

    Args:
        condition: One of the keys in GUIDELINE_KEYWORDS.

    Returns:
        List of keyword strings, or empty list if condition is unknown.
    """
    return GUIDELINE_KEYWORDS.get(condition.lower().replace(" ", "_"), [])
