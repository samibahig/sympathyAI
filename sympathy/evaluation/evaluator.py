"""
evaluator.py — High-level evaluation API for SympathyAI.

Provides the ``evaluate`` function for scoring a single response and
the ``run_benchmark`` function for evaluating across the full dataset.
"""

from __future__ import annotations
from typing import List, Optional
import json

from sympathy.rerank.reranker import sympathetic_score, rerank, ScoredResponse
from sympathy.datasets.clinical_cases import ClinicalCase


def evaluate(
    response: str,
    keywords: List[str],
    symptoms: Optional[List[str]] = None,
    diagnosis_terms: Optional[List[str]] = None,
) -> ScoredResponse:
    """
    Score a single response with the full sympathetic metric.

    Args:
        response:        LLM-generated clinical response.
        keywords:        Required clinical guideline keywords.
        symptoms:        Symptom terms (optional, improves causality score).
        diagnosis_terms: Diagnosis terms (optional, improves causality score).

    Returns:
        ScoredResponse with coherence, guideline, causality, and total scores.

    Example:
        >>> from sympathy import evaluate
        >>> r = evaluate(
        ...     "Symptoms suggest MDD; DSM-5 criteria are met.",
        ...     keywords=["depression", "DSM", "criteria"],
        ... )
        >>> r.total
        0.9
    """
    return sympathetic_score(response, keywords, symptoms, diagnosis_terms)


def run_benchmark(cases: List[ClinicalCase]) -> List[dict]:
    """
    Run the sympathetic evaluation benchmark over a list of ClinicalCase objects.

    For each case, both the good and poor reference responses are scored,
    and the re-ranker is applied to select the best among them.

    Args:
        cases: List of ClinicalCase objects (e.g., from load_cases()).

    Returns:
        List of result dicts with case metadata and scores.
    """
    results = []

    for case in cases:
        good_scored = sympathetic_score(
            case.good_response,
            case.keywords,
            case.symptoms,
            case.diagnosis_terms,
        )
        poor_scored = sympathetic_score(
            case.poor_response,
            case.keywords,
            case.symptoms,
            case.diagnosis_terms,
        )

        best = rerank(
            [case.good_response, case.poor_response],
            case.keywords,
            case.symptoms,
            case.diagnosis_terms,
        )
        rerank_correct = best.text == case.good_response

        results.append({
            "case_id": case.case_id,
            "condition": case.condition,
            "good_response_score": good_scored.total,
            "poor_response_score": poor_scored.total,
            "score_delta": round(good_scored.total - poor_scored.total, 4),
            "rerank_selects_correct": rerank_correct,
            "good_scores": good_scored.as_dict()["scores"],
            "poor_scores": poor_scored.as_dict()["scores"],
        })

    return results


def print_benchmark_table(results: List[dict]) -> None:
    """Pretty-print the benchmark results as a comparison table."""
    header = f"{'Case':<12} {'Condition':<30} {'Good':>6} {'Poor':>6} {'Delta':>7} {'Rerank ✓':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        mark = "✅" if r["rerank_selects_correct"] else "❌"
        print(
            f"{r['case_id']:<12} {r['condition']:<30} "
            f"{r['good_response_score']:>6.2f} {r['poor_response_score']:>6.2f} "
            f"{r['score_delta']:>+7.2f} {mark:>10}"
        )
