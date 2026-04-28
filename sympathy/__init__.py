"""
SympathyAI — Structure-aware evaluation and alignment for LLM reasoning.

Quick start:
    from sympathy import evaluate, rerank

    score = evaluate(response, keywords)
    best  = rerank(responses, keywords)
"""

from sympathy.scoring.coherence import coherence_score
from sympathy.scoring.guidelines import guideline_score
from sympathy.scoring.causality import causality_score
from sympathy.rerank.reranker import sympathetic_score, rerank
from sympathy.evaluation.evaluator import evaluate

__version__ = "0.1.0"
__all__ = [
    "coherence_score",
    "guideline_score",
    "causality_score",
    "sympathetic_score",
    "rerank",
    "evaluate",
]
