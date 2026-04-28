"""
test_sympathy.py — Unit tests for the SympathyAI scoring and re-ranking pipeline.

Run with:
    pytest tests/test_sympathy.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from sympathy.scoring.coherence import coherence_score
from sympathy.scoring.guidelines import guideline_score
from sympathy.scoring.causality import causality_score
from sympathy.rerank.reranker import sympathetic_score, rerank
from sympathy.datasets import load_cases
from sympathy.evaluation import run_benchmark


# ─────────────────────────────────────────────────────────────────────────────
# Coherence
# ─────────────────────────────────────────────────────────────────────────────

class TestCoherenceScore:
    def test_clear_response_scores_high(self):
        r = "Symptoms suggest MDD; therefore DSM-5 criteria are met."
        assert coherence_score(r) == 1.0

    def test_vague_response_penalised(self):
        r = "This is unclear and uncertain."
        assert coherence_score(r) < 1.0

    def test_vague_with_reasoning_not_penalised(self):
        r = "The diagnosis is uncertain because multiple conditions overlap."
        score = coherence_score(r)
        assert score >= 0.8

    def test_contradiction_penalised(self):
        r = "There is no evidence of infection, which is confirmed by labs."
        assert coherence_score(r) < 1.0

    def test_score_bounded(self):
        for text in ["", "unclear unclear unclear", "however however however"]:
            s = coherence_score(text)
            assert 0.0 <= s <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Guideline
# ─────────────────────────────────────────────────────────────────────────────

class TestGuidelineScore:
    def test_all_keywords_present(self):
        r = "The patient meets DSM-5 criteria for depression."
        kw = ["depression", "DSM", "criteria"]
        assert guideline_score(r, kw) == 1.0

    def test_no_keywords_present(self):
        r = "The patient seems fine."
        kw = ["depression", "DSM", "criteria"]
        assert guideline_score(r, kw) == 0.0

    def test_partial_keywords(self):
        r = "Depression is suspected."
        kw = ["depression", "DSM", "criteria"]
        assert guideline_score(r, kw) == pytest.approx(1 / 3, abs=0.01)

    def test_empty_keywords_returns_one(self):
        assert guideline_score("any response", []) == 1.0

    def test_case_insensitive(self):
        r = "DEPRESSION and DSM and CRITERIA are discussed."
        assert guideline_score(r, ["depression", "dsm", "criteria"]) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Causality
# ─────────────────────────────────────────────────────────────────────────────

class TestCausalityScore:
    def test_full_causal_response(self):
        r = "Persistent sadness and insomnia suggest major depressive disorder."
        score = causality_score(r, ["sadness", "insomnia"], ["major depressive disorder"])
        assert score >= 0.8

    def test_no_symptoms_no_diagnosis(self):
        r = "The patient is okay."
        score = causality_score(r, ["sadness"], ["depression"])
        assert score == 0.0

    def test_only_symptom(self):
        r = "The patient reports sadness."
        score = causality_score(r, ["sadness"], ["depression"])
        assert score == pytest.approx(0.4, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# Re-ranker
# ─────────────────────────────────────────────────────────────────────────────

class TestReranker:
    def test_selects_best_response(self):
        responses = [
            "This is unclear, possibly stress.",
            "Symptoms suggest MDD; DSM-5 criteria are met.",
        ]
        best = rerank(responses, ["depression", "DSM", "criteria"])
        assert "DSM" in best.text

    def test_return_all_is_sorted(self):
        responses = [
            "This is unclear.",
            "Symptoms suggest MDD; DSM-5 criteria are met.",
        ]
        ranked = rerank(responses, ["depression", "DSM"], return_all=True)
        assert ranked[0].total >= ranked[-1].total

    def test_empty_responses_raises(self):
        with pytest.raises(ValueError):
            rerank([], ["keyword"])

    def test_single_response_returned(self):
        best = rerank(["Only one response."], ["keyword"])
        assert best.text == "Only one response."


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmark:
    def test_benchmark_runs(self):
        cases = load_cases()
        results = run_benchmark(cases)
        assert len(results) == len(cases)

    def test_good_scores_higher_than_poor(self):
        cases = load_cases()
        results = run_benchmark(cases)
        for r in results:
            assert r["good_response_score"] > r["poor_response_score"], (
                f"Case {r['case_id']}: good score should exceed poor score."
            )

    def test_rerank_accuracy_perfect(self):
        cases = load_cases()
        results = run_benchmark(cases)
        accuracy = sum(1 for r in results if r["rerank_selects_correct"]) / len(results)
        assert accuracy == 1.0, f"Expected 100% re-rank accuracy, got {accuracy:.0%}"
