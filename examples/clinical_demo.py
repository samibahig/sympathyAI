"""
clinical_demo.py — SympathyAI end-to-end demonstration.

Shows:
  1. Scoring individual responses.
  2. Re-ranking candidate responses to select the best.
  3. Running the full benchmark across the curated clinical dataset.

Run from the repo root:
    python examples/clinical_demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sympathy import evaluate, rerank
from sympathy.datasets import load_cases
from sympathy.evaluation import run_benchmark, print_benchmark_table


# ─────────────────────────────────────────────────────────────────────────────
# 1. Score a single response
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("DEMO 1 — Scoring individual responses")
print("=" * 60)

responses = [
    "This is likely stress or anxiety — unclear diagnosis.",
    "Symptoms are consistent with major depressive disorder, "
    "meeting DSM-5 criteria: depressed mood, anhedonia, and "
    "insomnia persisting for more than two weeks.",
]
keywords = ["depression", "DSM", "criteria", "anhedonia"]

for resp in responses:
    score = evaluate(resp, keywords)
    print(f"\nResponse : {resp[:70]}...")
    print(f"  Coherence  : {score.coherence:.2f}")
    print(f"  Guideline  : {score.guideline:.2f}")
    print(f"  Causality  : {score.causality:.2f}")
    print(f"  ▶ Total    : {score.total:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Re-rank candidate responses
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("DEMO 2 — Re-ranking 3 candidate responses")
print("=" * 60)

candidates = [
    "The patient might be stressed. It is unclear.",
    "However, there is no definitive explanation for the symptoms.",
    "Chest pain radiating to the left arm suggests acute MI; "
    "ECG and troponin are urgently indicated.",
]

best = rerank(
    candidates,
    keywords=["myocardial infarction", "ECG", "troponin"],
    symptoms=["chest pain", "radiation to left arm"],
    diagnosis_terms=["myocardial infarction", "MI"],
)

print(f"\nSelected response:\n  → {best.text}")
print(f"  Score: {best.total:.2f}  "
      f"(coherence={best.coherence}, guideline={best.guideline}, causality={best.causality})")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Full benchmark
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("DEMO 3 — Full benchmark on curated clinical dataset")
print("=" * 60)

cases   = load_cases()
results = run_benchmark(cases)

print()
print_benchmark_table(results)

correct = sum(1 for r in results if r["rerank_selects_correct"])
print(f"\nRe-ranking accuracy: {correct}/{len(results)} ({correct/len(results)*100:.0f}%)")
avg_delta = sum(r["score_delta"] for r in results) / len(results)
print(f"Average score delta (good − poor): {avg_delta:+.3f}")
