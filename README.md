# SympathyAI

**Structure-aware evaluation and alignment for LLM reasoning**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![No dependencies](https://img.shields.io/badge/dependencies-none-brightgreen.svg)]()

---

## Motivation

Large language models optimised via likelihood maximisation produce *fluent* text — but fluency is not reasoning.

In clinical settings, a response can be grammatically perfect, statistically plausible, and still:

- contradict itself internally,
- ignore established diagnostic guidelines (DSM-5, NICE, AHA),
- fail to link symptoms causally to a diagnosis.

SympathyAI introduces a **sympathetic evaluation framework**: a lightweight, composable set of structure-aware metrics that go *beyond* perplexity to measure whether a model response actually *reasons* correctly.

> *Inspired by François Chollet's critique of gradient-based optimisation as a sufficient proxy for intelligence.*

---

## Three-Line Quick Start

```python
from sympathy import evaluate, rerank

# Score a single response
score = evaluate("Symptoms suggest MDD; DSM-5 criteria are met.", keywords=["depression", "DSM"])

# Select the best response from a candidate list
best = rerank(responses, keywords=["myocardial infarction", "ECG", "troponin"])
```

---

## What SympathyAI Measures

| Metric | What it captures | Fast heuristic | LLM-judge |
|---|---|:---:|:---:|
| **Coherence** | Internal consistency; absence of contradictions | ✅ | ✅ |
| **Guideline adherence** | Coverage of required clinical concepts | ✅ | — |
| **Causal alignment** | Symptom → diagnosis linkage | ✅ | — |

All three are combined into a single **sympathetic score**:

```
sympathetic_score = 0.4 × coherence + 0.4 × guideline + 0.2 × causality
```

Weights are configurable.

---

## Benchmark Results

Evaluated on 5 curated clinical vignettes (MDD, MI, Pneumonia, GAD, T2DM).
Each case has a clinically correct response and a plausible-but-flawed response.

| Case | Condition | Good ↑ | Poor ↓ | Δ | Re-rank ✓ |
|---|---|:---:|:---:|:---:|:---:|
| MDD_001 | Major Depressive Disorder | 0.92 | 0.28 | +0.64 | ✅ |
| MI_001 | Myocardial Infarction | 0.80 | 0.36 | +0.44 | ✅ |
| PNE_001 | Community-Acquired Pneumonia | 0.80 | 0.40 | +0.40 | ✅ |
| GAD_001 | Generalised Anxiety Disorder | 0.89 | 0.28 | +0.61 | ✅ |
| DM2_001 | Type 2 Diabetes Mellitus | 0.88 | 0.36 | +0.52 | ✅ |

**Re-ranking accuracy: 5/5 (100%) · Average score delta: +0.52**

> Re-ranking consistently selects the clinically correct response without any additional model training.

---

## Installation

```bash
git clone https://github.com/your-username/sympathyAI.git
cd sympathyAI
pip install -e .
```

No external dependencies required for the heuristic pipeline.

For LLM-as-judge coherence scoring:
```bash
pip install -e ".[llm]"
```

---

## Project Structure

```
sympathyAI/
│
├── sympathy/
│   ├── scoring/
│   │   ├── coherence.py      # Heuristic + LLM-as-judge coherence
│   │   ├── causality.py      # Symptom-to-diagnosis causal alignment
│   │   └── guidelines.py     # Clinical keyword coverage
│   │
│   ├── rerank/
│   │   └── reranker.py       # Composite scoring + re-ranking
│   │
│   ├── datasets/
│   │   └── clinical_cases.py # 5 curated clinical vignettes
│   │
│   └── evaluation/
│       └── evaluator.py      # Benchmark runner + table printer
│
├── examples/
│   └── clinical_demo.py      # End-to-end demonstration
│
└── tests/
    └── test_sympathy.py      # Full unit test suite
```

---

## Run the Demo

```bash
python examples/clinical_demo.py
```

Expected output:
```
Response : Symptoms are consistent with major depressive disorder...
  Coherence  : 1.00
  Guideline  : 0.75
  Causality  : 1.00
  ▶ Total    : 0.90

Re-ranking accuracy: 5/5 (100%)
Average score delta (good − poor): +0.523
```

---

## Run Tests

```bash
pytest tests/test_sympathy.py -v
```

---

## Roadmap

- [ ] LLM-as-judge coherence scoring (GPT-4 meta-evaluation)
- [ ] Clinical grounding with DSM-5 and NICE guidelines knowledge base
- [ ] Contradiction detection via NLI models
- [ ] Prompt perturbation robustness analysis
- [ ] REST API / Python wrapper for integration into inference pipelines
- [ ] Extended benchmark (50+ clinical cases with expert annotation)

---

## Citation

If you use SympathyAI in your research, please cite:

```bibtex
@misc{sympathyai2025,
  title  = {Beyond Likelihood: Evaluating Structure-Aware Reasoning in Clinical Language Models},
  author = {[Authors]},
  year   = {2025},
  note   = {https://github.com/your-username/sympathyAI}
}
```

---

## License

MIT — free to use, extend, and publish with.
