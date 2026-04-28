"""
coherence.py — Heuristic + LLM-based coherence scoring.

A coherent clinical response:
  - Does not contradict itself.
  - Reaches a clear conclusion (avoids "unclear" hedging without reasoning).
  - Uses causal connectors appropriately (e.g. "therefore", "because").
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Heuristic coherence (fast, no external dependencies)
# ---------------------------------------------------------------------------

CONTRADICTION_PAIRS = [
    ("rule out", "confirmed"),
    ("no evidence", "strongly suggests"),
    ("not indicated", "is indicated"),
    ("no evidence of", "confirmed by"),
]

VAGUE_TOKENS = ["unclear", "uncertain", "unknown", "maybe", "possibly"]
REASONING_TOKENS = ["therefore", "because", "thus", "consequently", "since"]


def coherence_score(response: str) -> float:
    """
    Lightweight heuristic coherence score in [0, 1].

    Penalises:
      - Vague language without supporting reasoning connectors.
      - Surface-level contradictions between adjacent claims.

    Args:
        response: The model-generated clinical response string.

    Returns:
        A float in [0.0, 1.0], where 1.0 = fully coherent.

    Example:
        >>> coherence_score("Symptoms suggest MDD; therefore DSM-5 criteria are met.")
        1.0
        >>> coherence_score("This is unclear, no further reasoning.")
        0.5
    """
    score = 1.0
    text = response.lower()

    # Penalise vague language not backed by reasoning
    has_vague = any(tok in text for tok in VAGUE_TOKENS)
    has_reasoning = any(tok in text for tok in REASONING_TOKENS)

    if has_vague and not has_reasoning:
        score -= 0.3

    # Penalise "however" used without a concluding connector
    if "however" in text and not has_reasoning:
        score -= 0.2

    # Penalise explicit contradiction pairs
    for (neg, pos) in CONTRADICTION_PAIRS:
        if neg in text and pos in text:
            score -= 0.3
            break  # one contradiction is enough

    return max(round(score, 4), 0.0)


# ---------------------------------------------------------------------------
# LLM-based coherence (higher quality, requires API key)
# ---------------------------------------------------------------------------

def llm_coherence_score(
    response: str,
    model: str = "gpt-4o-mini",
    client=None,
) -> float:
    """
    Use an LLM-as-judge to rate internal coherence on a 0–1 scale.

    Args:
        response: The clinical response to evaluate.
        model: OpenAI-compatible model identifier.
        client: An initialised ``openai.OpenAI`` client (optional).
                If None, falls back to heuristic coherence_score.

    Returns:
        Float in [0.0, 1.0].
    """
    if client is None:
        return coherence_score(response)

    prompt = (
        "You are a clinical AI evaluator. Rate the INTERNAL COHERENCE of the "
        "following clinical response on a scale from 0.0 to 1.0.\n\n"
        "Criteria:\n"
        "  - Does it contradict itself?\n"
        "  - Does it reach a clear, reasoned conclusion?\n"
        "  - Are causal connectors used correctly?\n\n"
        f"Response:\n{response}\n\n"
        "Reply with a single float between 0.0 and 1.0, nothing else."
    )

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        raw = completion.choices[0].message.content.strip()
        return max(0.0, min(1.0, float(raw)))
    except Exception:
        # Graceful degradation
        return coherence_score(response)
