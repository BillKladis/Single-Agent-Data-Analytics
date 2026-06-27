"""Scoring metrics for the agent comparison."""
from __future__ import annotations


def tool_scores(predicted: list[str], expected: set[str]) -> dict[str, float]:
    """Precision/recall/F1 of the *set* of tools used vs the expected set.

    Recall = did the agent reach for the right primitive(s)?
    Precision = how much of its tool use was on-target (penalizes flailing)?
    """
    pred_set = set(predicted)
    if not expected:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(pred_set & expected)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(expected)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def aggregate(rows: list[dict]) -> dict[str, float]:
    """Mean of numeric per-task records for one strategy."""
    n = len(rows) or 1
    keys = ["correct", "tool_f1", "tool_recall", "tool_precision",
            "total_tokens", "latency_s", "n_llm_calls", "n_tool_calls"]
    out = {}
    for k in keys:
        out[k] = sum(r.get(k, 0) for r in rows) / n
    out["n_tasks"] = len(rows)
    out["accuracy"] = out["correct"]  # alias
    return out
