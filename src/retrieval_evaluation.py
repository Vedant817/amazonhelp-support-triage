"""Metrics for held-out historical resolution-act retrieval."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


def ranked_act_metrics(
    expected_acts: Sequence[str],
    ranked_neighbors: Sequence[Sequence[Dict[str, Any]]],
    k: int = 3,
) -> Dict[str, Any]:
    if len(expected_acts) != len(ranked_neighbors) or not expected_acts:
        raise ValueError("Expected acts and neighbor rankings must be non-empty and aligned")
    recalls = []
    reciprocal_ranks = []
    by_act: Dict[str, list[bool]] = {}
    for expected, neighbors in zip(expected_acts, ranked_neighbors):
        acts = [str(item.get("resolution_act", "")) for item in neighbors[:k]]
        matching = [index + 1 for index, act in enumerate(acts) if act == expected]
        recalls.append(bool(matching))
        by_act.setdefault(str(expected), []).append(bool(matching))
        reciprocal_ranks.append(1.0 / matching[0] if matching else 0.0)
    per_act = {
        act: {"recall_at_k": round(float(np.mean(values)), 4), "support": len(values)}
        for act, values in sorted(by_act.items())
    }
    return {
        f"recall_at_{k}": round(float(np.mean(recalls)), 4),
        f"macro_recall_at_{k}": round(float(np.mean([
            item["recall_at_k"] for item in per_act.values()
        ])), 4),
        f"mrr_at_{k}": round(float(np.mean(reciprocal_ranks)), 4),
        "query_count": len(expected_acts),
        "per_act": per_act,
    }
