
"""Promotion rules for PredictMaint AI."""

import math

# Quality gates documented before v2 evaluation.
MIN_AP = 0.40
MIN_AP_IMPROVEMENT = 0.02
MIN_RECALL = 0.50


def evaluate_promotion(
    champion_ap: float,
    candidate_ap: float,
    candidate_recall: float,
) -> dict:
    """Decide whether a candidate can replace the champion."""

    metrics = {
        "champion_ap": champion_ap,
        "candidate_ap": candidate_ap,
        "candidate_recall": candidate_recall,
    }

    # Reject invalid metric values.
    for name, value in metrics.items():
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid metric: {name}={value}")

    improvement = candidate_ap - champion_ap

    # All three conditions must pass.
    checks = {
        "minimum_average_precision": candidate_ap >= MIN_AP,
        "minimum_improvement": improvement >= MIN_AP_IMPROVEMENT,
        "minimum_recall": candidate_recall >= MIN_RECALL,
    }

    return {
        "approved": all(checks.values()),
        "ap_improvement": improvement,
        "checks": checks,
        "failed_gates": [
            name for name, passed in checks.items()
            if not passed
        ],
    }