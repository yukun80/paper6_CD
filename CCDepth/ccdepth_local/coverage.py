"""Opt-in evidence-based eligibility; numerical weights remain unchanged."""
import numpy as np

WEAK_BOUNDARY_QA = 1024


def eligibility(beta, rows, cols, p, mode='baseline_v1'):
    if mode not in ('baseline_v1', 'weak_boundary_v1'):
        raise ValueError('Unknown coverage mode')
    high = beta >= p.highWeight
    original = bool(high.sum() >= p.minAnchors and
                    max(np.ptp(rows[high]), np.ptp(cols[high])) >= p.minSpanPixels)
    # Positive beta already requires valid wet/dry samples, valid slope and
    # a non-inverted interval. Never manufacture anchors or alter their weights.
    weak = bool(not original and mode == 'weak_boundary_v1' and
                np.any(np.isfinite(beta) & (beta > 0)))
    return original or weak, weak
