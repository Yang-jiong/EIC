"""Vehicle-level aggregation of EIC window scores in the three-fleet evaluation.

Input: calibrated scores for one vehicle in ascending mileage order.
The caller supplies the tail percentage previously selected on validation vehicles.
Requires pandas. Run this module directly for a synthetic example.
"""

from math import isfinite

import pandas as pd


def vehicle_score(window_scores, tail_percent):
    """Average the highest-scoring fraction after centred 32-window smoothing."""
    if tail_percent not in range(5, 100, 5):
        raise ValueError("tail_percent must be 5, 10, ..., or 95")
    scores = pd.Series(list(window_scores), dtype="float64")
    if scores.empty or not all(isfinite(value) for value in scores):
        raise ValueError("Provide at least one score; all scores must be finite")
    smoothed = scores.rolling(window=32, center=True, min_periods=1).mean()
    count = max(1, len(scores) * int(tail_percent) // 100)
    return float(smoothed.nlargest(count).mean())


if __name__ == "__main__":
    # Forty synthetic window scores in mileage order, with a 5% tail.
    print(vehicle_score(range(40), tail_percent=5))  # 30.75
