from typing import Optional

import numpy as np

# If fewer than this many historical signals, use absolute threshold instead of z-score
MIN_HISTORY_FOR_ZSCORE = 5
DEFAULT_ABSOLUTE_THRESHOLD = 1.0


def detect_shock(
    value: float,
    historical_values: list[float],
    threshold_sd: float = 2.0,
    absolute_threshold: float = DEFAULT_ABSOLUTE_THRESHOLD,
) -> tuple[Optional[float], bool]:
    """Detect whether a signal value constitutes a shock.

    Returns (z_score, is_shock). z_score is None if insufficient history.
    """
    if len(historical_values) < MIN_HISTORY_FOR_ZSCORE:
        return None, abs(value) > absolute_threshold

    mean = np.mean(historical_values)
    std = np.std(historical_values, ddof=1)

    if std == 0:
        return None, abs(value) > absolute_threshold

    z_score = (value - mean) / std
    is_shock = abs(z_score) > threshold_sd
    return float(z_score), is_shock
