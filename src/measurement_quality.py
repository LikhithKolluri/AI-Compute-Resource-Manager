import math


def evaluate_measurement_quality(
    active_gpu_samples,
    monitoring_window_seconds,
    maximum_sample_gap_seconds,
    minimum_samples=10,
    maximum_gap_seconds=2.0,
):
    """
    Evaluate GPU monitoring coverage for a historical session.

    This does not guarantee that every GPU memory peak
    was captured.
    """

    values = (
        active_gpu_samples,
        monitoring_window_seconds,
        maximum_sample_gap_seconds,
    )

    if not all(math.isfinite(value) for value in values):
        return {
            "valid": False,
            "reason": "Non-finite monitoring measurements",
        }

    if (
        active_gpu_samples < 0
        or monitoring_window_seconds <= 0
        or maximum_sample_gap_seconds < 0
        or maximum_sample_gap_seconds > monitoring_window_seconds
    ):
        return {
            "valid": False,
            "reason": "Invalid monitoring measurements",
        }

    if active_gpu_samples < minimum_samples:
        return {
            "valid": False,
            "reason": "Insufficient GPU samples",
        }

    if maximum_sample_gap_seconds > maximum_gap_seconds:
        return {
            "valid": False,
            "reason": "Monitoring gap exceeds threshold",
        }

    return {
        "valid": True,
        "reason": "Monitoring coverage acceptable",
    }
