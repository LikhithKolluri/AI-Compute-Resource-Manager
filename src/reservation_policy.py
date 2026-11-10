
import math


def evaluate_reservation_policy(
    requested_gb,
    recommended_gb,
    matching_sessions,
    auto_apply=False,
    minimum_sessions=3,
    maximum_reduction=0.50,
):
    """
    Decide whether a reservation recommendation
    is eligible for automatic application.

    This function does not modify the job.
    """

    if not math.isfinite(requested_gb) or requested_gb <= 0:
        raise ValueError("Requested reservation must be positive")

    if not math.isfinite(recommended_gb) or recommended_gb <= 0:
        raise ValueError("Recommended reservation must be positive")

    if matching_sessions < 0:
        raise ValueError("Matching sessions cannot be negative")

    if minimum_sessions < 1:
        raise ValueError("Minimum sessions must be at least 1")

    if not 0 <= maximum_reduction < 1:
        raise ValueError("Maximum reduction must be between 0 and 1")

    if not auto_apply:
        return {
            "approved": False,
            "reason": "Auto-apply disabled",
            "reservation_gb": requested_gb,
        }

    if matching_sessions < minimum_sessions:
        return {
            "approved": False,
            "reason": "Insufficient historical sessions",
            "reservation_gb": requested_gb,
        }

    if recommended_gb >= requested_gb:
        return {
            "approved": False,
            "reason": "No reservation reduction available",
            "reservation_gb": requested_gb,
        }

    # Limit the maximum reduction in a single decision.
    minimum_allowed_gb = requested_gb * (
        1 - maximum_reduction
    )

    adjusted_gb = max(
        recommended_gb,
        minimum_allowed_gb,
    )

    return {
        "approved": True,
        "reason": "Eligible under initial policy",
        "reservation_gb": adjusted_gb,
    }
