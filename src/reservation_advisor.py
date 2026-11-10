
import math


def recommend_vram_reservation(
    estimated_peak_increase_mb,
    safety_margin=0.25,
    minimum_reservation_gb=0.5,
    rounding_gb=0.25,
):
    if estimated_peak_increase_mb < 0:
        raise ValueError("Peak memory increase cannot be negative")

    if safety_margin < 0:
        raise ValueError("Safety margin cannot be negative")

    if minimum_reservation_gb <= 0 or rounding_gb <= 0:
        raise ValueError(
            "Minimum reservation and rounding must be positive"
        )

    estimated_gb = estimated_peak_increase_mb / 1024

    buffered_gb = estimated_gb * (1 + safety_margin)

    recommended_gb = max(
        minimum_reservation_gb,
        math.ceil(buffered_gb / rounding_gb) * rounding_gb,
    )

    return {
        "estimated_peak_gb": estimated_gb,
        "safety_margin": safety_margin,
        "buffered_gb": buffered_gb,
        "recommended_gb": recommended_gb,
        "advisory_only": True,
    }
