import math


def validate_memory_evidence(
    nvml_peak_mb,
    pytorch_allocated_mb,
    pytorch_reserved_mb,
):
    values = (
        nvml_peak_mb,
        pytorch_allocated_mb,
        pytorch_reserved_mb,
    )

    if any(value is None for value in values):
        return {
            "valid": False,
            "reason": "Missing memory measurements",
        }

    try:
        values = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return {
            "valid": False,
            "reason": "Invalid memory measurements",
        }

    if not all(math.isfinite(value) and value > 0 for value in values):
        return {
            "valid": False,
            "reason": "Invalid memory measurements",
        }

    nvml_peak, allocated, reserved = values

    if reserved < allocated:
        return {
            "valid": False,
            "reason": "PyTorch reserved memory below allocated memory",
        }

    return {
        "valid": True,
        "reason": "Memory measurements available",
    }