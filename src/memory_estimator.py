import math


def estimate_conservative_peak_mb(
    nvml_peak_mb,
    pytorch_reserved_mb=None,
):
    nvml_peak_mb = float(nvml_peak_mb)

    if not math.isfinite(nvml_peak_mb) or nvml_peak_mb <= 0:
        raise ValueError("Invalid NVML peak memory")

    if pytorch_reserved_mb is None or pytorch_reserved_mb == "":
        return {
            "estimated_peak_mb": nvml_peak_mb,
            "source": "nvml_only",
            "complete_evidence": False,
        }

    pytorch_reserved_mb = float(pytorch_reserved_mb)

    if (
        not math.isfinite(pytorch_reserved_mb)
        or pytorch_reserved_mb <= 0
    ):
        raise ValueError("Invalid PyTorch reserved memory")

    estimated_peak_mb = max(
        nvml_peak_mb,
        pytorch_reserved_mb,
    )

    return {
        "estimated_peak_mb": estimated_peak_mb,
        "source": "combined",
        "complete_evidence": True,
    }