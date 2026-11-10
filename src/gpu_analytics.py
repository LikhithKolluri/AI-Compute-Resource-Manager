
import csv
from pathlib import Path
from statistics import mean

GPU_METRICS_PATH = Path("outputs/gpu_metrics.csv")


def load_gpu_metrics(path=GPU_METRICS_PATH):
    path = Path(path)

    if not path.exists():
        return []

    samples = []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            samples.append({
                "timestamp": float(row["timestamp"]),
                "gpu_utilization_pct": float(
                    row["gpu_utilization_pct"]
                ),
                "vram_used_mb": float(row["vram_used_mb"]),
                "vram_total_mb": float(row["vram_total_mb"]),
                "session_id": row.get("session_id", ""),
                "gpu_index": int(row.get("gpu_index") or 0),
            })

    return samples


def analyze_gpu_metrics(samples):
    if not samples:
        return None

    utilization = [
        sample["gpu_utilization_pct"]
        for sample in samples
    ]

    vram_usage = [
        sample["vram_used_mb"]
        for sample in samples
    ]

    total_vram = samples[0]["vram_total_mb"]

    return {
        "sample_count": len(samples),
        "average_gpu_utilization_pct": mean(utilization),
        "peak_gpu_utilization_pct": max(utilization),
        "average_vram_used_mb": mean(vram_usage),
        "peak_vram_used_mb": max(vram_usage),
        "total_vram_mb": total_vram,
        "peak_vram_usage_pct": (
            max(vram_usage) / total_vram * 100
            if total_vram > 0
            else 0
        ),
    }


def analyze_workload_memory(
    baseline_samples,
    active_samples
):
    if not baseline_samples or not active_samples:
        return None

    baseline_mb = mean(
        sample["vram_used_mb"]
        for sample in baseline_samples
    )

    active_peak_mb = max(
        sample["vram_used_mb"]
        for sample in active_samples
    )

    active_average_mb = mean(
        sample["vram_used_mb"]
        for sample in active_samples
    )

    estimated_peak_increase_mb = max(
        0.0,
        active_peak_mb - baseline_mb
    )

    return {
        "baseline_vram_mb": baseline_mb,
        "active_average_vram_mb": active_average_mb,
        "active_peak_vram_mb": active_peak_mb,
        "estimated_peak_increase_mb": (
            estimated_peak_increase_mb
        ),
        "estimated_peak_increase_gb": (
            estimated_peak_increase_mb / 1024
        ),
    }


if __name__ == "__main__":
    samples = load_gpu_metrics()
    results = analyze_gpu_metrics(samples)

    if results is None:
        print("No GPU metrics available.")
    else:
        print("\n--- GPU RESOURCE ANALYTICS ---")
        print(f"Samples: {results['sample_count']}")
        print(
            f"Average GPU utilization: "
            f"{results['average_gpu_utilization_pct']:.2f}%"
        )
        print(
            f"Peak GPU utilization: "
            f"{results['peak_gpu_utilization_pct']:.2f}%"
        )
        print(
            f"Average VRAM used: "
            f"{results['average_vram_used_mb']:.2f} MB"
        )
        print(
            f"Peak VRAM used: "
            f"{results['peak_vram_used_mb']:.2f} MB"
        )
        print(
            f"Peak VRAM usage: "
            f"{results['peak_vram_usage_pct']:.2f}%"
        )
