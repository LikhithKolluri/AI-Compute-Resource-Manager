import csv
import math
from pathlib import Path


def find_historical_baseline(
    job,
    csv_path="outputs/analytics/batch_comparison.csv",
):
    original_batch_size = job.original_batch_size

    if original_batch_size is None:
        return None

    path = Path(csv_path)

    if not path.exists():
        return None

    throughputs = []

    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            try:
                matches = (
                    row["workload"] == job.workload
                    and row["model_version"] == job.model_version
                    and row["precision"] == job.precision
                    and int(row["batch_size"]) == original_batch_size
                )

                if not matches:
                    continue

                throughput = float(row["average_throughput"])

                if math.isfinite(throughput) and throughput > 0:
                    throughputs.append(throughput)

            except (KeyError, ValueError, TypeError):
                continue

    if not throughputs:
        return None

    return {
        "baseline_throughput": sum(throughputs) / len(throughputs),
        "matching_sessions": len(throughputs),
        "original_batch_size": original_batch_size,
    }
    
def find_distributed_baseline(
    job,
    metrics_path=None,
):
    if job.original_batch_size is None:
        return None

    if metrics_path is None:
        metrics_path = (
            f"outputs/distributed/"
            f"{job.name}_baseline_metrics.json"
        )

    path = Path(metrics_path)

    if not path.exists():
        return None

    import json

    try:
        metrics = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None

    if (
        metrics.get("status") != "COMPLETED"
        or metrics.get("workers") != job.workers
        or metrics.get("batch_size_per_worker")
            != job.original_batch_size
        or metrics.get("dataset_samples") != job.samples
    ):
        return None

    
    epochs = metrics.get("epochs", [])

    steady_epochs = [
        epoch for epoch in epochs
        if epoch.get("epoch", 0) > 1
    ]

    if not steady_epochs:
        return None

    try:
        total_samples = sum(
            epoch["samples_processed"]
            for epoch in steady_epochs
        )
        total_duration = sum(
            epoch["duration_seconds"]
            for epoch in steady_epochs
        )
    except (KeyError, TypeError, ValueError):
        return None

    throughput = (
        total_samples / total_duration
        if total_duration > 0 else 0.0
    )

    if (
        not isinstance(throughput, (int, float))
        or not math.isfinite(throughput)
        or throughput <= 0
    ):
        return None

    return {
        "baseline_throughput": throughput,
        "original_batch_size": job.original_batch_size,
        "workers": job.workers,
        "samples": job.samples,
    }
