import csv
import torch
from pathlib import Path
from src.optimization_history import find_previous_decision
from src.optimization_history import find_distributed_decision



BENCHMARK_PATH = Path("outputs/benchmark_results.csv")


def load_benchmarks(path=BENCHMARK_PATH):
    path = Path(path)

    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    required = {
        "batch_size",
        "mean_throughput",
        "peak_allocated_gb",
        "workload",
        "model_version",
        "precision",
        "gpu_name",
    }

    if not rows:
        return []

    if not required.issubset(rows[0]):
        print(
            "[ADVISOR] Benchmark metadata missing. "
            "Please rerun the updated benchmark."
        )
        return []

    benchmarks = []

    for row in rows:
        benchmarks.append({
            "batch_size": int(row["batch_size"]),
            "mean_throughput": float(row["mean_throughput"]),
            "peak_allocated_gb": float(row["peak_allocated_gb"]),
            "workload": row["workload"],
            "model_version": row["model_version"],
            "precision": row["precision"],
            "gpu_name": row["gpu_name"],
        })

    return benchmarks


def recommend_batch_size(
    current_batch_size,
    workload="synthetic-classifier",
    model_version="classifier-v1",
    precision="fp32",
    max_vram_gb=7.0,
    benchmark_path=BENCHMARK_PATH,
):
    if not torch.cuda.is_available():
        return None

    current_gpu = torch.cuda.get_device_name(0)
    benchmarks = load_benchmarks(benchmark_path)

    eligible = [
        row
        for row in benchmarks
        if row["workload"] == workload
        and row["model_version"] == model_version
        and row["precision"] == precision
        and row["gpu_name"] == current_gpu
        and row["peak_allocated_gb"] <= max_vram_gb
    ]

    
    if not eligible:
        return None

    # Use the existing job fields expected by the history lookup.
    from types import SimpleNamespace

    lookup_job = SimpleNamespace(
        workload=workload,
        model_version=model_version,
        precision=precision,
        batch_size=current_batch_size,
    )

    allowed = []

    for row in eligible:
        previous = find_previous_decision(
            lookup_job,
            proposed_batch_size=row["batch_size"],
        )

        if previous is not None and previous["decision"] == "REJECTED":
            print(
                f"[OPTIMIZER] Skipping batch size {row['batch_size']}: "
                f"previously rejected",
                flush=True,
            )
            continue

        allowed.append(row)

    if not allowed:
        return None

    best = max(
        allowed,
        key=lambda row: row["mean_throughput"]
    )


    current = next(
        (
            row for row in eligible
            if row["batch_size"] == current_batch_size
        ),
        None
    )

    improvement = None

    if current and current["mean_throughput"] > 0:
        improvement = (
            best["mean_throughput"]
            / current["mean_throughput"]
            - 1
        ) * 100

    return {
        "current_batch_size": current_batch_size,
        "recommended_batch_size": best["batch_size"],
        "recommended_throughput": best["mean_throughput"],
        "estimated_improvement_pct": improvement,
        "peak_allocated_gb": best["peak_allocated_gb"],
        "workload": workload,
        "model_version": model_version,
        "gpu_name": current_gpu,
    }


def print_recommendation(job_name, recommendation):
    if recommendation is None:
        print(
            f"[ADVISOR] {job_name}: "
            "No compatible benchmark found."
        )
        return

    current = recommendation["current_batch_size"]
    recommended = recommendation["recommended_batch_size"]
    improvement = recommendation["estimated_improvement_pct"]

    if current == recommended:
        print(
            f"[ADVISOR] {job_name}: "
            f"Batch size {current} is already optimal "
            "among compatible tested configurations."
        )
        return

    print(
        f"[ADVISOR] {job_name}: "
        f"Current batch size = {current}; "
        f"recommended = {recommended}."
    )

    if improvement is not None:
        print(
            f"[ADVISOR] Potential measured throughput "
            f"improvement: {improvement:.1f}%"
        )
    else:
        print(
            "[ADVISOR] Current batch size was not benchmarked."
        )

    print(
        "[ADVISOR] Recommendation only. "
        "Job configuration remains unchanged."
    )

# Distributed training benchmark recommendations

DISTRIBUTED_BENCHMARK_PATH = Path(
    "outputs/distributed/benchmarks/batch_comparison.csv"
)


def load_distributed_benchmarks(
    path=DISTRIBUTED_BENCHMARK_PATH,
):
    path = Path(path)

    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    required = {
        "workers",
        "batch_size_per_worker",
        "epochs",
        "dataset_samples",
        "samples_per_second",
        "duration_seconds",
    }

    if not rows:
        return []

    if not required.issubset(rows[0]):
        print(
            "[DISTRIBUTED ADVISOR] Benchmark columns missing."
        )
        return []

    benchmarks = []

    for row in rows:
        try:
            benchmarks.append({
                "workers": int(row["workers"]),
                "batch_size_per_worker": int(
                    row["batch_size_per_worker"]
                ),
                "epochs": int(row["epochs"]),
                "dataset_samples": int(row["dataset_samples"]),
                "samples_per_second": float(
                    row["samples_per_second"]
                ),
                "duration_seconds": float(
                    row["duration_seconds"]
                ),
            })
        except (ValueError, KeyError):
            print(
                "[DISTRIBUTED ADVISOR] "
                "Skipping invalid benchmark row."
            )

    return benchmarks


def recommend_distributed_batch_size(
    current_batch_size,
    workers,
    samples=4096,
    benchmark_path=DISTRIBUTED_BENCHMARK_PATH,
):
    benchmarks = load_distributed_benchmarks(benchmark_path)

    eligible = [
        row
        for row in benchmarks
        if row["workers"] == workers
        and row["dataset_samples"] == samples
        and row["samples_per_second"] > 0
    ]

    if not eligible:
        return None

    # Average repeated trials for each batch size.
    from collections import defaultdict
    import statistics

    grouped = defaultdict(list)

    for row in eligible:
        grouped[row["batch_size_per_worker"]].append(
            row["samples_per_second"]
        )

    
    averages = {
        batch_size: statistics.mean(throughputs)
        for batch_size, throughputs in grouped.items()
    }

    allowed = {}

    for batch_size, throughput in averages.items():
        previous = find_distributed_decision(
            current_batch_size=current_batch_size,
            proposed_batch_size=batch_size,
            workers=workers,
            samples=samples,
        )

        if previous is not None and previous["decision"] == "REJECTED":
            print(
                f"[DISTRIBUTED OPTIMIZER] Skipping batch size "
                f"{batch_size}: previously rejected",
                flush=True,
            )
            continue

        allowed[batch_size] = throughput

    if not allowed:
        return None

    best_batch_size = max(
        allowed,
        key=allowed.get,
    )


    current_throughput = averages.get(current_batch_size)

    improvement = None

    if current_throughput is not None and current_throughput > 0:
        improvement = (
            averages[best_batch_size] / current_throughput - 1
        ) * 100

    return {
        "current_batch_size": current_batch_size,
        "recommended_batch_size": best_batch_size,
        "recommended_throughput": averages[best_batch_size],
        "estimated_improvement_pct": improvement,
        "workers": workers,
        "global_batch_size": best_batch_size * workers,
        "benchmark_type": "distributed_cpu_ddp",
    }


def print_distributed_recommendation(job_name, recommendation):
    if recommendation is None:
        print(
            f"[DISTRIBUTED ADVISOR] {job_name}: "
            "No compatible distributed benchmark found."
        )
        return

    current = recommendation["current_batch_size"]
    recommended = recommendation["recommended_batch_size"]
    workers = recommendation["workers"]
    improvement = recommendation["estimated_improvement_pct"]

    print(f"\n[DISTRIBUTED ADVISOR] Job: {job_name}")
    print(f"[DISTRIBUTED ADVISOR] Workers: {workers}")
    print(
        f"[DISTRIBUTED ADVISOR] "
        f"Current batch size per worker: {current}"
    )
    print(
        f"[DISTRIBUTED ADVISOR] "
        f"Recommended batch size per worker: {recommended}"
    )
    print(
        f"[DISTRIBUTED ADVISOR] "
        f"Recommended global batch size: "
        f"{recommendation['global_batch_size']}"
    )

    if improvement is not None:
        print(
            f"[DISTRIBUTED ADVISOR] "
            f"Measured end-to-end throughput difference: "
            f"{improvement:.1f}%"
        )
    else:
        print(
            "[DISTRIBUTED ADVISOR] "
            "Current batch size was not benchmarked."
        )

    print(
        "[DISTRIBUTED ADVISOR] Recommendation only. "
        "Job configuration remains unchanged."
    )

