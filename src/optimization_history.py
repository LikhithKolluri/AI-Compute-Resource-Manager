import torch
import csv
from datetime import datetime, timezone
from pathlib import Path

HISTORY_PATH = Path("outputs/analytics/optimization_decisions.csv")
DISTRIBUTED_HISTORY_PATH = Path("outputs/analytics/distributed_optimization_decisions.csv")


def record_optimization_decision(
    job,
    decision,
    reason,
    baseline_throughput=None,
    optimized_throughput=None,
):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "timestamp",
        "job_name",
        "workload",
        "gpu_name",
        "model_version",
        "precision",
        "original_batch_size",
        "optimized_batch_size",
        "decision",
        "reason",
        "baseline_throughput",
        "optimized_throughput",
        
    ]

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "job_name": job.name,
        "workload": job.workload,
        "model_version": job.model_version,
        "precision": job.precision,
        "original_batch_size": job.original_batch_size,
        "optimized_batch_size": job.batch_size,
        "decision": decision,
        "reason": reason,
        "baseline_throughput": baseline_throughput,
        "optimized_throughput": optimized_throughput,
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),
    }

    write_header = not HISTORY_PATH.exists() or HISTORY_PATH.stat().st_size == 0

    with HISTORY_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def find_previous_decision(job, proposed_batch_size):
    if not HISTORY_PATH.exists():
        return None

    with HISTORY_PATH.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        
        reader.fieldnames = [
            name.strip() for name in reader.fieldnames
        ]

        matching_decisions = []

        for row in reader:
            row = {
                key: value.strip() if value is not None else None
                for key, value in row.items()
            }
            
        if (
            row["gpu_name"] == (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "CPU"
            )
            and row["workload"] == job.workload
            and row["model_version"] == job.model_version
            and row["precision"] == job.precision
            and row["original_batch_size"] == str(job.batch_size)
            and row["optimized_batch_size"] == str(proposed_batch_size)
        ):
            matching_decisions.append(row)


    if not matching_decisions:
        return None

    return matching_decisions[-1]


def record_distributed_decision(
    current_batch_size,
    recommended_batch_size,
    workers,
    samples,
    decision,
    reason,
    measured_throughput=None,
):
    DISTRIBUTED_HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "timestamp",
        "benchmark_type",
        "workers",
        "dataset_samples",
        "original_batch_size",
        "recommended_batch_size",
        "decision",
        "reason",
        "measured_throughput",
    ]

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_type": "distributed_cpu_ddp",
        "workers": workers,
        "dataset_samples": samples,
        "original_batch_size": current_batch_size,
        "recommended_batch_size": recommended_batch_size,
        "decision": decision,
        "reason": reason,
        "measured_throughput": measured_throughput,
    }

    write_header = (
        not DISTRIBUTED_HISTORY_PATH.exists()
        or DISTRIBUTED_HISTORY_PATH.stat().st_size == 0
    )

    with DISTRIBUTED_HISTORY_PATH.open(
        "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def find_distributed_decision(
    current_batch_size,
    proposed_batch_size,
    workers,
    samples,
):
    if not DISTRIBUTED_HISTORY_PATH.exists():
        return None

    with DISTRIBUTED_HISTORY_PATH.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        matching_decisions = []

        for row in reader:
            if (
                row["benchmark_type"] == "distributed_cpu_ddp"
                and row["workers"] == str(workers)
                and row["dataset_samples"] == str(samples)
                and row["original_batch_size"] == str(current_batch_size)
                and row["recommended_batch_size"] == str(proposed_batch_size)
            ):
                matching_decisions.append(row)

    if not matching_decisions:
        return None

    return matching_decisions[-1]

