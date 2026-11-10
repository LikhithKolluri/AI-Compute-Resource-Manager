import math
import csv
from pathlib import Path
from src.measurement_quality import evaluate_measurement_quality
from src.reservation_advisor import recommend_vram_reservation
from src.memory_evidence import validate_memory_evidence
from src.memory_estimator import estimate_conservative_peak_mb

def find_matching_recommendation(
    job,
    csv_path="outputs/analytics/batch_comparison.csv",
):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        return None

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    matching_rows = []

    for row in rows:
        try:
            matches = (
                row["workload"] == job.workload
                and row["model_version"] == job.model_version
                and int(row["batch_size"]) == job.batch_size
                and row["precision"] == job.precision
            )

            if matches:
                quality = evaluate_measurement_quality(
                    active_gpu_samples=int(row["active_gpu_samples"]),
                    monitoring_window_seconds=float(
                        row["monitoring_window_seconds"]
                    ),
                    maximum_sample_gap_seconds=float(
                        row["maximum_sample_gap_seconds"]
                    ),
                )

                if quality["valid"]:
                    matching_rows.append(row)

        except (KeyError, ValueError, TypeError):
            continue

    if not matching_rows:
        return None

    peak_memory_values = []
    auto_eligible_sessions = 0

    for row in matching_rows:
        try:
            nvml_peak = float(row["estimated_peak_memory_mb"])

            if not math.isfinite(nvml_peak) or nvml_peak <= 0:
                continue

            # Check whether this session has complete memory evidence.
            evidence = validate_memory_evidence(
                nvml_peak_mb=nvml_peak,
                pytorch_allocated_mb=row.get(
                    "pytorch_peak_allocated_mb"
                ) or None,
                pytorch_reserved_mb=row.get(
                    "pytorch_peak_reserved_mb"
                ) or None,
            )

            # Invalid or missing PyTorch evidence must not
            # qualify a session for automatic optimization.
            pytorch_reserved = (
                row.get("pytorch_peak_reserved_mb")
                if evidence["valid"]
                else None
            )

            estimate = estimate_conservative_peak_mb(
                nvml_peak_mb=nvml_peak,
                pytorch_reserved_mb=pytorch_reserved,
            )

            peak_memory_values.append(
                estimate["estimated_peak_mb"]
            )

            if evidence["valid"]:
                auto_eligible_sessions += 1

        except (ValueError, TypeError, KeyError):
            continue

    if not peak_memory_values:
        return None

    # Use the largest observed estimate across matching sessions.
    peak_memory_mb = max(peak_memory_values)

    recommendation = recommend_vram_reservation(
        estimated_peak_increase_mb=peak_memory_mb
    )

    return {
        "matching_sessions": len(peak_memory_values),
        "auto_eligible_sessions": auto_eligible_sessions,
        "estimated_peak_memory_mb": peak_memory_mb,
        "recommended_reservation_gb": recommendation["recommended_gb"],
        "advisory_only": True,
    }