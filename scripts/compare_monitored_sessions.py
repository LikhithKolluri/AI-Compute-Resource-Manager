import json
import argparse
import csv
from pathlib import Path
from statistics import mean

from src.gpu_analytics import load_gpu_metrics, analyze_workload_memory
from src.scheduling_analytics import load_events


def compare_session(session_id, events):
    gpu_path = Path(f"outputs/analytics/gpu_{session_id}.csv")
    training_path = (
        Path("outputs/metrics")
        / "session-training"
        / session_id
        / "training_metrics.csv"
    )

    memory_path = training_path.parent / "memory_metrics.json"
    
    if not gpu_path.exists():
        raise FileNotFoundError(gpu_path)

    if not training_path.exists():
        raise FileNotFoundError(training_path)

    samples = [
        sample for sample in load_gpu_metrics(gpu_path)
        if sample.get("session_id") == session_id
    ]

    session_events = [
        event for event in events
        if event.get("session_id") == session_id
    ]
    
    
    # Use the admission event to identify the measured job.
    admission_events = [
        event for event in session_events
        if event["event"] == "ADMITTED"
    ]

    if len(admission_events) != 1:
        raise ValueError(
            f"Session {session_id} requires exactly one ADMITTED event"
        )

    admission = admission_events[0]

    workload = admission.get("workload")
    model_version = admission.get("model_version")
    batch_size = admission.get("batch_size")
    precision = admission.get("precision")

    
    submitted_events = [
        event for event in session_events
        if event["event"] == "SUBMITTED"
    ]

    if len(submitted_events) != 1:
        raise ValueError(
            f"Session {session_id} requires exactly one SUBMITTED event"
        )

    admissions = [
        event["timestamp"].timestamp()
        for event in session_events
        if event["event"] == "ADMITTED"
    ]

    completions = [
        event["timestamp"].timestamp()
        for event in session_events
        if event["event"] == "COMPLETED"
    ]

    if len(admissions) != 1 or len(completions) != 1:
        raise ValueError(
            f"Session {session_id} requires exactly one "
            "ADMITTED and COMPLETED event"
        )

    start, end = admissions[0], completions[0]

    if end <= start:
        raise ValueError("Completion must follow admission")

    baseline = [s for s in samples if s["timestamp"] < start]
    active = [s for s in samples if start <= s["timestamp"] <= end]
    post = [s for s in samples if s["timestamp"] > end]

    active_timestamps = sorted(
        sample["timestamp"] for sample in active
    )

    if active_timestamps:
        coverage_gaps = [
            active_timestamps[0] - start,
            end - active_timestamps[-1],
        ]

        coverage_gaps.extend(
            current - previous
            for previous, current in zip(
                active_timestamps,
                active_timestamps[1:]
            )
        )

        maximum_sample_gap_seconds = max(coverage_gaps)
    else:
        maximum_sample_gap_seconds = end - start
    
    memory = analyze_workload_memory(baseline, active)
    if memory is None or not post:
        raise ValueError(
            f"Insufficient monitoring windows for {session_id}"
        )

    with training_path.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        training_rows = list(csv.DictReader(file))

    if not training_rows:
        raise ValueError(f"No training metrics for {session_id}")

    durations = [float(r["epoch_seconds"]) for r in training_rows]
    throughputs = [
        float(r["samples_per_second"]) for r in training_rows
    ]
    
    total_training_seconds = sum(durations)

    if total_training_seconds <= 0:
        raise ValueError("Training duration must be positive")

    total_samples = sum(
        duration * throughput
        for duration, throughput in zip(durations, throughputs)
    )

    overall_throughput = total_samples / total_training_seconds

    pytorch_peak_allocated_mb = None
    pytorch_peak_reserved_mb = None

    if memory_path.exists():
        with memory_path.open("r", encoding="utf-8") as file:
            memory_metrics = json.load(file)

        pytorch_peak_allocated_mb = memory_metrics.get(
            "peak_allocated_mb"
        )
        pytorch_peak_reserved_mb = memory_metrics.get(
            "peak_reserved_mb"
        )
    
    return {
        "session_id": session_id,
        "workload": workload,
        "model_version": model_version,
        "batch_size": batch_size,
        "precision": precision,
        "epochs": len(training_rows),
        "training_seconds": sum(durations),
        "average_throughput": overall_throughput,
        "active_gpu_samples": len(active),
        "monitoring_window_seconds": end - start,
        "average_gpu_utilization": mean(
            s["gpu_utilization_pct"] for s in active
        ),
        "peak_gpu_utilization": max(
            s["gpu_utilization_pct"] for s in active
        ),
        "estimated_peak_memory_mb": memory[
            "estimated_peak_increase_mb"
        ],
        "post_training_memory_difference_mb": (
            mean(s["vram_used_mb"] for s in post)
            - memory["baseline_vram_mb"]
        ),
        "maximum_sample_gap_seconds": maximum_sample_gap_seconds,
        "pytorch_peak_allocated_mb": pytorch_peak_allocated_mb,
        "pytorch_peak_reserved_mb": pytorch_peak_reserved_mb,
    }

def export_comparison_csv(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        raise ValueError("No comparison results to export")

    existing_results = {}

    if output_path.exists():
        with output_path.open(
            "r", newline="", encoding="utf-8"
        ) as file:
            for row in csv.DictReader(file):
                session_id = row.get("session_id")

                if session_id:
                    existing_results[session_id] = row

    # Update existing sessions or insert new ones.
    for result in results:
        existing_results[result["session_id"]] = result

    # Preserve the current schema and any historical columns.
    fieldnames = list(
        dict.fromkeys(
            key
            for row in existing_results.values()
            for key in row.keys()
        )
    )

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(existing_results.values())

    return output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sessions",
        nargs="+",
        help="Session IDs to compare",
    )
    args = parser.parse_args()

    events = load_events()
    results = [
        compare_session(session_id, events)
        for session_id in args.sessions
    ]

    print("\n--- MONITORED SESSION COMPARISON ---")

    for result in results:
        print(f"\nSession: {result['session_id']}")
        print(f"Epochs: {result['epochs']}")
        print(
            f"Training time: "
            f"{result['training_seconds']:.2f}s"
        )
        print(
            f"Average throughput: "
            f"{result['average_throughput']:,.0f} samples/s"
        )
        print(
            f"Average GPU utilization: "
            f"{result['average_gpu_utilization']:.2f}%"
        )
        print(
            f"Peak GPU utilization: "
            f"{result['peak_gpu_utilization']:.2f}%"
        )
        print(
            f"Estimated peak workload VRAM: "
            f"{result['estimated_peak_memory_mb']:.2f} MB"
        )
        print(
            f"Post-training memory difference: "
            f"{result['post_training_memory_difference_mb']:.2f} MB"
        )
        if result["pytorch_peak_allocated_mb"] is not None:
            print(
                f"PyTorch peak allocated VRAM: "
                f"{result['pytorch_peak_allocated_mb']:.2f} MB"
            )
            print(
                f"PyTorch peak reserved VRAM: "
                f"{result['pytorch_peak_reserved_mb']:.2f} MB"
            )
        
    output_path = export_comparison_csv(
        results,
        "outputs/analytics/batch_comparison.csv"
    )
    print(f"\nComparison CSV saved: {output_path}")


if __name__ == "__main__":
    main()
