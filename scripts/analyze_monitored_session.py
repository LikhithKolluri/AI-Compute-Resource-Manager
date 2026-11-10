
import sys
from pathlib import Path

from src.gpu_analytics import (
    load_gpu_metrics,
    analyze_workload_memory,
)
from src.scheduling_analytics import load_events
from src.resource_correlation import correlate_resources
from src.reservation_advisor import recommend_vram_reservation


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m scripts.analyze_monitored_session "
            "<session_id>"
        )

    session_id = sys.argv[1]

    gpu_path = Path(
        f"outputs/analytics/gpu_{session_id}.csv"
    )

    if not gpu_path.exists():
        raise FileNotFoundError(gpu_path)

    samples = load_gpu_metrics(gpu_path)
    events = load_events()

    session_samples = [
        sample
        for sample in samples
        if sample.get("session_id") == session_id
    ]

    session_events = [
        event
        for event in events
        if event.get("session_id") == session_id
    ]

    if not session_samples or not session_events:
        raise RuntimeError(
            "Missing GPU samples or scheduler events "
            "for this session"
        )


    admitted_times = [
        event["timestamp"].timestamp()
        for event in session_events
        if event["event"] == "ADMITTED"
    ]

    completed_times = [
        event["timestamp"].timestamp()
        for event in session_events
        if event["event"] == "COMPLETED"
    ]

    if len(admitted_times) != 1 or len(completed_times) != 1:
        raise RuntimeError(
            "Expected exactly one admission and completion "
            "event for this training session"
        )

    training_start = admitted_times[0]
    training_end = completed_times[0]

    if training_end <= training_start:
        raise RuntimeError(
            "Training completion must follow admission"
        )


    baseline_samples = [
        sample
        for sample in session_samples
        if sample["timestamp"] < training_start
    ]

    active_samples = [
        sample
        for sample in session_samples
        if training_start <= sample["timestamp"] <= training_end
    ]

    post_training_samples = [
        sample
        for sample in session_samples
        if sample["timestamp"] > training_end
    ]

    print("\n--- MONITORING WINDOWS ---")
    print(f"Baseline samples: {len(baseline_samples)}")
    print(f"Active training samples: {len(active_samples)}")
    print(
        f"Post-training samples: "
        f"{len(post_training_samples)}"
    )



    memory_analysis = analyze_workload_memory(
        baseline_samples,
        active_samples,
    )

    if memory_analysis is None:
        raise RuntimeError(
            "Insufficient baseline or active GPU samples"
        )

    print("\n--- WORKLOAD MEMORY ANALYSIS ---")
    print(
        f"Baseline VRAM: "
        f"{memory_analysis['baseline_vram_mb']:.2f} MB"
    )
    print(
        f"Active average VRAM: "
        f"{memory_analysis['active_average_vram_mb']:.2f} MB"
    )
    print(
        f"Active peak VRAM: "
        f"{memory_analysis['active_peak_vram_mb']:.2f} MB"
    )
    print(
        f"Estimated workload peak increase: "
        f"{memory_analysis['estimated_peak_increase_mb']:.2f} MB"
    )

    print("\n--- POST-TRAINING MEMORY RECOVERY ---")

    if post_training_samples:
        post_average_mb = sum(
            sample["vram_used_mb"]
            for sample in post_training_samples
        ) / len(post_training_samples)

        baseline_mb = memory_analysis["baseline_vram_mb"]

        recovery_difference_mb = (
            post_average_mb - baseline_mb
        )

        print(f"Baseline VRAM: {baseline_mb:.2f} MB")
        print(
            f"Post-training average VRAM: "
            f"{post_average_mb:.2f} MB"
        )
        print(
            f"Post-training minus baseline: "
            f"{recovery_difference_mb:.2f} MB"
        )
    else:
        print("No post-training GPU samples available.")

    results = correlate_resources(
        samples,
        events,
        session_id=session_id,
        start_time=training_start,
        end_time=training_end,
        initial_reserved_gb=0.0,
    )

    if not results:
        raise RuntimeError(
            "No correlated GPU measurements found"
        )

    avg_reserved = sum(
        row["reserved_vram_gb"]
        for row in results
    ) / len(results)

    avg_actual = sum(
        row["actual_vram_gb"]
        for row in results
    ) / len(results)

    avg_utilization = sum(
        row["gpu_utilization_pct"]
        for row in results
    ) / len(results)

    print("\n--- MONITORED SESSION ANALYSIS ---")
    print(f"Session ID: {session_id}")
    print(f"GPU samples: {len(session_samples)}")
    print(f"Correlated samples: {len(results)}")
    print(f"Average reserved VRAM: {avg_reserved:.3f} GB")
    print(f"Average physical VRAM: {avg_actual:.3f} GB")
    print(f"Average GPU utilization: {avg_utilization:.2f}%")
    print(
        f"Average reservation minus usage: "
        f"{avg_reserved - avg_actual:.3f} GB"
    )


    peak_actual = max(
        row["actual_vram_gb"]
        for row in results
    )

    peak_utilization = max(
        row["gpu_utilization_pct"]
        for row in results
    )

    # Read GPU capacity from actual measurements
    total_vram_gb = (
        active_samples[0]["vram_total_mb"] / 1024
    )

    peak_vram_pct = (
        peak_actual / total_vram_gb * 100
        if total_vram_gb > 0
        else 0.0
    )

    print("\n--- PEAK GPU RESOURCE USAGE ---")
    print(f"Peak physical VRAM: {peak_actual:.3f} GB")
    print(f"Peak GPU utilization: {peak_utilization:.2f}%")
    print(f"Peak VRAM capacity used: {peak_vram_pct:.2f}%")

    recommendation = recommend_vram_reservation(
        estimated_peak_increase_mb=memory_analysis[
            "estimated_peak_increase_mb"
        ],
        safety_margin=0.25,
        minimum_reservation_gb=0.5,
        rounding_gb=0.25,
    )

    print("\n--- VRAM RESERVATION ADVISOR ---")
    print(
        f"Estimated workload peak: "
        f"{recommendation['estimated_peak_gb']:.3f} GiB"
    )
    print(
        f"Buffered estimate: "
        f"{recommendation['buffered_gb']:.3f} GiB"
    )
    print(
        f"Candidate reservation: "
        f"{recommendation['recommended_gb']:.3f} GiB"
    )
    print(
        f"Advisory only: "
        f"{recommendation['advisory_only']}"
    )

if __name__ == "__main__":
    main()
