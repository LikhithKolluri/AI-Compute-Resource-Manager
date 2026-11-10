
import argparse

from src.performance_report import analyze_job


def generate_recommendations(metrics):
    recommendations = []

    job_name = metrics["job_name"]
    utilization = metrics["avg_gpu_utilization"]
    throughput = metrics["avg_throughput"]
    epoch_seconds = metrics["avg_epoch_seconds"]
    peak_vram = metrics["peak_vram_gb"]

    # Rule 1: Low GPU utilization
    if utilization < 40:
        recommendations.append(
            "LOW GPU UTILIZATION: Investigate CPU/data loading "
            "bottlenecks, small batch sizes, or GPU contention."
        )

    elif utilization < 70:
        recommendations.append(
            "MODERATE GPU UTILIZATION: Consider benchmarking "
            "different batch sizes and checking GPU idle periods."
        )

    else:
        recommendations.append(
            "GOOD GPU UTILIZATION: GPU activity is relatively high "
            "during the recorded monitoring period."
        )

    # Rule 2: GPU memory usage
    if peak_vram > 7:
        recommendations.append(
            "HIGH VRAM USAGE: Peak observed device memory usage "
            "exceeds 7 GB. Monitor for out-of-memory errors."
        )

    # Rule 3: Throughput
    if throughput <= 0:
        recommendations.append(
            "THROUGHPUT WARNING: No positive training throughput "
            "was recorded."
        )

    # Rule 4: Epoch duration
    if epoch_seconds > 10:
        recommendations.append(
            "LONG EPOCH DURATION: Review training step count, "
            "batch size, and workload bottlenecks."
        )

    return {
        "job_name": job_name,
        "recommendations": recommendations,
        "metrics": metrics,
    }


def print_advisor_report(results):
    print("\n" + "=" * 70)
    print("GPU PERFORMANCE ADVISOR")
    print("=" * 70)

    for result in results:
        metrics = result["metrics"]

        print(f"\nJOB: {result['job_name']}")
        print("-" * 50)

        print(
            f"Average GPU Utilization : "
            f"{metrics['avg_gpu_utilization']:.2f}%"
        )
        print(
            f"Peak Device VRAM Usage  : "
            f"{metrics['peak_vram_gb']:.2f} GB"
        )
        print(
            f"Average Throughput      : "
            f"{metrics['avg_throughput']:,.0f} samples/sec"
        )
        print(
            f"Average Epoch Duration  : "
            f"{metrics['avg_epoch_seconds']:.2f} seconds"
        )

        print("\nRecommendations:")

        for index, recommendation in enumerate(
            result["recommendations"],
            start=1
        ):
            print(f"  {index}. {recommendation}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate GPU performance recommendations"
    )

    parser.add_argument(
        "--jobs",
        nargs="+",
        default=["training-A", "training-B"]
    )

    args = parser.parse_args()

    results = []

    for job_name in args.jobs:
        try:
            metrics = analyze_job(job_name)
            results.append(generate_recommendations(metrics))
        except (FileNotFoundError, ValueError, KeyError) as error:
            print(f"Skipping {job_name}: {error}")

    if not results:
        print("No valid metrics found.")
        return

    print_advisor_report(results)


if __name__ == "__main__":
    main()
