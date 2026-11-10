
import argparse
import csv
from pathlib import Path
from statistics import mean


METRICS_ROOT = Path("outputs/metrics")


def load_csv(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {file_path}")

    with file_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def analyze_job(job_name):
    job_dir = METRICS_ROOT / job_name

    gpu_rows = load_csv(job_dir / "gpu_metrics.csv")
    training_rows = load_csv(job_dir / "training_metrics.csv")

    if not gpu_rows or not training_rows:
        raise ValueError(f"Missing metrics data for {job_name}")

    utilization = [
        float(row["gpu_utilization_pct"])
        for row in gpu_rows
    ]

    vram_used = [
        float(row["vram_used_mb"])
        for row in gpu_rows
    ]

    throughput = [
        float(row["samples_per_second"])
        for row in training_rows
    ]

    epoch_times = [
        float(row["epoch_seconds"])
        for row in training_rows
    ]

    final_epoch = max(
        training_rows,
        key=lambda row: int(row["epoch"])
    )

    return {
        "job_name": job_name,
        "avg_gpu_utilization": mean(utilization),
        "peak_vram_gb": max(vram_used) / 1024,
        "avg_throughput": mean(throughput),
        "avg_epoch_seconds": mean(epoch_times),
        "final_loss": float(final_epoch["avg_loss"]),
        "recorded_epochs": len(training_rows),
        "last_epoch": int(final_epoch["epoch"]),
    }


def print_report(results):
    print("\n" + "=" * 75)
    print("GPU JOB PERFORMANCE REPORT")
    print("=" * 75)

    for result in results:
        print(f"\nJob: {result['job_name']}")
        print("-" * 45)

        print(
            f"Average GPU Utilization : "
            f"{result['avg_gpu_utilization']:.2f}%"
        )
        print(
            f"Peak GPU VRAM Usage     : "
            f"{result['peak_vram_gb']:.2f} GB"
        )
        print(
            f"Average Throughput      : "
            f"{result['avg_throughput']:,.0f} samples/sec"
        )
        print(
            f"Average Epoch Duration  : "
            f"{result['avg_epoch_seconds']:.2f} seconds"
        )
        print(
            f"Final Training Loss     : "
            f"{result['final_loss']:.5f}"
        )
        print(
            f"Recorded Epochs         : "
            f"{result['recorded_epochs']}"
        )
        print(
            f"Last Epoch Number       : "
            f"{result['last_epoch']}"
        )

    if len(results) >= 2:
        print("\n" + "=" * 75)
        print("JOB COMPARISON")
        print("=" * 75)

        fastest = max(
            results,
            key=lambda result: result["avg_throughput"]
        )

        print(
            f"\nHighest average throughput: "
            f"{fastest['job_name']} "
            f"({fastest['avg_throughput']:,.0f} samples/sec)"
        )

        shortest_epoch = min(
            results,
            key=lambda result: result["avg_epoch_seconds"]
        )

        print(
            f"Shortest average epoch: "
            f"{shortest_epoch['job_name']} "
            f"({shortest_epoch['avg_epoch_seconds']:.2f} seconds)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze GPU training job performance"
    )

    parser.add_argument(
        "--jobs",
        nargs="+",
        default=["training-A", "training-B"],
        help="Job names to compare"
    )

    args = parser.parse_args()

    results = []

    for job_name in args.jobs:
        try:
            results.append(analyze_job(job_name))
        except (FileNotFoundError, ValueError, KeyError) as error:
            print(f"Skipping {job_name}: {error}")

    if not results:
        print("No valid job metrics found.")
        return

    print_report(results)


if __name__ == "__main__":
    main()
