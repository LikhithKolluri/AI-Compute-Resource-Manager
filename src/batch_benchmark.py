
import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


OUTPUT_DIR = Path("outputs")


def run_trial(batch_size, trial, fixed_samples, warmup_steps):
    job_name = f"benchmark-bs{batch_size}-trial{trial}"

    result_path = (
        OUTPUT_DIR / "metrics" / job_name / "benchmark_result.json"
    )

    # Prevent a failed trial from reusing a previous result.
    result_path.unlink(missing_ok=True)

    command = [
        sys.executable,
        "-m",
        "src.train",
        "--benchmark",
        "--job-name",
        job_name,
        "--batch-size",
        str(batch_size),
        "--fixed-samples",
        str(fixed_samples),
        "--warmup-steps",
        str(warmup_steps),
    ]

    print(
        f"\nBenchmark: batch={batch_size}, "
        f"trial={trial}",
        flush=True
    )

    completed = subprocess.run(command, check=False)

    if completed.returncode != 0:
        print(
            f"FAILED: batch={batch_size}, trial={trial}, "
            f"exit={completed.returncode}"
        )
        return None

    if not result_path.exists():
        print(f"FAILED: Result file missing: {result_path}")
        return None

    with result_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def summarize_batch(batch_size, trials):
    throughputs = [
        trial["samples_per_second"]
        for trial in trials
    ]

    memory = [
        trial["peak_allocated_gb"]
        for trial in trials
    ]

    durations = [
        trial["duration_seconds"]
        for trial in trials
    ]

    return {
        "batch_size": batch_size,
        "successful_trials": len(trials),
        "mean_throughput": statistics.mean(throughputs),
        "std_throughput": (
            statistics.stdev(throughputs)
            if len(throughputs) > 1
            else 0.0
        ),
        "mean_duration": statistics.mean(durations),
        "peak_allocated_gb": max(memory),
        "workload": trials[0]["workload"],
        "model_version": trials[0]["model_version"],
        "precision": trials[0]["precision"],
        "input_features": trials[0]["input_features"],
        "num_classes": trials[0]["num_classes"],
        "gpu_name": trials[0]["gpu_name"],
        "pytorch_version": trials[0]["pytorch_version"],
        "cuda_version": trials[0]["cuda_version"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Reliable GPU batch-size benchmarking"
    )

    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[128, 256, 512, 1024]
    )

    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--fixed-samples", type=int, default=102400)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--max-vram-gb", type=float, default=7.0)

    args = parser.parse_args()

    if any(batch <= 0 for batch in args.batch_sizes):
        parser.error("All batch sizes must be positive")

    if args.trials < 1:
        parser.error("trials must be positive")

    if args.fixed_samples < 1:
        parser.error("fixed-samples must be positive")

    if args.warmup_steps < 0:
        parser.error("warmup-steps cannot be negative")

    if args.max_vram_gb <= 0:
        parser.error("max-vram-gb must be positive")

    summaries = []

    for batch_size in args.batch_sizes:
        successful_trials = []

        for trial in range(1, args.trials + 1):
            result = run_trial(
                batch_size=batch_size,
                trial=trial,
                fixed_samples=args.fixed_samples,
                warmup_steps=args.warmup_steps
            )

            if result is not None:
                successful_trials.append(result)

        if len(successful_trials) != args.trials:
            print(
                f"Skipping batch {batch_size}: "
                f"only {len(successful_trials)}/{args.trials} "
                f"trials succeeded."
            )
            continue

        summaries.append(
            summarize_batch(batch_size, successful_trials)
        )

    if not summaries:
        print("\nNo batch size completed all trials.")
        return

    output_path = OUTPUT_DIR / "benchmark_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summaries[0])
        )

        writer.writeheader()
        writer.writerows(summaries)

    print("\n" + "=" * 80)
    print("REPEATED BATCH-SIZE BENCHMARK RESULTS")
    print("=" * 80)

    print(
        f"{'Batch':<10}"
        f"{'Mean samples/s':<20}"
        f"{'Std dev':<15}"
        f"{'Mean time':<15}"
        f"{'Peak VRAM':<12}"
    )

    for result in summaries:
        print(
            f"{result['batch_size']:<10}"
            f"{result['mean_throughput']:<20,.0f}"
            f"{result['std_throughput']:<15,.0f}"
            f"{result['mean_duration']:<15.3f}"
            f"{result['peak_allocated_gb']:<12.3f}"
        )

    eligible = [
        result
        for result in summaries
        if result["peak_allocated_gb"] <= args.max_vram_gb
    ]

    if not eligible:
        print("\nNo batch size met the memory limit.")
        return

    best = max(
        eligible,
        key=lambda result: result["mean_throughput"]
    )

    print("\nRECOMMENDED CONFIGURATION")
    print(f"Batch size: {best['batch_size']}")
    print(
        f"Mean throughput: "
        f"{best['mean_throughput']:,.0f} samples/sec"
    )
    print(
        f"Throughput std dev: "
        f"{best['std_throughput']:,.0f} samples/sec"
    )
    print(
        f"Peak PyTorch allocated memory: "
        f"{best['peak_allocated_gb']:.3f} GB"
    )
    print(f"\nSaved results: {output_path}")


if __name__ == "__main__":
    main()
