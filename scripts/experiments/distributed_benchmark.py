
import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


OUTPUT_DIR = Path("outputs/distributed/benchmarks")


def run_trial(workers, epochs, batch_size, samples, trial):
    report_path = (
        OUTPUT_DIR
        / f"workers_{workers}_batch_{batch_size}_trial_{trial}.json"
    )

    checkpoint_path = (
        OUTPUT_DIR
        / f"workers_{workers}_batch_{batch_size}_trial_{trial}.pt"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path.unlink(missing_ok=True)

    command = [
        sys.executable,
        "-m", "src.distributed_launcher",
        "--workers", str(workers),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--samples", str(samples),
        "--max-retries", "0",
        "--checkpoint", str(checkpoint_path),
        "--report", str(report_path),
        "--stop-signal", str(
            OUTPUT_DIR
            / f"workers_{workers}_batch_{batch_size}_trial_{trial}.flag"
        ),
    ]

    print(
        f"\n[BENCHMARK] Workers={workers}, "
        f"batch={batch_size}, trial={trial}",
        flush=True
    )

    start = time.perf_counter()
    result = subprocess.run(command, check=False)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        raise RuntimeError(
            f"Benchmark failed: workers={workers}, "
            f"batch={batch_size}, trial={trial}, "
            f"exit={result.returncode}"
        )

    if not report_path.exists():
        raise RuntimeError("Distributed report was not generated")

    report = json.loads(report_path.read_text(encoding="utf-8"))

    if report["status"] != "COMPLETED":
        raise RuntimeError(
            f"Distributed job did not complete: {report['status']}"
        )

    total_samples = samples * epochs

    return {
        "workers": workers,
        "trial": trial,
        "epochs": epochs,
        "dataset_samples": samples,
        "batch_size_per_worker": batch_size,
        "global_batch_size": workers * batch_size,
        "total_samples": total_samples,
        "duration_seconds": round(elapsed, 4),
        "samples_per_second": round(total_samples / elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["scaling", "batch"],
        default="batch"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[32, 64, 128, 256]
    )

    args = parser.parse_args()

    if min(
        args.epochs,
        args.batch_size,
        args.samples,
        args.trials,
        args.workers,
        *args.batch_sizes
    ) < 1:
        parser.error("All benchmark values must be positive")

    if args.mode == "scaling":
        configurations = [
            (1, args.batch_size),
            (2, args.batch_size),
        ]
        output = OUTPUT_DIR / "comparison.csv"
    else:
        configurations = [
            (args.workers, batch)
            for batch in args.batch_sizes
        ]
        output = OUTPUT_DIR / "batch_comparison.csv"

    rows = []

    for workers, batch_size in configurations:
        for trial in range(1, args.trials + 1):
            rows.append(
                run_trial(
                    workers=workers,
                    epochs=args.epochs,
                    batch_size=batch_size,
                    samples=args.samples,
                    trial=trial,
                )
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n--- DISTRIBUTED BENCHMARK RESULTS ---")

    summaries = []

    for workers, batch_size in configurations:
        group = [
            row for row in rows
            if row["workers"] == workers
            and row["batch_size_per_worker"] == batch_size
        ]

        mean_duration = statistics.mean(
            row["duration_seconds"] for row in group
        )

        mean_throughput = statistics.mean(
            row["samples_per_second"] for row in group
        )

        summaries.append({
            "workers": workers,
            "batch_size": batch_size,
            "mean_duration": mean_duration,
            "mean_throughput": mean_throughput,
        })

        print(
            f"Workers={workers}, "
            f"batch/worker={batch_size}, "
            f"global batch={workers * batch_size}: "
            f"mean time={mean_duration:.3f}s, "
            f"mean throughput={mean_throughput:.1f} samples/s"
        )

    if args.mode == "scaling":
        one = next(
            item for item in summaries if item["workers"] == 1
        )
        two = next(
            item for item in summaries if item["workers"] == 2
        )

        speedup = one["mean_duration"] / two["mean_duration"]

        print(f"\n2-worker speedup: {speedup:.3f}x")

    else:
        best = max(
            summaries,
            key=lambda item: item["mean_throughput"]
        )

        print(
            f"\n[BENCHMARK] Best measured batch size per worker: "
            f"{best['batch_size']}"
        )
        print(
            f"[BENCHMARK] Mean end-to-end throughput: "
            f"{best['mean_throughput']:.1f} samples/s"
        )

    print(f"\nCSV saved: {output}")


if __name__ == "__main__":
    main()
