import sys
import argparse
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import torch.multiprocessing as mp


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def worker(
    rank, world_size, port, epochs, batch_size, samples,
    fail_rank, fail_epoch, checkpoint, resume, stop_signal, metrics_path
):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["USE_LIBUV"] = "0"

    from src.distributed_train import main
    import sys

    sys.argv = [
        "distributed_train",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--samples", str(samples),
        "--fail-rank", str(fail_rank),
        "--fail-epoch", str(fail_epoch),
        "--checkpoint", str(checkpoint),
        "--stop-signal", str(stop_signal)
    ]

    if resume:
        sys.argv.append("--resume")
    if metrics_path is not None:
        sys.argv.extend(["--metrics-path", str(metrics_path)])

    print(
        f"[SUPERVISOR] Worker rank={rank} started, "
        f"PID={os.getpid()}, resume={resume}",
        flush=True
    )

    main()


def save_report(report, path):
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = report_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )
    os.replace(temporary_path, report_path)

    print(f"[SUPERVISOR] Report saved: {report_path}", flush=True)


def launch_attempt(args, attempt_number, resume, fail_rank):
    port = find_free_port()

    print(
        f"\n[RECOVERY] Attempt {attempt_number}: "
        f"workers={args.workers}, resume={resume}",
        flush=True
    )

    context = None

    attempt_report = {
        "attempt": attempt_number,
        "resume": resume,
        "status": "RUNNING",
        "workers": [],
        "error": None,
    }

    try:
        context = mp.start_processes(
            worker,
            args=(
                args.workers,
                port,
                args.epochs,
                args.batch_size,
                args.samples,
                fail_rank,
                args.fail_epoch,
                args.checkpoint,
                resume,
                args.stop_signal,
                args.metrics_path,
            ),
            nprocs=args.workers,
            join=False,
            start_method="spawn",
        )

        for rank, process in enumerate(context.processes):
            attempt_report["workers"].append({
                "rank": rank,
                "pid": process.pid,
                "status": "RUNNING",
                "exit_code": None,
            })

            print(
                f"[SUPERVISOR] Tracking rank={rank}, "
                f"PID={process.pid}",
                flush=True
            )

        while True:
            finished = context.join(timeout=1)

            if finished:
                attempt_report["status"] = "COMPLETED"
                print(
                    "[SUPERVISOR] All workers completed.",
                    flush=True
                )
                break

    except Exception as exc:
        attempt_report["status"] = "FAILED"
        attempt_report["error"] = str(exc)

        print(
            f"[RECOVERY] Attempt {attempt_number} failed: {exc}",
            flush=True
        )

    finally:
        if context is not None:
            if attempt_report["status"] != "COMPLETED":
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()

            for process in context.processes:
                process.join(timeout=5)

            for entry, process in zip(
                attempt_report["workers"], context.processes
            ):
                entry["exit_code"] = process.exitcode

                if process.exitcode is None:
                    entry["status"] = "UNCONFIRMED"
                elif process.exitcode == 0:
                    entry["status"] = "COMPLETED"
                else:
                    entry["status"] = "FAILED"

            if any(
                entry["exit_code"] != 0
                for entry in attempt_report["workers"]
            ):
                attempt_report["status"] = "FAILED"

    return attempt_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--fail-rank", type=int, default=-1)
    parser.add_argument("--fail-epoch", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--persistent-failure", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", default="outputs/distributed/checkpoint.pt")
    parser.add_argument("--report", default="outputs/distributed/worker_report.json")
    parser.add_argument("--stop-signal", default="outputs/distributed/stop.flag")
    parser.add_argument("--metrics-path", default=None)
    
    args = parser.parse_args()
    
    print(
        f"[SUPERVISOR] Supervisor PID={os.getpid()}",
        flush=True
    )
    
    windows_job_handle = None

    if sys.platform == "win32":
        from src.windows_job_object import create_kill_on_close_job

        windows_job_handle = create_kill_on_close_job()

    if args.workers < 1 or args.epochs < 1:
        parser.error("workers and epochs must be positive")

    if args.batch_size < 1 or args.samples < args.workers:
        parser.error("invalid batch-size or sample count")

    if args.max_retries < 0:
        parser.error("max-retries cannot be negative")

    if args.fail_rank < -1 or args.fail_rank >= args.workers:
        parser.error("fail-rank must be -1 or a valid rank")

    if args.fail_epoch < 1 or args.fail_epoch > args.epochs:
        parser.error("fail-epoch must be within the epoch range")

    stop_path = Path(args.stop_signal)
    stop_path.parent.mkdir(parents=True, exist_ok=True)
    stop_path.unlink(missing_ok=True)

    checkpoint_path = Path(args.checkpoint)

    # A new run must not accidentally resume from an old checkpoint.
    if not args.resume:
        checkpoint_path.unlink(missing_ok=True)

    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "world_size": args.workers,
        "max_retries": args.max_retries,
        "status": "RUNNING",
        "attempts": [],
        "error": None,
    }

    start_time = time.perf_counter()
    successful = False

    try:
        for attempt_index in range(args.max_retries + 1):
            attempt_number = attempt_index + 1

            resume = (
                args.resume or attempt_index > 0
            ) and checkpoint_path.exists()

            # Controlled failure occurs only on the first attempt,
            # unless persistent failure is explicitly requested.
            inject_failure = (
                args.fail_rank
                if attempt_index == 0 or args.persistent_failure
                else -1
            )

            attempt_report = launch_attempt(
                args,
                attempt_number,
                resume,
                inject_failure
            )
            
            if stop_path.exists() and attempt_report["status"] == "COMPLETED":
                report["status"] = "STOPPED"
                report["attempts"].append(attempt_report)
                successful = True
                break

            report["attempts"].append(attempt_report)

            if attempt_report["status"] == "COMPLETED":
                successful = True
                report["status"] = "COMPLETED"
                break

            report["error"] = attempt_report["error"]

            if attempt_index < args.max_retries:
                print(
                    "[RECOVERY] Restarting all workers...",
                    flush=True
                )

        if not successful:
            report["status"] = "FAILED"
            print(
                "[RECOVERY] Retry limit exhausted.",
                flush=True
            )

    finally:
        report["duration_seconds"] = round(
            time.perf_counter() - start_time, 3
        )
        save_report(report, args.report)

    if not successful:
        raise SystemExit(1)

    if report["status"] == "STOPPED":
        print(
            "[SHUTDOWN] Distributed training stopped successfully.",
            flush=True
        )
    else:
        print(
            "[RECOVERY] Distributed training completed successfully.",
            flush=True
        )


if __name__ == "__main__":
    main()
