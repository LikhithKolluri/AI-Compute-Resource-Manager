import argparse
import time
from pathlib import Path

from src.scheduler import GPUScheduler, Job
from src.process_manager import ProcessRegistry
from src.gpu_monitor import GPUMonitor



def main():
    parser = argparse.ArgumentParser(
        description="Run monitored GPU training"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    registry = ProcessRegistry()


    def start_training(job):
        registry.launch_training(
            job_name=job.name,
            epochs=job.epochs,
            steps=job.steps,
            batch_size=job.batch_size,
            checkpoint=("outputs/checkpoints/session_training.pt"),
            resume=False,
            stop_signal=("outputs/session_training.flag"),
            session_id=session_id,
        )
        return True

    scheduler = GPUScheduler(
        capacity_gb=8.0,
        protected_gb=2.0,
        start_callback=start_training,
    )

    session_id = scheduler.event_logger.session_id

    output_path = Path(
        f"outputs/analytics/gpu_{session_id}.csv"
    )

    stop_signal = Path(
        "outputs/session_training.flag"
    )
    stop_signal.unlink(missing_ok=True)

    monitor = GPUMonitor(
        output_path=str(output_path),
        gpu_index=0,
        interval=0.5,
        append=False,
        session_id=session_id,
    )

    job = Job(
        name="session-training",
        vram_gb=4.0,
        priority=2,
        epochs=3,
        steps=1000,
        batch_size=args.batch_size,
    )

    print(f"Session ID: {session_id}")
    print(f"Batch size: {args.batch_size}")

    monitor.start()

    try:
        print("Collecting baseline GPU measurements...")
        time.sleep(7) 

        print("Starting training workload...")
        scheduler.submit(job)

        while registry.processes:
            registry.sync_completed(scheduler)
            time.sleep(0.5)

        if scheduler.running or scheduler.waiting:
            raise RuntimeError(
                "Training did not reach a terminal state"
            )

        if job.status.value != "COMPLETED":
            raise RuntimeError(
                f"Training ended with status {job.status}"
            )
        print("Collecting post-training GPU measurements...")
        time.sleep(5)

    finally:
        monitor.stop()

    print(f"Final job status: {job.status.value}")
    print(f"Final reservation: {scheduler.reserved_gb} GB")
    print(f"GPU samples: {len(monitor.samples)}")
    print(f"GPU metrics saved: {output_path}")


if __name__ == "__main__":
    main()
