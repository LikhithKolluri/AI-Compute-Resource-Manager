import argparse
import time
from pathlib import Path
from src.job_loader import load_jobs
from src.scheduler import GPUScheduler, Status, Job
from src.process_manager import ProcessRegistry
from src.benchmark_advisor import (recommend_batch_size, print_recommendation,recommend_distributed_batch_size,
    print_distributed_recommendation,)
from src.batch_optimizer import optimize_job_batch_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/jobs.json")
    parser.add_argument("--test-failure", action="store_true")
    parser.add_argument("--test-shutdown", action="store_true")
    args = parser.parse_args()

    registry = ProcessRegistry()
    
    def start_training(job):
        if job.priority == 1:
            print(f"SIMULATED INFERENCE STARTED: {job.name}", flush=True)
            return True

        print(f"START CALLBACK: {job.name}", flush=True)

        if job.distributed:
            print(
                f"[DISTRIBUTED] Starting {job.name} "
                f"with {job.workers} workers",
                flush=True
            )

            checkpoint = f"outputs/distributed/{job.name}.pt"
            report = f"outputs/distributed/{job.name}_report.json"

            stop_signal = f"outputs/distributed/{job.name}.flag"
            
            process = registry.launch_distributed_training(
                job_name=job.name,
                workers=job.workers,
                epochs=job.epochs,
                batch_size=job.batch_size,
                samples=job.samples,
                max_retries=job.max_retries,
                checkpoint=checkpoint,
                report=report,
                resume=False,
                fail_rank=1 if args.test_failure else -1,
                fail_epoch=2,
                persistent_failure=args.test_failure,
                stop_signal=stop_signal,
            )

            return process.poll() is None

        # Existing standard training path
        checkpoint = f"outputs/checkpoints/{job.name}.pt"

        resume = (
            job.status == Status.PAUSED
            and Path(checkpoint).exists()
        )

        job_stop_signal = Path("outputs/stop_signals") / f"{job.name}.flag"
        job_stop_signal.parent.mkdir(parents=True, exist_ok=True)
        job_stop_signal.unlink(missing_ok=True)

        process = registry.launch_training(
            job_name=job.name,
            epochs=job.epochs,
            steps=job.steps,
            batch_size=job.batch_size,
            checkpoint=checkpoint,
            stop_signal=str(job_stop_signal),
            resume=resume,
            job=job,
            simulate_failure=(
                args.test_failure
                and job.original_batch_size is not None
                and not job.rollback_attempted
            ),
        )

        return process.poll() is None
    
    def stop_training(job):
        print(f"STOP CALLBACK: {job.name}", flush=True)
        if job.distributed:
            print(f"[DISTRIBUTED] Preemption not yet supported: {job.name}", flush=True)
            return False
        
        stop_signal = Path("outputs/stop_signals") / f"{job.name}.flag"
        exit_code = registry.request_stop(
            job.name,
            stop_signal,
            timeout=30
        )

        return exit_code == 0 and not registry.is_running(job.name)
    
    scheduler = GPUScheduler(
        capacity_gb=8.0,
        protected_gb=2.0,
        start_callback=start_training,
        stop_callback= stop_training
    )

    
    jobs = load_jobs(args.config)
    
    print("\n--- BENCHMARK-AWARE JOB RECOMMENDATIONS ---")

    for job in jobs:
        if job.priority == 1:
            continue

        if job.distributed:
            recommendation = recommend_distributed_batch_size(
                current_batch_size=job.batch_size,
                workers=job.workers,
                samples=job.samples,
            )

            print_distributed_recommendation(
                job.name,
                recommendation,
            )

        else:
            recommendation = recommend_batch_size(
                current_batch_size=job.batch_size,
                workload=job.workload,
                model_version=job.model_version,
                precision=job.precision,
            )

            print_recommendation(
                job.name,
                recommendation,
            )

    print("--- END RECOMMENDATIONS ---\n")

    print(f"\nLoaded {len(jobs)} jobs")
    
    print("\n--- AUTOMATIC BATCH OPTIMIZATION ---")


    for job in jobs:
        if job.distributed:
            if not job.optimize_batch_size:
                continue

            recommendation = recommend_distributed_batch_size(
                current_batch_size=job.batch_size,
                workers=job.workers,
                samples=job.samples,
            )

            if recommendation is None:
                print(
                    f"[DISTRIBUTED OPTIMIZATION] {job.name}: "
                    "No recommendation available",
                    flush=True,
                )
                continue

            
            recommended_batch_size = recommendation["recommended_batch_size"]

            if recommended_batch_size == job.batch_size:
                print(
                    f"[DISTRIBUTED OPTIMIZATION] {job.name}: "
                    "Already using recommended batch size",
                    flush=True,
                )
                continue

            job.original_batch_size = job.batch_size
            job.batch_size = recommended_batch_size

            print(
                f"[DISTRIBUTED OPTIMIZATION] {job.name}: "
                f"Batch size {job.original_batch_size} "
                f"-> {job.batch_size} per worker",
                flush=True,
            )
            continue


        optimize_job_batch_size(job)


    print("--- END OPTIMIZATION ---\n")

    scheduler.submit_jobs(jobs)

    # Temporary test for distributed shutdown
    if args.test_shutdown:
        distributed_job = next(
            (job for job in scheduler.running if job.distributed),
            None
        )

        if distributed_job is None:
            raise RuntimeError("No running distributed job to stop")

        print("[TEST] Waiting for distributed training...", flush=True)
        time.sleep(10)

        stop_path = Path(
            f"outputs/distributed/{distributed_job.name}.flag"
        )
        stop_path.touch()

        print(
            f"[TEST] Stop requested for {distributed_job.name}",
            flush=True
        )
    
    if Path(args.config).name == "preemption_jobs.json":
        print("\n--- WAITING FOR TRAINING TO BEGIN ---")
        time.sleep(5)

        critical_job = Job(
            name="critical-inference",
            vram_gb=2.0,
            priority=1
        )

        print("\n--- SUBMITTING CRITICAL INFERENCE ---")
        scheduler.submit(critical_job)

        scheduler.report()

        print("\n--- KEEPING INFERENCE ACTIVE FOR MONITORING ---")
        time.sleep(20)

        print("\n--- COMPLETING SIMULATED INFERENCE ---")
        scheduler.complete("critical-inference")

    print("\n--- INITIAL SCHEDULER STATE ---")
    scheduler.report()

    print("\n--- MONITORING JOBS ---")
    registry.monitor_until_done(scheduler, interval=1)
    
    # Complete simulated inference jobs that have no OS process.
    for job in list(scheduler.running):
        if job.priority == 1 and not registry.is_running(job.name):
            print(f"SIMULATED INFERENCE COMPLETED: {job.name}")
            scheduler.complete(job.name)

    print("\n--- FINAL JOB RESULTS ---")
    for job in jobs:
        print(f"{job.name}: {job.status}")

    scheduler.report()


if __name__ == "__main__":
    main()