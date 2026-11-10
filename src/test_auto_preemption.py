from pathlib import Path

from src.process_manager import ProcessRegistry
from src.scheduler import GPUScheduler, Job

registry = ProcessRegistry()

checkpoint = Path("outputs/checkpoints/auto_preemption.pt")
stop_signal = Path("outputs/stop_training.flag")

stop_signal.unlink(missing_ok=True)
checkpoint.unlink(missing_ok=True)


def start_training(job):
    print(f"START CALLBACK: {job.name}", flush=True)

    process = registry.launch_training(
        job_name=job.name,
        epochs=10,
        steps=1000,
        batch_size=512,
        checkpoint=str(checkpoint),
        resume=checkpoint.exists()
    )

    return process.poll() is None


def stop_training(job):
    print(f"STOP CALLBACK: {job.name}", flush=True)

    exit_code = registry.request_stop(
        job.name,
        stop_signal,
        timeout=30
    )

    return exit_code == 0 and not registry.is_running(job.name)
def start_job(job):
    if job.name == "critical-inference":
        print("SIMULATED INFERENCE STARTED", flush=True)
        return True

    return start_training(job)
scheduler = GPUScheduler(
    capacity_gb=8.0,
    protected_gb=2.0,
    start_callback=start_job,
    stop_callback=stop_training
)

import time

training_job = Job(
    name="flexible-training",
    vram_gb=7.0,
    priority=2,
    preemptible=True
)

critical_job = Job(
    name="critical-inference",
    vram_gb=2.0,
    priority=1
)

print("\n--- SUBMITTING TRAINING JOB ---")
scheduler.submit(training_job)

assert registry.is_running("flexible-training"), "Training failed to start"

print("\n--- WAITING FOR TRAINING TO BEGIN ---")
time.sleep(5)

assert registry.is_running("flexible-training"), "Training exited before preemption"

print("\n--- SUBMITTING CRITICAL INFERENCE ---")
scheduler.submit(critical_job)

scheduler.report()

print("\n--- COMPLETING CRITICAL INFERENCE ---")

scheduler.complete("critical-inference")

print("\n--- CHECKING AUTOMATIC TRAINING RESUMPTION ---")

assert registry.is_running("flexible-training"), \
    "Training did not restart automatically"

scheduler.report()
print("\n--- WAITING FOR RESUMED TRAINING TO COMPLETE ---")

exit_code = registry.wait_for("flexible-training")

if exit_code == 0:
    scheduler.complete("flexible-training")
else:
    print(f"Training exited with error code: {exit_code}")

print("\n--- FINAL SCHEDULER STATE ---")
scheduler.report()