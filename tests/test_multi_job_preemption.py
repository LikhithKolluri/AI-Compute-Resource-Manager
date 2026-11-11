from pathlib import Path

from src.scheduler import GPUScheduler, Job, Status
from src.process_manager import ProcessRegistry
registry = ProcessRegistry()

training_a = Job(
    name="training-A",
    vram_gb=3.0,
    priority=2,
    preemptible=True,
    epochs=10,
    steps=1000,
    batch_size=256
)

training_b = Job(
    name="training-B",
    vram_gb=4.0,
    priority=2,
    preemptible=True,
    epochs=10,
    steps=1000,
    batch_size=256
)

critical_job = Job(
    name="critical-inference",
    vram_gb=2.0,
    priority=1
)
def get_stop_signal(job):
    return Path("outputs/stop_signals") / f"{job.name}.flag"
def start_job(job):
    if job.priority == 1:
        print(f"SIMULATED INFERENCE STARTED: {job.name}", flush=True)
        return True

    checkpoint = Path("outputs/checkpoints") / f"{job.name}.pt"
    stop_signal = get_stop_signal(job)

    stop_signal.parent.mkdir(parents=True, exist_ok=True)
    stop_signal.unlink(missing_ok=True)

    resume = (
        job.status == Status.PAUSED
        and checkpoint.exists()
    )

    print(
        f"STARTING {job.name}, resume={resume}",
        flush=True
    )

    process = registry.launch_training(
        job_name=job.name,
        epochs=job.epochs,
        steps=job.steps,
        batch_size=job.batch_size,
        checkpoint=str(checkpoint),
        resume=resume,
        stop_signal=str(stop_signal)
    )

    return process.poll() is None
def stop_job(job):
    print(f"STOPPING {job.name}", flush=True)

    exit_code = registry.request_stop(
        job.name,
        get_stop_signal(job),
        timeout=60
    )

    return exit_code == 0 and not registry.is_running(job.name)
scheduler = GPUScheduler(
    capacity_gb=8.0,
    protected_gb=2.0,
    start_callback=start_job,
    stop_callback=stop_job
)

import time

# Start Training-A first
print("\n--- SUBMITTING TRAINING-A ---")
scheduler.submit(training_a)

# Start Training-B second
print("\n--- SUBMITTING TRAINING-B ---")
scheduler.submit(training_b)

print("\n--- INITIAL SCHEDULER STATE ---")
scheduler.report()

assert training_a.status == Status.RUNNING
assert training_b.status == Status.RUNNING
assert training_b.borrowed_gb > 0

# Give both PyTorch processes time to begin training
time.sleep(5)

# Submit critical inference
print("\n--- CRITICAL INFERENCE ARRIVES ---")
scheduler.submit(critical_job)

scheduler.report()

# Verify the scheduler preempted only Training-B
assert training_b.status == Status.PAUSED, (
    "Training-B should have been preempted"
)
assert training_a.status == Status.RUNNING, (
    "Training-A should remain running"
)
assert critical_job.status == Status.RUNNING, (
    "Critical inference should be admitted"
)

assert registry.is_running("training-A"), (
    "Training-A process should still be running"
)
assert not registry.is_running("training-B"), (
    "Training-B process should have stopped"
)

print("\nSUCCESS: Training-B was preempted.")
print("SUCCESS: Training-A was not interrupted.")

# Complete the simulated inference job
print("\n--- CRITICAL INFERENCE COMPLETES ---")
scheduler.complete("critical-inference")

# Scheduler should now resume Training-B
scheduler.report()

assert training_b.status == Status.RUNNING, (
    "Training-B should resume after inference completes"
)

print("\nSUCCESS: Training-B resumed from its checkpoint.")

# Monitor both jobs until completion
print("\n--- MONITORING TRAINING JOBS ---")
registry.monitor_until_done(scheduler, interval=1)

print("\n--- FINAL SCHEDULER STATE ---")
scheduler.report()

assert training_a.status == Status.COMPLETED
assert training_b.status == Status.COMPLETED
assert critical_job.status == Status.COMPLETED

print("\nSUCCESS: Multi-job preemption test passed!")