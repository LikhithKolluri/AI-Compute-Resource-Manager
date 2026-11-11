import subprocess
import sys

from src.process_manager import ProcessRegistry
from src.scheduler import GPUScheduler, Job, Status

registry = ProcessRegistry()


def start_job(job):
    if job.name == "training-A":
        print("STARTING: training-A (simulated failure)", flush=True)

        process = subprocess.Popen([
            sys.executable,
            "-c",
            "import sys; sys.exit(1)"
        ])

        registry.register(job.name, process)
        return True

    print(f"STARTING: {job.name}", flush=True)

    process = registry.launch_training(
        job_name=job.name,
        epochs=2,
        steps=100,
        batch_size=512,
        checkpoint=f"outputs/checkpoints/{job.name}.pt",
        resume=False
    )

    return process.poll() is None


scheduler = GPUScheduler(
    capacity_gb=8.0,
    protected_gb=2.0,
    start_callback=start_job
)

training_a = Job(
    name="training-A",
    vram_gb=5.0,
    priority=2
)

training_b = Job(
    name="training-B",
    vram_gb=4.0,
    priority=2
)

print("\n--- SUBMITTING JOBS ---")

scheduler.submit(training_a)
scheduler.submit(training_b)

scheduler.report()

print("\n--- MONITORING ---")

registry.monitor_until_done(scheduler, interval=1)

print("\n--- FINAL RESULTS ---")

print("training-A:", training_a.status)
print("training-B:", training_b.status)

scheduler.report()

assert training_a.status == Status.FAILED
assert training_b.status == Status.COMPLETED
assert scheduler.reserved_gb == 0

print("\nFAILURE RECOVERY TEST PASSED")