import time

from src.process_manager import ProcessRegistry
from src.scheduler import GPUScheduler, Job, Status

registry = ProcessRegistry()


def start_training(job):
    print(f"START CALLBACK: {job.name}", flush=True)

    process = registry.launch_training(
        job_name=job.name,
        epochs=job.epochs,
        steps=job.steps,
        batch_size=job.batch_size,
        checkpoint=f"outputs/checkpoints/{job.name}.pt",
        resume=False
    )

    return process.poll() is None


scheduler = GPUScheduler(
    capacity_gb=8.0,
    protected_gb=2.0,
    start_callback=start_training
)

training_a = Job(
    name="training-A",
    vram_gb=5.0,
    priority=2,
    epochs=3,
    steps=100,
    batch_size=256
)

training_b = Job(
    name="training-B",
    vram_gb=4.0,
    priority=2,
    epochs=5,
    steps=200,
    batch_size=512
)

print("\n--- SUBMITTING TWO JOBS ---")

scheduler.submit_jobs([training_a, training_b])

scheduler.report()

print("\n--- MONITORING PROCESSES ---")

registry.monitor_until_done(scheduler, interval=1)

print("\n--- FINAL SCHEDULER STATE ---")
scheduler.report()