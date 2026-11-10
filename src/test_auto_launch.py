from pathlib import Path

from src.process_manager import ProcessRegistry
from src.scheduler import GPUScheduler, Job

registry = ProcessRegistry()

checkpoint = Path("outputs/checkpoints/auto_training.pt")


def start_training(job):
    print(f"START CALLBACK: launching {job.name}", flush=True)

    # Resume if a checkpoint already exists
    resume = checkpoint.exists()

    process = registry.launch_training(
        job_name=job.name,
        epochs=3,
        steps=100,
        batch_size=512,
        checkpoint=str(checkpoint),
        resume=resume
    )

    return process.poll() is None


scheduler = GPUScheduler(
    capacity_gb=8.0,
    protected_gb=2.0,
    start_callback=start_training
)

training_job = Job(
    name="flexible-training",
    vram_gb=7.0,
    priority=2,
    preemptible=True
)

scheduler.submit(training_job)

print(
    "Actual training process running:",
    registry.is_running("flexible-training")
)

scheduler.report()

exit_code = registry.wait_for("flexible-training")

if exit_code == 0:
    scheduler.complete("flexible-training")

scheduler.report()