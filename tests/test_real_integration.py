import subprocess
import sys
import time
from pathlib import Path
from src.process_manager import ProcessRegistry
from src.scheduler import GPUScheduler, Job

registry = ProcessRegistry()

stop_signal = Path("outputs/stop_training.flag")


def stop_training(job):
    print(f"Stopping actual process for {job.name}")

    exit_code = registry.request_stop(
        job.name,
        stop_signal,
        timeout=30
    )

    return (
        exit_code == 0
        and not registry.is_running(job.name)
    )
    
# Remove any stop signal left over from an earlier test
stop_signal.unlink(missing_ok=True)

# Start a real PyTorch training process
command = [
    sys.executable,
    "-m", "src.train",
    "--epochs", "20",
    "--steps", "1000",
    "--batch-size", "512",
    "--checkpoint", "outputs/checkpoints/managed_training.pt"
]

process = subprocess.Popen(command)

# Register the process so we can manage it
registry.register("flexible-training", process)

print(f"Training running: {registry.is_running('flexible-training')}")

# Give training a few seconds to start
time.sleep(5)
# Connect the scheduler to our real stop callback
scheduler = GPUScheduler(
    capacity_gb=8.0,
    protected_gb=2.0,
    stop_callback=stop_training
)

# Register the training job with the scheduler
training_job = Job(
    name="flexible-training",
    vram_gb=7.0,
    priority=2,
    preemptible=True
)

scheduler.submit(training_job)

# A critical inference job arrives
critical_job = Job(
    name="critical-inference",
    vram_gb=2.0,
    priority=1
)

print("\nCritical inference job arriving...")
scheduler.submit(critical_job)

scheduler.report()

print(
    "Actual training process still running:",
    registry.is_running("flexible-training")
)
print("\nCritical inference completed.")
scheduler.complete("critical-inference")
scheduler.report()
if training_job in scheduler.running:
    print("\nResuming actual training...", flush=True)

    resume_command = [
        sys.executable,
        "-m", "src.train",
        "--epochs", "20",
        "--steps", "1000",
        "--batch-size", "512",
        "--checkpoint", "outputs/checkpoints/managed_training.pt",
        "--resume"
    ]

    resumed_process = subprocess.Popen(resume_command)
    registry.register("flexible-training", resumed_process)

    print(
        "Actual training running after restart:",
        registry.is_running("flexible-training")
    )

    exit_code = registry.wait_for("flexible-training")

    if exit_code == 0:
        scheduler.complete("flexible-training")

scheduler.report()