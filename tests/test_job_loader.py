from src.job_loader import load_jobs

jobs = load_jobs("configs/jobs.json")

print(f"Loaded {len(jobs)} jobs")

for job in jobs:
    print(
        f"{job.name}: "
        f"VRAM={job.vram_gb} GB, "
        f"priority={job.priority}, "
        f"epochs={job.epochs}, "
        f"steps={job.steps}, "
        f"batch_size={job.batch_size}"
    )