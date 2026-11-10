from src.scheduler import GPUScheduler, Job

scheduler = GPUScheduler(
    capacity_gb=8.0,
    protected_gb=2.0
)

scheduler.submit(
    Job("flexible-training", 7, priority=2, preemptible=True)
)

scheduler.report()
job = scheduler.running[0]

print(f"Job: {job.name}")
print(f"Borrowed VRAM: {job.borrowed_gb:.1f} GB")