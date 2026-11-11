from src.scheduler import GPUScheduler, Job


def stop_training(job):
    print(f"CALLBACK: stopping {job.name}")
    return True


scheduler = GPUScheduler(
    capacity_gb=8.0,
    protected_gb=2.0,
    stop_callback=stop_training
)

scheduler.submit(
    Job(
        "flexible-training",
        7,
        priority=2,
        preemptible=True
    )
)

print("\n--- BEFORE CRITICAL JOB ---")
scheduler.report()

scheduler.submit(
    Job("critical-inference", 2, priority=1)
)

print("\n--- AFTER CRITICAL JOB ---")
scheduler.report()