from src.scheduler import GPUScheduler, Job

scheduler = GPUScheduler(
    capacity_gb=8.0,
    protected_gb=2.0
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
print("\n--- CRITICAL INFERENCE COMPLETES ---")

scheduler.complete("critical-inference")

print("\n--- AFTER INFERENCE COMPLETES ---")
scheduler.report()