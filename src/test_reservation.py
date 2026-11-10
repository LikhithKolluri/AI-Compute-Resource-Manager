from src.scheduler import GPUScheduler, Job

scheduler = GPUScheduler(
    capacity_gb=8.0,
    protected_gb=2.0
)

scheduler.submit(Job("training-A", 5, priority=2))
scheduler.submit(Job("training-B", 2, priority=3))
scheduler.submit(Job("critical-inference", 2, priority=1))

scheduler.report()