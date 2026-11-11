from src.scheduler import GPUScheduler, Job

scheduler = GPUScheduler(capacity_gb=8.0, protected_gb=0.0)

# Fill the GPU first so new jobs must wait
scheduler.submit(Job("blocking-training", 8, priority=2))

# Submit competing jobs in reverse priority order
scheduler.submit(Job("research-training", 4, priority=3))
scheduler.submit(Job("fine-tuning", 3, priority=2))
scheduler.submit(Job("critical-inference", 2, priority=1))

print("\n--- BEFORE GPU IS RELEASED ---")
scheduler.report()

# Free the GPU and trigger scheduling
scheduler.complete("blocking-training")

print("\n--- AFTER GPU IS RELEASED ---")
scheduler.report()