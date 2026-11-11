from src.scheduler import GPUScheduler, Job, Status


def start_job(job):
    print(f"START CALLBACK: {job.name}", flush=True)

    if job.name == "training-A":
        print("SIMULATED STARTUP FAILURE", flush=True)
        return False

    print(f"SIMULATED START SUCCESS: {job.name}", flush=True)
    return True


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

print("\n--- SUBMITTING TRAINING-A ---")
scheduler.submit(training_a)

print("\n--- SUBMITTING TRAINING-B ---")
scheduler.submit(training_b)

print("\n--- FINAL SCHEDULER STATE ---")
scheduler.report()

print("\n--- JOB STATUSES ---")
print("training-A:", training_a.status)
print("training-B:", training_b.status)
print("\n--- COMPLETING TRAINING-B ---")

scheduler.complete("training-B")

print("\n--- CHECKING RETRY LIMIT ---")

print("\n--- FINAL RESULTS ---")

print("training-A failures:", training_a.start_failures)
print("training-A status:", training_a.status)
print("training-B status:", training_b.status)

scheduler.report()
assert training_a.start_failures == 3
assert training_a.status == Status.FAILED
assert training_b.status == Status.COMPLETED
assert scheduler.reserved_gb == 0
assert len(scheduler.waiting) == 0

print("\nSTARTUP FAILURE TEST PASSED")