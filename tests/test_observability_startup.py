
from src.scheduler import GPUScheduler, Job, Status


def fail_to_start(job):
    return False


scheduler = GPUScheduler(
    capacity_gb=8.0,
    protected_gb=2.0,
    start_callback=fail_to_start
)

job = Job("startup-failure-test", 4.0, priority=2)

scheduler.submit(job)

for _ in range(2):
    scheduler.schedule()

assert job.status == Status.FAILED
assert scheduler.reserved_gb == 0

print("Startup failure test PASSED")
