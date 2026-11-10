
from src.scheduler import GPUScheduler, Job, Status

scheduler = GPUScheduler(
    capacity_gb=8.0,
    protected_gb=2.0
)

job = Job(
    name="rollback-test",
    vram_gb=4.0,
    priority=2,
    batch_size=1024,
    original_batch_size=512,
    optimize_batch_size=True
)

scheduler.submit(job)

assert job.status == Status.RUNNING

result = scheduler.rollback_optimization("rollback-test")

assert result is True
assert job.batch_size == 512
assert job.rollback_attempted is True
assert job.optimize_batch_size is False
assert job.status == Status.RUNNING

print("Rollback observability test PASSED")

