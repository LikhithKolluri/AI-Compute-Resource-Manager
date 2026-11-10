
from src.scheduler import GPUScheduler, Job, Status


launches = []


def fake_start(job):
    launches.append((job.name, job.batch_size))
    print(f"[TEST LAUNCH] {job.name}: batch_size={job.batch_size}")
    return True


scheduler = GPUScheduler(
    capacity_gb=8.0,
    protected_gb=2.0,
    start_callback=fake_start,
)

job = Job(
    name="rollback-test",
    vram_gb=4.0,
    priority=2,
    distributed=True,
    workers=2,
    batch_size=256,
    original_batch_size=128,
    optimize_batch_size=True,
)

scheduler.submit(job)

assert job.status == Status.RUNNING
assert launches == [("rollback-test", 256)]

recovered = scheduler.rollback_optimization("rollback-test")

assert recovered is True
assert job.batch_size == 128
assert job.rollback_attempted is True
assert job.optimize_batch_size is False
assert job.status == Status.RUNNING
assert launches == [
    ("rollback-test", 256),
    ("rollback-test", 128),
]

print("[TEST PASSED] Distributed rollback restored and relaunched the job")
