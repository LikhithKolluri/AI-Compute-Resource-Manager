
from unittest.mock import patch

from src.scheduler import GPUScheduler, Job, Status
from src.process_manager import ProcessRegistry


launches = []


class FakeProcess:
    def __init__(self):
        self.pid = 99998

    def poll(self):
        return 0


def fake_start(job):
    launches.append(job.batch_size)
    registry.register(job.name, FakeProcess())
    return True


registry = ProcessRegistry()

scheduler = GPUScheduler(
    capacity_gb=8.0,
    protected_gb=2.0,
    start_callback=fake_start,
)

job = Job(
    name="distributed-inconclusive-test",
    vram_gb=4.0,
    priority=2,
    distributed=True,
    workers=2,
    samples=4096,
    batch_size=256,
    original_batch_size=128,
    optimize_batch_size=True,
)

with (
    patch(
        "src.process_manager.find_distributed_baseline",
        return_value={"baseline_throughput": 7824.08},
    ),
    patch(
        "src.process_manager.read_distributed_metrics",
        return_value=7600.0,
    ),
    patch(
        "src.process_manager.record_distributed_decision",
    ) as record_mock,
    patch(
        "src.process_manager.Path.read_text",
        return_value='{"status":"COMPLETED","attempts":[{"status":"COMPLETED"}]}',
    ),
):
    scheduler.submit(job)
    registry.sync_completed(scheduler)

    assert record_mock.call_count == 1
    assert record_mock.call_args.kwargs["decision"] == "INCONCLUSIVE"

    assert job.status == Status.COMPLETED
    assert job.batch_size == 256
    assert job.rollback_attempted is False
    assert launches == [256]
    assert job.name not in registry.processes
    assert not scheduler.running

print("[TEST PASSED] INCONCLUSIVE decision completed without rollback")
