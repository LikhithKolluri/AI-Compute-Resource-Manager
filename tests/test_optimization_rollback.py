
import unittest

from src.scheduler import GPUScheduler, Job, Status
from src.process_manager import ProcessRegistry


class FakeProcess:
    def __init__(self, pid):
        self.pid = pid
        self.exit_code = None

    def poll(self):
        return self.exit_code


class RollbackTests(unittest.TestCase):

    def make_system(self):
        registry = ProcessRegistry()
        launched = []

        def start_job(job):
            process = FakeProcess(pid=1000 + len(launched))
            launched.append((job.batch_size, process))
            registry.register(job.name, process)
            return True

        scheduler = GPUScheduler(
            capacity_gb=8.0,
            protected_gb=2.0,
            start_callback=start_job,
        )

        job = Job(
            name="rollback-test",
            vram_gb=5.0,
            priority=2,
            batch_size=1024,
            original_batch_size=256,
            optimize_batch_size=True,
        )

        scheduler.submit(job)
        return scheduler, registry, job, launched

    def test_failed_optimization_recovers(self):
        scheduler, registry, job, launched = self.make_system()

        self.assertEqual(job.status, Status.RUNNING)
        self.assertEqual(launched[0][0], 1024)

        # Simulate the optimized process failing.
        launched[0][1].exit_code = 1
        registry.sync_completed(scheduler)

        self.assertTrue(job.rollback_attempted)
        self.assertEqual(job.batch_size, 256)
        self.assertEqual(job.status, Status.RUNNING)
        self.assertEqual(len(launched), 2)
        self.assertEqual(launched[1][0], 256)

        # Simulate the original configuration succeeding.
        launched[1][1].exit_code = 0
        registry.sync_completed(scheduler)

        self.assertEqual(job.status, Status.COMPLETED)
        self.assertEqual(scheduler.reserved_gb, 0)

    def test_second_failure_is_permanent(self):
        scheduler, registry, job, launched = self.make_system()

        launched[0][1].exit_code = 1
        registry.sync_completed(scheduler)

        launched[1][1].exit_code = 1
        registry.sync_completed(scheduler)

        self.assertEqual(job.status, Status.FAILED)
        self.assertEqual(len(launched), 2)
        self.assertEqual(scheduler.reserved_gb, 0)

    def test_nonoptimized_failure_does_not_retry(self):
        registry = ProcessRegistry()
        launched = []

        def start_job(job):
            process = FakeProcess(pid=2000)
            launched.append(process)
            registry.register(job.name, process)
            return True

        scheduler = GPUScheduler(
            start_callback=start_job
        )

        job = Job(
            name="normal-test",
            vram_gb=3.0,
            batch_size=512
        )

        scheduler.submit(job)

        launched[0].exit_code = 1
        registry.sync_completed(scheduler)

        self.assertEqual(job.status, Status.FAILED)
        self.assertEqual(len(launched), 1)


if __name__ == "__main__":
    unittest.main()
