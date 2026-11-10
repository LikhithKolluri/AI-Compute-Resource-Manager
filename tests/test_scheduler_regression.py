
import unittest

from src.scheduler import GPUScheduler, Job, Status


class SchedulerRegressionTests(unittest.TestCase):

    def test_protected_capacity(self):
        scheduler = GPUScheduler(
            capacity_gb=8,
            protected_gb=2
        )

        training = Job("training", 6, priority=2)
        extra = Job("extra", 1, priority=3)

        scheduler.submit(training)
        scheduler.submit(extra)

        self.assertEqual(training.status, Status.RUNNING)
        self.assertEqual(extra.status, Status.QUEUED)
        self.assertEqual(scheduler.reserved_gb, 6)

    def test_preemptible_job_can_borrow(self):
        scheduler = GPUScheduler(
            capacity_gb=8,
            protected_gb=2
        )

        training = Job(
            "flexible-training",
            7,
            priority=2,
            preemptible=True
        )

        scheduler.submit(training)

        self.assertEqual(training.status, Status.RUNNING)
        self.assertAlmostEqual(training.borrowed_gb, 1)
        self.assertEqual(scheduler.reserved_gb, 7)

    def test_critical_job_preempts_borrower(self):
        stopped = []

        def stop_job(job):
            stopped.append(job.name)
            return True

        scheduler = GPUScheduler(
            capacity_gb=8,
            protected_gb=2,
            stop_callback=stop_job
        )

        training = Job(
            "training",
            7,
            priority=2,
            preemptible=True
        )

        critical = Job(
            "critical-inference",
            2,
            priority=1
        )

        scheduler.submit(training)
        scheduler.submit(critical)

        self.assertEqual(training.status, Status.PAUSED)
        self.assertEqual(critical.status, Status.RUNNING)
        self.assertIn("training", stopped)
        self.assertEqual(scheduler.reserved_gb, 2)

    def test_failed_preemption_keeps_reservation(self):
        scheduler = GPUScheduler(
            capacity_gb=8,
            protected_gb=2,
            stop_callback=lambda job: False
        )

        training = Job(
            "training",
            7,
            priority=2,
            preemptible=True
        )

        critical = Job(
            "critical",
            2,
            priority=1
        )

        scheduler.submit(training)
        scheduler.submit(critical)

        self.assertEqual(training.status, Status.RUNNING)
        self.assertEqual(critical.status, Status.QUEUED)
        self.assertEqual(scheduler.reserved_gb, 7)

    def test_startup_failure_limit(self):
        attempts = []

        def fail_start(job):
            attempts.append(job.name)
            return False

        scheduler = GPUScheduler(
            start_callback=fail_start
        )

        job = Job("broken-job", 3)
        scheduler.submit(job)

        self.assertEqual(job.status, Status.QUEUED)

        scheduler.schedule()
        scheduler.schedule()

        self.assertEqual(len(attempts), 3)
        self.assertEqual(job.status, Status.FAILED)
        self.assertNotIn(job, scheduler.waiting)
        self.assertEqual(scheduler.reserved_gb, 0)

    def test_completion_releases_capacity(self):
        scheduler = GPUScheduler(
            capacity_gb=8,
            protected_gb=2
        )

        first = Job("first", 6)
        second = Job("second", 4)

        scheduler.submit(first)
        scheduler.submit(second)

        self.assertEqual(second.status, Status.QUEUED)

        scheduler.complete("first")

        self.assertEqual(first.status, Status.COMPLETED)
        self.assertEqual(second.status, Status.RUNNING)
        self.assertEqual(scheduler.reserved_gb, 4)

    def test_invalid_job_rejected(self):
        scheduler = GPUScheduler(capacity_gb=8)

        invalid = Job("too-large", 10)
        scheduler.submit(invalid)

        self.assertEqual(invalid.status, Status.REJECTED)
        self.assertNotIn(invalid, scheduler.waiting)
        self.assertNotIn(invalid, scheduler.running)

    def test_higher_priority_waiting_job_runs_first(self):
        scheduler = GPUScheduler(
            capacity_gb=8,
            protected_gb=2
        )

        blocker = Job("blocker", 6, priority=2)
        low = Job("low", 4, priority=3)
        high = Job("high", 4, priority=1)

        scheduler.submit(blocker)
        scheduler.submit(low)
        scheduler.submit(high)

        self.assertEqual(low.status, Status.QUEUED)
        self.assertEqual(high.status, Status.QUEUED)

        scheduler.complete("blocker")

        self.assertEqual(high.status, Status.RUNNING)
        self.assertEqual(low.status, Status.QUEUED)


if __name__ == "__main__":
    unittest.main()
