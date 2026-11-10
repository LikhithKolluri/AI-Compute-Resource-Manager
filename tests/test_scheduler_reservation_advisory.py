
import unittest
from contextlib import redirect_stdout
from io import StringIO

from src.scheduler import GPUScheduler, Job, Status


class SchedulerReservationAdvisoryTests(unittest.TestCase):

    def test_advisory_does_not_change_reservation(self):
        scheduler = GPUScheduler(
            capacity_gb=8.0,
            protected_gb=2.0,
        )

        job = Job(
            name="advisory-training",
            vram_gb=4.0,
            priority=2,
            recommended_vram_gb=0.5,
        )

        output = StringIO()

        with redirect_stdout(output):
            scheduler.submit(job)

        self.assertEqual(job.status, Status.RUNNING)
        self.assertEqual(scheduler.reserved_gb, 4.0)
        self.assertEqual(scheduler.available_gb, 4.0)

        text = output.getvalue()
        self.assertIn("[ADVISORY]", text)
        self.assertIn("recommended=0.50 GiB", text)
        self.assertIn("reserved=4.00 GiB", text)


if __name__ == "__main__":
    unittest.main()
