
import unittest
from datetime import datetime, timedelta, timezone

from src.scheduling_analytics import analyze_jobs, calculate_live_metrics


class SchedulingAnalyticsTests(unittest.TestCase):

    def test_preemption_and_resume_metrics(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        job_id = "test-job-001"

        def event(minutes, event_type):
            return {
                "job_id": job_id,
                "job_name": "training",
                "event": event_type,
                "timestamp": start + timedelta(minutes=minutes),
            }

        events = [
            event(0, "SUBMITTED"),
            event(5, "QUEUED"),
            event(5, "ADMITTED"),
            event(10, "PREEMPTED"),
            event(15, "ADMITTED"),
            event(25, "COMPLETED"),
        ]

        result = analyze_jobs(events)[job_id]

        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["queue_seconds"], 600)
        self.assertEqual(result["execution_seconds"], 900)
        self.assertEqual(result["preemptions"], 1)

        turnaround = (
            result["finished_at"] - result["submitted_at"]
        ).total_seconds()

        self.assertEqual(turnaround, 1500)
        
    
    def test_rejected_job(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)

        events = [
            {
                "job_id": "rejected-001",
                "job_name": "invalid-training",
                "event": "REJECTED",
                "timestamp": start,
            }
        ]

        result = analyze_jobs(events)["rejected-001"]

        self.assertEqual(result["status"], "REJECTED")
        self.assertEqual(result["queue_seconds"], 0)
        self.assertEqual(result["execution_seconds"], 0)
        self.assertEqual(result["preemptions"], 0)

    def test_failed_job(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)

        def event(seconds, event_type):
            return {
                "job_id": "failed-001",
                "job_name": "failed-training",
                "event": event_type,
                "timestamp": start + timedelta(seconds=seconds),
            }

        events = [
            event(0, "SUBMITTED"),
            event(5, "ADMITTED"),
            event(15, "FAILED"),
        ]

        result = analyze_jobs(events)["failed-001"]

        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["queue_seconds"], 5)
        self.assertEqual(result["execution_seconds"], 10)

    def test_same_name_different_job_ids(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)

        events = [
            {
                "job_id": "run-A",
                "job_name": "training",
                "event": "SUBMITTED",
                "timestamp": start,
            },
            {
                "job_id": "run-A",
                "job_name": "training",
                "event": "ADMITTED",
                "timestamp": start + timedelta(seconds=2),
            },
            {
                "job_id": "run-A",
                "job_name": "training",
                "event": "COMPLETED",
                "timestamp": start + timedelta(seconds=12),
            },
            {
                "job_id": "run-B",
                "job_name": "training",
                "event": "SUBMITTED",
                "timestamp": start + timedelta(seconds=20),
            },
            {
                "job_id": "run-B",
                "job_name": "training",
                "event": "ADMITTED",
                "timestamp": start + timedelta(seconds=25),
            },
            {
                "job_id": "run-B",
                "job_name": "training",
                "event": "COMPLETED",
                "timestamp": start + timedelta(seconds=45),
            },
        ]

        results = analyze_jobs(events)

        self.assertEqual(len(results), 2)
        self.assertEqual(results["run-A"]["queue_seconds"], 2)
        self.assertEqual(results["run-A"]["execution_seconds"], 10)
        self.assertEqual(results["run-B"]["queue_seconds"], 5)
        self.assertEqual(results["run-B"]["execution_seconds"], 20)

    def test_live_running_duration(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)

        events = [
            {
                "job_id": "live-001",
                "job_name": "training",
                "event": "SUBMITTED",
                "timestamp": start,
            },
            {
                "job_id": "live-001",
                "job_name": "training",
                "event": "ADMITTED",
                "timestamp": start + timedelta(seconds=5),
            },
        ]

        job = analyze_jobs(events)["live-001"]

        snapshot = calculate_live_metrics(
            job,
            as_of=start + timedelta(seconds=35)
        )

        self.assertEqual(snapshot["queue_seconds"], 5)
        self.assertEqual(snapshot["execution_seconds"], 30)


if __name__ == "__main__":
    unittest.main()
