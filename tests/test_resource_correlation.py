
import unittest
from datetime import datetime, timezone

from src.resource_correlation import correlate_resources


class ResourceCorrelationTests(unittest.TestCase):

    def test_reservation_changes_over_time(self):
        def event(seconds, reserved):
            return {
                "timestamp": datetime.fromtimestamp(
                    seconds, tz=timezone.utc
                ),
                "reserved_vram_gb": reserved,
            }

        samples = [
            {
                "timestamp": 100.0,
                "gpu_utilization_pct": 40.0,
                "vram_used_mb": 512.0,
            },
            {
                "timestamp": 110.0,
                "gpu_utilization_pct": 60.0,
                "vram_used_mb": 768.0,
            },
            {
                "timestamp": 120.0,
                "gpu_utilization_pct": 80.0,
                "vram_used_mb": 1024.0,
            },
        ]

        events = [
            event(105, 4.0),
            event(115, 2.0),
        ]

        results = correlate_resources(samples, events)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["reserved_vram_gb"], 4.0)
        self.assertEqual(results[1]["reserved_vram_gb"], 2.0)

        self.assertAlmostEqual(
            results[0]["actual_vram_gb"], 0.75
        )
        self.assertAlmostEqual(
            results[1]["actual_vram_gb"], 1.0
        )

    def test_no_matching_events(self):
        samples = [{
            "timestamp": 100.0,
            "gpu_utilization_pct": 50.0,
            "vram_used_mb": 512.0,
        }]

        events = [{
            "timestamp": datetime.fromtimestamp(
                200, tz=timezone.utc
            ),
            "reserved_vram_gb": 4.0,
        }]

        self.assertEqual(
            correlate_resources(samples, events), []
        )

    
    def test_monitoring_window(self):
        def event(seconds, reserved):
            return {
                "timestamp": datetime.fromtimestamp(
                    seconds, tz=timezone.utc
                ),
                "reserved_vram_gb": reserved,
            }

        samples = [
            {
                "timestamp": 100.0,
                "gpu_utilization_pct": 20.0,
                "vram_used_mb": 256.0,
            },
            {
                "timestamp": 110.0,
                "gpu_utilization_pct": 60.0,
                "vram_used_mb": 512.0,
            },
            {
                "timestamp": 120.0,
                "gpu_utilization_pct": 80.0,
                "vram_used_mb": 768.0,
            },
        ]

        events = [event(105, 4.0)]

        results = correlate_resources(
            samples,
            events,
            start_time=105.0,
            end_time=115.0,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["timestamp"], 110.0)
        self.assertEqual(results[0]["reserved_vram_gb"], 4.0)

    
    def test_session_id_filtering(self):
        samples = [
            {
                "timestamp": 110.0,
                "session_id": "session-A",
                "gpu_utilization_pct": 60.0,
                "vram_used_mb": 512.0,
            },
            {
                "timestamp": 120.0,
                "session_id": "session-B",
                "gpu_utilization_pct": 80.0,
                "vram_used_mb": 768.0,
            },
        ]

        events = [
            {
                "timestamp": datetime.fromtimestamp(
                    105, tz=timezone.utc
                ),
                "session_id": "session-A",
                "reserved_vram_gb": 4.0,
            },
            {
                "timestamp": datetime.fromtimestamp(
                    115, tz=timezone.utc
                ),
                "session_id": "session-B",
                "reserved_vram_gb": 2.0,
            },
        ]

        results = correlate_resources(
            samples,
            events,
            session_id="session-A",
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["timestamp"], 110.0)
        self.assertEqual(results[0]["reserved_vram_gb"], 4.0)
        self.assertEqual(results[0]["actual_vram_gb"], 0.5)
        
    
    def test_initial_reservation_and_completion(self):
        samples = [
            {
                "timestamp": 100.0,
                "session_id": "session-A",
                "gpu_utilization_pct": 10.0,
                "vram_used_mb": 256.0,
            },
            {
                "timestamp": 110.0,
                "session_id": "session-A",
                "gpu_utilization_pct": 70.0,
                "vram_used_mb": 768.0,
            },
            {
                "timestamp": 120.0,
                "session_id": "session-A",
                "gpu_utilization_pct": 15.0,
                "vram_used_mb": 256.0,
            },
        ]

        events = [
            {
                "timestamp": datetime.fromtimestamp(
                    105, tz=timezone.utc
                ),
                "session_id": "session-A",
                "event": "ADMITTED",
                "reserved_vram_gb": 4.0,
            },
            {
                "timestamp": datetime.fromtimestamp(
                    115, tz=timezone.utc
                ),
                "session_id": "session-A",
                "event": "COMPLETED",
                "reserved_vram_gb": 0.0,
            },
        ]

        results = correlate_resources(
            samples,
            events,
            session_id="session-A",
            initial_reserved_gb=0.0,
        )

        self.assertEqual(len(results), 3)

        self.assertEqual(
            [row["reserved_vram_gb"] for row in results],
            [0.0, 4.0, 0.0],
        )

if __name__ == "__main__":
    unittest.main()
