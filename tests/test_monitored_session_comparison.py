
import csv
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from scripts.compare_monitored_sessions import (compare_session, export_comparison_csv)


class MonitoredSessionComparisonTests(unittest.TestCase):

    def test_compare_session(self):
        session_id = "test-session"

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            gpu_path = (
                root / "outputs" / "analytics"
                / f"gpu_{session_id}.csv"
            )
            training_path = (
                root / "outputs" / "metrics"
                / "session-training"
                / session_id
                / "training_metrics.csv"
            )

            gpu_path.parent.mkdir(parents=True)
            training_path.parent.mkdir(parents=True)

            with gpu_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "timestamp",
                        "gpu_utilization_pct",
                        "vram_used_mb",
                        "vram_total_mb",
                        "session_id",
                        "gpu_index",
                    ],
                )
                writer.writeheader()
                writer.writerows([
                    {
                        "timestamp": 90,
                        "gpu_utilization_pct": 0,
                        "vram_used_mb": 100,
                        "vram_total_mb": 8192,
                        "session_id": session_id,
                        "gpu_index": 0,
                    },
                    {
                        "timestamp": 110,
                        "gpu_utilization_pct": 40,
                        "vram_used_mb": 300,
                        "vram_total_mb": 8192,
                        "session_id": session_id,
                        "gpu_index": 0,
                    },
                    {
                        "timestamp": 120,
                        "gpu_utilization_pct": 60,
                        "vram_used_mb": 400,
                        "vram_total_mb": 8192,
                        "session_id": session_id,
                        "gpu_index": 0,
                    },
                    {
                        "timestamp": 140,
                        "gpu_utilization_pct": 0,
                        "vram_used_mb": 100,
                        "vram_total_mb": 8192,
                        "session_id": session_id,
                        "gpu_index": 0,
                    },
                ])

            with training_path.open(
                "w", newline="", encoding="utf-8"
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "epoch",
                        "avg_loss",
                        "epoch_seconds",
                        "samples_per_second",
                    ],
                )
                writer.writeheader()
                writer.writerows([
                    {
                        "epoch": 1,
                        "avg_loss": 0.5,
                        "epoch_seconds": 2,
                        "samples_per_second": 1000,
                    },
                    {
                        "epoch": 2,
                        "avg_loss": 0.4,
                        "epoch_seconds": 3,
                        "samples_per_second": 2000,
                    },
                ])

            def event(seconds, event_name):
                return {
                    "timestamp": datetime.fromtimestamp(
                        seconds, tz=timezone.utc
                    ),
                    "event": event_name,
                    "session_id": session_id,
                }

            events = [
                event(100, "ADMITTED"),
                event(130, "COMPLETED"),
            ]

            with patch("pathlib.Path.cwd", return_value=root):
                # Relative paths are resolved from the process
                # working directory, not Path.cwd().
                import os

                original_cwd = os.getcwd()
                try:
                    os.chdir(root)
                    result = compare_session(session_id, events)
                finally:
                    os.chdir(original_cwd)

            self.assertEqual(result["epochs"], 2)
            self.assertAlmostEqual(result["training_seconds"], 5.0)
            self.assertAlmostEqual(result["average_throughput"], 1600.0)
            self.assertAlmostEqual(result["average_gpu_utilization"], 50.0)
            self.assertAlmostEqual(result["peak_gpu_utilization"], 60.0)
            self.assertAlmostEqual(result["estimated_peak_memory_mb"], 300.0)
            self.assertAlmostEqual(
                result["post_training_memory_difference_mb"], 0.0
            )

    
    def test_export_comparison_csv(self):
        results = [
            {
                "session_id": "session-128",
                "epochs": 3,
                "training_seconds": 8.86,
                "average_throughput": 43361.0,
                "average_gpu_utilization": 11.62,
                "peak_gpu_utilization": 29.0,
                "estimated_peak_memory_mb": 208.9,
                "post_training_memory_difference_mb": 0.0,
            },
            {
                "session_id": "session-2048",
                "epochs": 3,
                "training_seconds": 8.77,
                "average_throughput": 700259.0,
                "average_gpu_utilization": 39.31,
                "peak_gpu_utilization": 72.0,
                "estimated_peak_memory_mb": 244.9,
                "post_training_memory_difference_mb": 0.0,
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = (
                Path(temp_dir)
                / "analytics"
                / "batch_comparison.csv"
            )

            returned_path = export_comparison_csv(
                results, output_path
            )

            self.assertEqual(returned_path, output_path)
            self.assertTrue(output_path.exists())

            with output_path.open(
                "r", newline="", encoding="utf-8"
            ) as file:
                reader = csv.DictReader(file)
                exported_rows = list(reader)

                self.assertEqual(
                    reader.fieldnames,
                    list(results[0].keys())
                )

            self.assertEqual(len(exported_rows), 2)

            for actual, expected in zip(exported_rows, results):
                self.assertEqual(
                    actual["session_id"],
                    expected["session_id"]
                )
                self.assertEqual(
                    int(actual["epochs"]),
                    expected["epochs"]
                )

                for field in (
                    "training_seconds",
                    "average_throughput",
                    "average_gpu_utilization",
                    "peak_gpu_utilization",
                    "estimated_peak_memory_mb",
                    "post_training_memory_difference_mb",
                ):
                    self.assertAlmostEqual(
                        float(actual[field]),
                        expected[field]
                    )

    def test_export_empty_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = (
                Path(temp_dir) / "batch_comparison.csv"
            )

            with self.assertRaisesRegex(
                ValueError,
                "No comparison results to export"
            ):
                export_comparison_csv([], output_path)

            self.assertFalse(output_path.exists())

    def test_export_overwrites_existing_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = (
                Path(temp_dir) / "batch_comparison.csv"
            )

            first_results = [
                {"session_id": "old-session", "epochs": 3}
            ]

            updated_results = [
                {"session_id": "new-session", "epochs": 5}
            ]

            export_comparison_csv(first_results, output_path)
            export_comparison_csv(updated_results, output_path)

            with output_path.open(
                "r", newline="", encoding="utf-8"
            ) as file:
                exported_rows = list(csv.DictReader(file))

            self.assertEqual(len(exported_rows), 1)
            self.assertEqual(
                exported_rows[0]["session_id"],
                "new-session"
            )
            self.assertEqual(
                int(exported_rows[0]["epochs"]),
                5
            )


if __name__ == "__main__":
    unittest.main()
