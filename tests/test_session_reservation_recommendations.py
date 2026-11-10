
import csv
import tempfile
import unittest
from pathlib import Path

from scripts.recommend_session_reservations import (
    recommend_from_comparison,
)


class SessionReservationRecommendationTests(unittest.TestCase):

    def test_recommendations_from_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "batch_comparison.csv"

            with csv_path.open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "session_id",
                        "estimated_peak_memory_mb",
                    ],
                )
                writer.writeheader()
                writer.writerows([
                    {
                        "session_id": "session-128",
                        "estimated_peak_memory_mb": 208.9,
                    },
                    {
                        "session_id": "session-2048",
                        "estimated_peak_memory_mb": 2048,
                    },
                ])

            results = recommend_from_comparison(csv_path)

            self.assertEqual(len(results), 2)
            self.assertEqual(
                results[0]["session_id"], "session-128"
            )
            self.assertEqual(
                results[0]["recommended_reservation_gb"], 0.5
            )
            self.assertEqual(
                results[1]["recommended_reservation_gb"], 2.5
            )
            self.assertTrue(results[0]["advisory_only"])

    def test_empty_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "empty.csv"
            csv_path.write_text(
                "session_id,estimated_peak_memory_mb\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                recommend_from_comparison(csv_path)

    def test_missing_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "missing.csv"

            with self.assertRaises(FileNotFoundError):
                recommend_from_comparison(csv_path)


if __name__ == "__main__":
    unittest.main()
