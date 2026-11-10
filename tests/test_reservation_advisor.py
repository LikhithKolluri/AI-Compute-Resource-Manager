
import unittest

from src.reservation_advisor import recommend_vram_reservation


class ReservationAdvisorTests(unittest.TestCase):

    def test_minimum_reservation(self):
        result = recommend_vram_reservation(210.90)

        self.assertEqual(result["recommended_gb"], 0.5)
        self.assertTrue(result["advisory_only"])

    def test_rounding_and_margin(self):
        result = recommend_vram_reservation(
            estimated_peak_increase_mb=1024,
            safety_margin=0.25,
        )

        self.assertEqual(result["recommended_gb"], 1.25)

    def test_negative_memory_rejected(self):
        with self.assertRaises(ValueError):
            recommend_vram_reservation(-100)

    def test_invalid_configuration_rejected(self):
        with self.assertRaises(ValueError):
            recommend_vram_reservation(
                500,
                rounding_gb=0,
            )


if __name__ == "__main__":
    unittest.main()
