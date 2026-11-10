
import unittest

from src.gpu_analytics import analyze_gpu_metrics


class GPUAnalyticsTests(unittest.TestCase):

    def test_gpu_statistics(self):
        samples = [
            {
                "timestamp": 1000.0,
                "gpu_utilization_pct": 20.0,
                "vram_used_mb": 400.0,
                "vram_total_mb": 8192.0,
            },
            {
                "timestamp": 1000.5,
                "gpu_utilization_pct": 40.0,
                "vram_used_mb": 450.0,
                "vram_total_mb": 8192.0,
            },
            {
                "timestamp": 1001.0,
                "gpu_utilization_pct": 60.0,
                "vram_used_mb": 500.0,
                "vram_total_mb": 8192.0,
            },
            {
                "timestamp": 1001.5,
                "gpu_utilization_pct": 80.0,
                "vram_used_mb": 550.0,
                "vram_total_mb": 8192.0,
            },
        ]

        result = analyze_gpu_metrics(samples)

        self.assertEqual(result["sample_count"], 4)
        self.assertEqual(result["average_gpu_utilization_pct"], 50)
        self.assertEqual(result["peak_gpu_utilization_pct"], 80)
        self.assertEqual(result["average_vram_used_mb"], 475)
        self.assertEqual(result["peak_vram_used_mb"], 550)

        self.assertAlmostEqual(
            result["peak_vram_usage_pct"],
            550 / 8192 * 100
        )

    def test_empty_samples(self):
        self.assertIsNone(analyze_gpu_metrics([]))


    def test_workload_memory_increase(self):
        from src.gpu_analytics import analyze_workload_memory

        baseline = [
            {"vram_used_mb": 150.0},
            {"vram_used_mb": 170.0},
        ]

        active = [
            {"vram_used_mb": 400.0},
            {"vram_used_mb": 600.0},
            {"vram_used_mb": 500.0},
        ]

        result = analyze_workload_memory(baseline, active)

        assert result["baseline_vram_mb"] == 160.0
        assert result["active_peak_vram_mb"] == 600.0
        assert result["estimated_peak_increase_mb"] == 440.0


    def test_workload_memory_requires_both_windows(self):
        from src.gpu_analytics import analyze_workload_memory

        assert analyze_workload_memory([], []) is None
        assert analyze_workload_memory(
            [{"vram_used_mb": 100.0}], []
        ) is None



if __name__ == "__main__":
    unittest.main()
