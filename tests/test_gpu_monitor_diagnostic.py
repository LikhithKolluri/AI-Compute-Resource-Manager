
import time

from src.gpu_monitor import GPUMonitor


def test_gpu_monitor_diagnostic(tmp_path):
    monitor = GPUMonitor(
        output_path=str(tmp_path / "diagnostic.csv"),
        gpu_index=0,
        interval=0.5,
        session_id="diagnostic-session",
    )

    start = time.perf_counter()
    monitor.start()

    try:
        time.sleep(5)
    finally:
        monitor.stop()

    elapsed = time.perf_counter() - start

    print(f"\nElapsed time: {elapsed:.2f}s")
    print(f"Successful samples: {len(monitor.samples)}")

    timestamps = [
        sample["timestamp"]
        for sample in monitor.samples
    ]

    for i in range(1, len(timestamps)):
        difference = timestamps[i] - timestamps[i - 1]
        print(f"Sample interval {i}: {difference:.2f}s")

    assert len(monitor.samples) > 0
