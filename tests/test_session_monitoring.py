
import time

from src.gpu_monitor import GPUMonitor
from src.scheduler import GPUScheduler


def test_shared_monitoring_session(tmp_path):
    scheduler = GPUScheduler(
        capacity_gb=8.0,
        protected_gb=2.0,
    )

    session_id = scheduler.event_logger.session_id

    monitor = GPUMonitor(
        output_path=str(tmp_path / "gpu_metrics.csv"),
        gpu_index=0,
        interval=0.2,
        session_id=session_id,
    )

    monitor.start()

    try:
        time.sleep(1)
    finally:
        monitor.stop()

    assert len(monitor.samples) > 0

    assert all(
        sample["session_id"] == session_id
        for sample in monitor.samples
    )

    assert all(
        sample["gpu_index"] == 0
        for sample in monitor.samples
    )
