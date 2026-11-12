
from prometheus_client import Gauge

RUNNING_JOBS = Gauge(
    "ai_rm_scheduler_running_jobs",
    "Number of jobs currently running",
)

QUEUED_JOBS = Gauge(
    "ai_rm_scheduler_queued_jobs",
    "Number of jobs waiting for GPU capacity",
)

RESERVED_VRAM = Gauge(
    "ai_rm_scheduler_reserved_vram_gb",
    "Logical GPU memory reserved by running jobs in GiB",
)

AVAILABLE_VRAM = Gauge(
    "ai_rm_scheduler_available_vram_gb",
    "Logical GPU memory available in GiB",
)

PROTECTED_VRAM = Gauge(
    "ai_rm_scheduler_protected_vram_gb",
    "GPU memory capacity protected for critical jobs in GiB",
)


def update_scheduler_metrics(scheduler):
    RUNNING_JOBS.set(len(scheduler.running))
    QUEUED_JOBS.set(len(scheduler.waiting))
    RESERVED_VRAM.set(scheduler.reserved_gb)
    AVAILABLE_VRAM.set(scheduler.available_gb)
    PROTECTED_VRAM.set(scheduler.protected_gb)

