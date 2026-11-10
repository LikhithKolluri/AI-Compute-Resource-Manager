from dataclasses import dataclass
from enum import Enum
from time import monotonic


class OptimizationState(str, Enum):
    WAITING_FOR_GPU = "WAITING_FOR_GPU"
    GPU_CONFIRMED = "GPU_CONFIRMED"
    COMPLETED = "COMPLETED"
    ROLLBACK_REQUIRED = "ROLLBACK_REQUIRED"


@dataclass
class OptimizationRecord:
    job_name: str
    original_batch_size: int
    optimized_batch_size: int
    state: OptimizationState = OptimizationState.WAITING_FOR_GPU
    gpu_confirmed: bool = False
    started_at: float = 0.0


class OptimizationMonitor:
    def __init__(self):
        self.records = {}

    def start(self, job):
        if job.original_batch_size is None:
            return False

        self.records[job.name] = OptimizationRecord(
            job_name=job.name,
            original_batch_size=job.original_batch_size,
            optimized_batch_size=job.batch_size,
            started_at=monotonic(),
        )

        return True

    def observe(self, job_name, gpu_state):
        record = self.records.get(job_name)

        if record is None:
            return None

        if record.state in (
            OptimizationState.COMPLETED,
            OptimizationState.ROLLBACK_REQUIRED,
        ):
            return record.state

        if gpu_state == "GPU_ACTIVE":
            record.gpu_confirmed = True
            record.state = OptimizationState.GPU_CONFIRMED

        return record.state

    def finish(self, job_name, exit_code):
        record = self.records.get(job_name)

        if record is None:
            return None

        if exit_code == 0:
            record.state = OptimizationState.COMPLETED
        else:
            record.state = OptimizationState.ROLLBACK_REQUIRED

        return record.state