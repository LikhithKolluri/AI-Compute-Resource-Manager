import json
import os
import time
import threading    
from pathlib import Path
from src.observability import EventLogger 
from dataclasses import dataclass
from enum import Enum
from uuid import uuid4
from src.workload_history import find_matching_recommendation
from src.scheduler_metrics import update_scheduler_metrics

class Status(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"
    PAUSED = "PAUSED"
    FAILED = "FAILED"

@dataclass
class Job:
    name: str
    vram_gb: float
    priority: int = 2 #
    status: Status = Status.QUEUED
    preemptible: bool = False
    borrowed_gb: float = 0.0
    start_failures: int = 0
    epochs: int = 10
    steps: int = 1000
    batch_size: int = 512
    workload: str = "synthetic-classifier"
    model_version: str = "classifier-v1"
    precision: str = "fp32"
    optimize_batch_size: bool = False
    original_batch_size: int | None = None
    rollback_attempted: bool = False
    job_id: str = ""
    recommended_vram_gb: float | None = None
    # Distributed training configuration
    distributed: bool = False
    workers: int = 2
    samples: int = 4096
    max_retries: int = 2
    


class GPUScheduler:
    def __init__(
        self,
        capacity_gb: float = 8.0,
        protected_gb: float = 2.0,
        stop_callback=None,
        start_callback=None
        
    ):
        if capacity_gb <= 0:
            raise ValueError("GPU capacity must be positive")

        if not 0 <= protected_gb <= capacity_gb:
            raise ValueError("Invalid protected capacity")

        self.capacity_gb = capacity_gb
        self.protected_gb = protected_gb
        self.stop_callback = stop_callback
        self.start_callback = start_callback
        self.preemption_count = 0

        self.running: list[Job] = []
        self.waiting: list[Job] = []
        self.max_start_failures = 3
        self.event_logger = EventLogger()
        self._metrics_file_lock = threading.Lock()
        
        update_scheduler_metrics(self)
        
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = threading.Thread(
            target=self.heartbeat_loop,
            args=(self._heartbeat_stop,),
            daemon=True,
        )
        self._heartbeat_thread.start()
        
    @property
    def reserved_gb(self):
        return sum(job.vram_gb for job in self.running)

    @property
    def available_gb(self):
        return self.capacity_gb - self.reserved_gb
    
    
    def write_metrics_state(self):
        state_file = (
            Path(__file__).resolve().parent.parent
            / "outputs"
            / "scheduler_state.json"
        )

        state_file.parent.mkdir(parents=True, exist_ok=True)

        with self._metrics_file_lock:
            state = {
                "updated_at": time.time(),
                "running_jobs": len(self.running),
                "queued_jobs": len(self.waiting),
                "preemptions_total": self.preemption_count,
                "reserved_vram_gb": self.reserved_gb,
                "available_vram_gb": self.available_gb,
                "protected_vram_gb": self.protected_gb,
            }
            temp_file = state_file.with_suffix(".tmp")

            with temp_file.open("w", encoding="utf-8") as file:
                json.dump(state, file)

            os.replace(temp_file, state_file)

    
    def log_event(self, job, event, reason=None):
        self.event_logger.log(
        job=job,
        event=event,
        reserved_gb=self.reserved_gb,
        reason=reason,
    )
        update_scheduler_metrics(self)
        self.write_metrics_state()

    def can_admit(self, job: Job) -> bool:
        # Critical jobs can use any unreserved capacity.
        if job.priority == 1:
            return job.vram_gb <= self.available_gb

        # Ordinary jobs must leave protected capacity free.
        if job.vram_gb <= self.available_gb - self.protected_gb:
            return True

        # Flexible jobs may borrow protected capacity.
        if job.preemptible and job.vram_gb <= self.available_gb:
            return True

        return False


    def submit(self, job: Job):
        job.job_id = str(uuid4())
        if job.vram_gb <= 0 or job.vram_gb > self.capacity_gb:
            job.status = Status.REJECTED
            self.log_event(job, "REJECTED", reason="Invalid VRAM request")

            print(
                f"REJECTED {job.name}: "
                "invalid or excessive VRAM request"
            )
            return

        if job.priority not in (1, 2, 3):
            job.status = Status.REJECTED
            self.log_event(job, "REJECTED", reason="Invalid priority")

            print(f"REJECTED {job.name}: invalid priority")
            return

        self.waiting.append(job)
        self.log_event(job, "SUBMITTED")

        if job.priority == 1:
            self.reclaim_for_critical(job)

        self.schedule()

        
    def submit_jobs(self, jobs):
        for job in jobs:
            self.submit(job)
        
    def attach_reservation_advisory(self, job):
        recommendation = find_matching_recommendation(job)

        job.recommended_vram_gb = (
            recommendation["recommended_reservation_gb"]
            if recommendation is not None
            else None
        )

    def schedule(self):
        # Priority 1 is highest
        self.waiting.sort(key=lambda job: job.priority)

        remaining = []

        for job in self.waiting:
            if self.can_admit(job):

                # Calculate borrowing for non-critical jobs
                if job.priority != 1:
                    normal_capacity = (
                        self.capacity_gb - self.protected_gb
                    )

                    noncritical_reserved = sum(
                        running_job.vram_gb
                        for running_job in self.running
                        if running_job.priority != 1
                    )

                    job.borrowed_gb = max(
                        0.0,
                        noncritical_reserved
                        + job.vram_gb
                        - normal_capacity
                    )
                else:
                    job.borrowed_gb = 0.0

                if self.start_callback is not None:
                    try:
                        started = self.start_callback(job)
                    except Exception as exc:
                        print(f"START FAILED {job.name}: {exc}", flush=True)
                        started = False

                    if not started:
                        job.start_failures += 1

                        print(
                            f"STARTUP FAILURE {job.name}: "
                            f"{job.start_failures}/{self.max_start_failures}",
                            flush=True
                        )

                        if job.start_failures >= self.max_start_failures:
                            job.status = Status.FAILED

                            self.log_event(job, "FAILED", reason="Maximum startup failures reached")

                            print(
                                f"FAILED {job.name}: maximum startup failures reached",
                                flush=True
                            )
                        else:
                            remaining.append(job)

                        continue
                    
                job.status = Status.RUNNING
                self.running.append(job)
                
                self.attach_reservation_advisory(job)
                
                self.log_event(job, "ADMITTED")

                print(
                    f"ADMITTED {job.name}: "
                    f"{job.vram_gb} GB reserved"
                )
                
                if job.recommended_vram_gb is not None:
                    print(
                        f"[ADVISORY] {job.name}: "
                        f"requested={job.vram_gb:.2f} GiB, "
                        f"recommended={job.recommended_vram_gb:.2f} GiB, "
                        f"reserved={job.vram_gb:.2f} GiB"
                    )

            else:
                remaining.append(job)
                
                self.log_event(
                    job,
                    "QUEUED",
                    reason="Insufficient available GPU capacity"
                )

                print(
                    f"QUEUED {job.name}: "
                    f"needs {job.vram_gb} GB, "
                    f"available {self.available_gb:.1f} GB"
                )

        self.waiting = remaining
        update_scheduler_metrics(self)
        self.write_metrics_state()
        
    def preempt(self, job: Job):
        if not job.preemptible:
            raise ValueError(f"{job.name} cannot be preempted")

        if job not in self.running:
            raise ValueError(f"{job.name} is not running")

        if self.stop_callback is not None:
            stopped = self.stop_callback(job)

            if not stopped:
                print(
                    f"PREEMPTION FAILED {job.name}: "
                    "process has not exited"
                )
                return False
        self.running.remove(job)
        job.status = Status.PAUSED
        job.borrowed_gb = 0.0
        self.preemption_count += 1
        
        self.log_event(
            job,
            "PREEMPTED",
            reason="GPU capacity reclaimed for higher-priority workload"
        )
        # A paused job returns to the waiting queue.
        self.waiting.append(job)

        print(f"PREEMPTED {job.name}: released {job.vram_gb} GB")
        return True
        
    def reclaim_for_critical(self, critical_job: Job):
        if critical_job.priority != 1:
            return

        if critical_job.vram_gb <= self.available_gb:
            return

        # Only interrupt flexible jobs that borrowed capacity.
        candidates = [
            job for job in self.running
            if job.preemptible and job.borrowed_gb > 0
        ]
        print("Preemption candidates:",[(j.name, j.borrowed_gb) for j in candidates])

        for job in candidates:
            success = self.preempt(job)

            if not success:
                continue

            if critical_job.vram_gb <= self.available_gb:
                break

    def complete(self, name: str):
        for job in self.running:
            if job.name == name:
                self.running.remove(job)
                job.status = Status.COMPLETED
                
                self.log_event(job, "COMPLETED")

                print(
                    f"COMPLETED {name}: "
                    f"released {job.vram_gb} GB"
                )

                self.schedule()
                return

        raise ValueError(
            f"No running job named {name!r}"
        )

    
    def rollback_optimization(self, name: str) -> bool:
        job = next(
            (j for j in self.running if j.name == name),
            None
        )

        if job is None:
            raise ValueError(f"No running job named {name}")

        if (
            job.original_batch_size is None
            or job.rollback_attempted
        ):
            return False

        original = job.original_batch_size
        optimized = job.batch_size

        job.rollback_attempted = True
        job.batch_size = original
        job.optimize_batch_size = False
        job.start_failures = 0

        # The failed process has already exited.
        self.running.remove(job)
        job.borrowed_gb = 0.0
        job.status = Status.QUEUED

        self.log_event(job,"ROLLBACK",reason=f"Restoring batch size from {optimized} to {original}")

        print(
            f"[ROLLBACK] {name}: Restoring batch size "
            f"{optimized} -> {original}",
            flush=True
        )

        # The retry must obey the usual capacity and priority rules.
        self.waiting.append(job)

        print(
            f"[ROLLBACK] {name}: Retrying with original configuration",
            flush=True
        )

        self.schedule()
        return True

    def fail(self, name):
        job = next(
            (j for j in self.running if j.name == name),
            None
        )

        if job is None:
            raise ValueError(f"No running job named {name}")

        self.running.remove(job)
        job.status = Status.FAILED
        job.borrowed_gb = 0.0
        
        self.log_event(job, "FAILED", reason="Workload execution failed")

        print(
            f"FAILED {job.name}: released {job.vram_gb} GB",
            flush=True
        )

        self.schedule()

    def pause_stopped(self, name: str):
        job = next(
            (j for j in self.running if j.name == name),
            None
        )

        if job is None:
            raise ValueError(f"No running job named {name}")

        self.running.remove(job)
        job.status = Status.PAUSED
        job.borrowed_gb = 0.0
        
        self.log_event(job, "PAUSED", reason="Workload stopped")

        print(
            f"PAUSED {name}: released {job.vram_gb} GB",
            flush=True
        )

        self.schedule()

    def heartbeat_loop(self, stop_event, interval=5):
        while not stop_event.is_set():
            self.write_metrics_state()
            stop_event.wait(interval)
            
    def stop_heartbeat(self):
        self._heartbeat_stop.set()
        self._heartbeat_thread.join(timeout=10)
    
    def report(self):
        print(
            f"\nGPU reservations: "
            f"{self.reserved_gb:.1f}/{self.capacity_gb:.1f} GB"
        )

        print(
            f"Protected capacity: {self.protected_gb:.1f} GB"
        )

        print(
            "Running:",
            ", ".join(job.name for job in self.running)
            or "none"
        )

        print(
            "Waiting:",
            ", ".join(job.name for job in self.waiting)
            or "none"
        )

        print()


if __name__ == "__main__":
    scheduler = GPUScheduler(
        capacity_gb=8.0,
        protected_gb=2.0
    )

    scheduler.submit(
        Job("model-training", 5, priority=2)
    )

    scheduler.submit(
        Job("production-inference", 2, priority=1)
    )

    scheduler.submit(
        Job("research-experiment", 4, priority=3)
    )

    scheduler.report()

    scheduler.complete("model-training")

    scheduler.report()
    scheduler.stop_heartbeat()