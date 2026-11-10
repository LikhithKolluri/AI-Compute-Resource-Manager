import subprocess
import sys
import time
import json
import psutil
import uuid
from src.optimization_monitor import OptimizationMonitor
from pathlib import Path
from src.regression_monitor import check_optimized_job_regression
from src.optimization_history import record_optimization_decision
from src.historical_baseline import find_distributed_baseline
from src.performance_regression import check_distributed_regression
from src.optimization_history import record_distributed_decision


def read_distributed_metrics(job):
    metrics_path = Path(
        f"outputs/distributed/{job.name}_metrics.json"
    )

    try:
        metrics = json.loads(
            metrics_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        print(
            f"[DISTRIBUTED METRICS] Unable to read metrics: {exc}",
            flush=True,
        )
        return None

    if (
        metrics.get("status") != "COMPLETED"
        or metrics.get("workers") != job.workers
        or metrics.get("batch_size_per_worker") != job.batch_size
        or metrics.get("dataset_samples") != job.samples
    ):
        print(
            f"[DISTRIBUTED METRICS] Configuration mismatch for {job.name}",
            flush=True,
        )
        return None

    
    epochs = metrics.get("epochs", [])

    steady_epochs = [
        epoch for epoch in epochs
        if epoch.get("epoch", 0) > 1
    ]

    if not steady_epochs:
        return None

    try:
        total_samples = sum(
            epoch["samples_processed"]
            for epoch in steady_epochs
        )
        total_duration = sum(
            epoch["duration_seconds"]
            for epoch in steady_epochs
        )
    except (KeyError, TypeError, ValueError):
        return None

    throughput = (
        total_samples / total_duration
        if total_duration > 0 else 0.0
    )

    if not isinstance(throughput, (int, float)) or throughput <= 0:
        return None

    return throughput


class ProcessRegistry:
    def __init__(self):
        self.optimization_monitor = OptimizationMonitor()
        self.processes = {}
        self.job_sessions = {}
        
    

    def inspect_process_tree(self, job_name):
        process = self.processes.get(job_name)

        if process is None:
            return []

        try:
            parent = psutil.Process(process.pid)

            processes = [parent] + parent.children(
                recursive=True
            )

            return [
                {
                    "pid": p.pid,
                    "name": p.name(),
                    "parent_pid": p.ppid(),
                }
                for p in processes
                if p.is_running()
            ]

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
        ):
            return []
    
    def register(self, job_name, process):
        self.processes[job_name] = process
        print(
            f"REGISTERED {job_name}: PID {process.pid}"
        )

    def is_running(self, job_name):
        process = self.processes.get(job_name)

        if process is None:
            return False

        return process.poll() is None
    
    def query_gpu_processes(self, timeout=3):
        try:
            result = subprocess.run(
                [sys.executable, "-m", "src.gpu_process_probe"],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
            )

            data = json.loads(result.stdout)

            if not data["monitoring_available"]:
                return {
                    "monitoring_available": False,
                    "reason": data.get("reason", "GPU monitoring failed"),
                    "processes": [],
                }

            return {
                "monitoring_available": True,
                "processes": data["processes"],
            }

        except subprocess.TimeoutExpired:
            return {
                "monitoring_available": False,
                "reason": f"NVML query timed out after {timeout} seconds",
                "processes": [],
            }

        except (subprocess.SubprocessError, ValueError, KeyError) as exc:
            return {
                "monitoring_available": False,
                "reason": str(exc),
                "processes": [],
            }
    
    def get_gpu_job_status(self, gpu_index=0):
        probe = self.query_gpu_processes(timeout=3)

        if not probe["monitoring_available"]:
            return {
                "monitoring_available": False,
                "reason": probe["reason"],
                "jobs": [],
            }

        gpu_processes = probe["processes"]

        gpu_process_by_pid = {
            p["pid"]: p for p in gpu_processes
        }

        jobs = []

        for job_name, process in self.processes.items():
            running = process.poll() is None

            # Include the registered process and its children.
            process_ids = set()

            if running:
                try:
                    parent = psutil.Process(process.pid)

                    process_ids.add(parent.pid)

                    for child in parent.children(recursive=True):
                        if child.is_running():
                            process_ids.add(child.pid)

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            # Matching the NVML processes against the entire process tree.
            matched_gpu_processes = [
                gpu_process_by_pid[pid]
                for pid in process_ids
                if pid in gpu_process_by_pid
            ]

            known_memory = [
                p["gpu_memory_mib"]
                for p in matched_gpu_processes
                if p["gpu_memory_mib"] is not None
            ]

            if not running:
                gpu_state = "EXITED"
            elif matched_gpu_processes:
                gpu_state = "GPU_ACTIVE"
            else:
                gpu_state = "GPU_UNCONFIRMED"
                
            
            jobs.append({
                "job_name": job_name,
                "pid": process.pid,
                "running": running,
                "gpu_detected": bool(matched_gpu_processes),
                "gpu_state": gpu_state,
                "gpu_process_pids": sorted(
                    p["pid"] for p in matched_gpu_processes
                ),
                "gpu_memory_mib": (
                    sum(known_memory)
                    if len(known_memory) == len(matched_gpu_processes)
                    and matched_gpu_processes
                    else None
                ),
            })

        return {
            "monitoring_available": True,
            "jobs": jobs,
        }
    
    def check_completed(self):
        completed = []

        for job_name, process in list(self.processes.items()):
            exit_code = process.poll() ##Unlike process.wait(), which blocks until a process finishes, poll() checks its current status immediately.

            if exit_code is not None:
                completed.append((job_name, exit_code))
                del self.processes[job_name]
                
        return completed
    
    def sync_completed(self, scheduler):
        completed = self.check_completed()

        for job_name, exit_code in completed:
            
            optimization_state = self.optimization_monitor.finish(
                job_name,
                exit_code,
            )
            
            if optimization_state is not None:
                print(f"[OPTIMIZATION] {job_name}: "
                f"{optimization_state.value}",
                flush=True
                )
            

            job = next(
                (j for j in scheduler.running if j.name == job_name),
                None
            )

            if job is None:
                print(
                    f"PROCESS EXITED: {job_name} "
                    f"(not currently running in scheduler)",
                    flush=True
                )
                continue
            
            if job.distributed and exit_code == 0:
                report_path = Path(
                    f"outputs/distributed/{job_name}_report.json"
                )

                try:
                    report = json.loads(
                        report_path.read_text(encoding="utf-8")
                    )
                    outcome = report["status"]
                except (OSError, ValueError, KeyError) as exc:
                    print(
                        f"[DISTRIBUTED] Invalid or missing report for "
                        f"{job_name}: {exc}",
                        flush=True
                    )
                    scheduler.fail(job_name)
                    continue

                if outcome == "STOPPED":
                    print(
                        f"[DISTRIBUTED] Job stopped: {job_name}",
                        flush=True
                    )
                    scheduler.pause_stopped(job_name)
                    continue

                if outcome != "COMPLETED":
                    print(
                        f"[DISTRIBUTED] Unexpected outcome: {outcome}",
                        flush=True
                    )
                    scheduler.fail(job_name)
                    continue
                
                
                if (
                    job.original_batch_size is not None
                    and job.batch_size != job.original_batch_size
                    and not job.rollback_attempted
                ):
                    
                    if (
                        len(report.get("attempts", [])) != 1
                        or report["attempts"][0].get("status") != "COMPLETED"
                    ):
                        print(
                            f"[DISTRIBUTED OPTIMIZATION] {job_name}: "
                            "Skipping performance evaluation because "
                            "training did not complete in one clean attempt",
                            flush=True,
                        )
                    else:
                        baseline = find_distributed_baseline(job)
                        optimized_throughput = read_distributed_metrics(job)
                        
                        if baseline is None or optimized_throughput is None:
                            print(
                                f"[DISTRIBUTED OPTIMIZATION] {job_name}: "
                                "Evaluation unavailable; no decision recorded.",
                                flush=True,
                            )
                        else:
                            regression = check_distributed_regression(
                                job,
                                baseline,
                                optimized_throughput,
                            )

                            print(
                                f"[DISTRIBUTED PERFORMANCE] {job_name}: "
                                f"{regression}",
                                flush=True,
                            )
                            
                            
                            if regression is not None:
                                
                                improvement_pct = regression["improvement_pct"]

                                if improvement_pct < -5.0:
                                    decision = "REJECTED"
                                elif improvement_pct > 5.0:
                                    decision = "ACCEPTED"
                                else:
                                    decision = "INCONCLUSIVE"


                                record_distributed_decision(
                                    current_batch_size=job.original_batch_size,
                                    recommended_batch_size=job.batch_size,
                                    workers=job.workers,
                                    samples=job.samples,
                                    decision=decision,
                                    reason=regression["reason"],
                                    measured_throughput=optimized_throughput,
                                )

                                print(
                                    f"[DISTRIBUTED OPTIMIZATION] "
                                    f"{job_name}: {decision}",
                                    flush=True,
                                )
                                
                                
                                if decision == "REJECTED":
                                    print(
                                        f"[DISTRIBUTED ROLLBACK] {job_name}: "
                                        "Restoring original batch size",
                                        flush=True,
                                    )

                                    recovered = scheduler.rollback_optimization(
                                        job_name
                                    )

                                    if not recovered:
                                        scheduler.fail(job_name)

                                    continue

            
            if (
                exit_code == 0
                and not job.distributed
                and job.original_batch_size is not None
                and not job.rollback_attempted
            ):
                session_id = self.job_sessions.get(job_name)

                if session_id is not None:
                    regression = check_optimized_job_regression(
                        job,
                        session_id=session_id,
                    )

                    print(
                        f"[PERFORMANCE CHECK] {job_name}: {regression}",
                        flush=True,
                    )
                    
                    if regression["evaluation_available"]:
                        decision = (
                        "REJECTED"
                        if regression["regression_detected"]
                        else "ACCEPTED"
                    )

                    record_optimization_decision(
                        job=job,
                        decision=decision,
                        reason=regression["reason"],
                        baseline_throughput=regression.get("baseline_throughput"),
                        optimized_throughput=regression.get("optimized_throughput"),
                    )

                    print(
                        f"[OPTIMIZATION HISTORY] {job_name}: {decision}",
                        flush=True,
                    )

                    if (
                        regression["evaluation_available"]
                        and regression["regression_detected"]
                    ):
                        print(
                            f"[PERFORMANCE ROLLBACK] {job_name}: "
                            f"Throughput regression detected",
                            flush=True,
                        )

                        recovered = scheduler.rollback_optimization(job_name)

                        if recovered:
                            print(
                                f"[PERFORMANCE ROLLBACK] {job_name}: "
                                f"Original configuration restored; job queued for retry",
                                flush=True,
                            )
                        else:
                            scheduler.fail(job_name)

                        continue


            if exit_code == 0:
                print(f"PROCESS FINISHED: {job_name}", flush=True)
                scheduler.complete(job_name)


            else:
                print(
                    f"PROCESS FAILED: {job_name}, exit code {exit_code}",
                    flush=True
                )

                if (
                    optimization_state is not None
                    and optimization_state.value == "ROLLBACK_REQUIRED"
                ):
                    print(
                        f"[OPTIMIZATION] Rollback required for {job_name}",
                        flush=True,
                    )
                
                
                recovered = False

                if (
                    job.original_batch_size is not None
                    and not job.rollback_attempted
                ):
                    recovered = scheduler.rollback_optimization(job_name)

                if not recovered:
                    scheduler.fail(job_name)
                    
        return completed

    def monitor_until_done( self, scheduler, interval=1, enable_gpu_monitoring=True,):
        while self.processes or scheduler.waiting:

            # Handle completed processes first.
            self.sync_completed(scheduler)

            # Collect GPU monitoring information.
            if enable_gpu_monitoring and self.processes:
                status = self.get_gpu_job_status()

                if not status["monitoring_available"]:
                    print(
                        f"[GPU MONITOR] Unavailable: {status['reason']}",
                        flush=True,
                    )
                else:
                    for job_status in status["jobs"]:
                        self.optimization_monitor.observe(
                            job_status["job_name"],
                            job_status["gpu_state"],
                        )
                       
            time.sleep(interval)

        print("All managed processes finished.", flush=True)
    
    def wait_for(self, job_name):
        process = self.processes.get(job_name)

        if process is None:
            raise ValueError(
                f"No process registered for {job_name}"
            )

        return process.wait()
    
    def request_stop(self, job_name, stop_signal, timeout=30):
        process = self.processes.get(job_name)

        if process is None:
            raise ValueError(
                f"No process registered for {job_name}"
            )

        if process.poll() is not None:
            return process.returncode

        from pathlib import Path
        import subprocess

        signal_path = Path(stop_signal)
        signal_path.parent.mkdir(parents=True, exist_ok=True)
        signal_path.touch()

        print(f"Stop requested for {job_name}", flush=True)

        try:
            exit_code = process.wait(timeout=timeout)

            if exit_code == 0:
                print(
                    f"STOPPED {job_name}: PID {process.pid}, exit code {exit_code}",
                    flush=True
                )

                self.processes.pop(job_name, None)

            return exit_code

        except subprocess.TimeoutExpired:
            print(
                f"TIMEOUT: {job_name} did not stop within {timeout} seconds",
                flush=True
            )
            return None
    def launch_training(
        self,
        job_name,
        epochs=20,
        steps=1000,
        batch_size=512,
        checkpoint="outputs/checkpoints/managed_training.pt",
        resume=False,
        stop_signal="outputs/stop_training.flag",
        session_id=None,
        job=None,
        simulate_failure=False,
    ):
        if self.is_running(job_name):
            raise RuntimeError(f"{job_name} is already running")
        
        if session_id is None:
            session_id = str(uuid.uuid4())

        command = [
            sys.executable,
            "-m", "src.train",
            "--epochs", str(epochs),
            "--steps", str(steps),
            "--batch-size", str(batch_size),
            "--checkpoint", str(checkpoint),
            "--stop-signal", str(stop_signal),
            "--job-name", job_name
        ]

        if resume:
            command.append("--resume")
        
        if simulate_failure:
            command.append("--simulate-failure")
        
        if session_id is not None:
            command.extend(["--session-id", str(session_id)])

        process = subprocess.Popen(command)
        self.register(job_name, process)
        self.job_sessions[job_name] = session_id
        
        if job is not None:
            self.optimization_monitor.start(job)

        return process
    
    def launch_distributed_training(
        self,
        job_name,
        workers=2,
        epochs=3,
        batch_size=64,
        samples=4096,
        max_retries=2,
        checkpoint=None,
        report=None,
        resume=False,
        fail_rank=-1,
        fail_epoch=2,
        persistent_failure=False,
        stop_signal=None,
        metrics_path=None,
    ):
        if self.is_running(job_name):
            raise RuntimeError(f"{job_name} is already running")

        if checkpoint is None:
            checkpoint = f"outputs/distributed/{job_name}.pt"

        if report is None:
            report = f"outputs/distributed/{job_name}_report.json"
        
        if metrics_path is None:
            metrics_path = f"outputs/distributed/{job_name}_metrics.json"
        # Remove metrics from previous runs of this job.
        metrics_file = Path(metrics_path)
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_file.unlink(missing_ok=True)

        if stop_signal is None:
            stop_signal = f"outputs/distributed/{job_name}.flag"
            
        Path(stop_signal).parent.mkdir(parents=True, exist_ok=True)
        Path(stop_signal).unlink(missing_ok=True)

        command = [
            sys.executable,
            "-m",
            "src.distributed_launcher",
            "--workers", str(workers),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--samples", str(samples),
            "--max-retries", str(max_retries),
            "--checkpoint", str(checkpoint),
            "--report", str(report),
            "--fail-rank", str(fail_rank),
            "--fail-epoch", str(fail_epoch),
            "--stop-signal", str(stop_signal),
            "--metrics-path", str(metrics_path),
        ]
        if persistent_failure:
            command.append("--persistent-failure")
        
        if resume:
            command.append("--resume")

        process = subprocess.Popen(command)

        self.register(job_name, process)

        print(
            f"[DISTRIBUTED] Launched {job_name}: "
            f"workers={workers}, PID={process.pid}",
            flush=True,
        )

        return process

def start_training():
    command = [
        sys.executable,
        "-m",
        "src.train",
        "--epochs", "10",
        "--steps", "1000",
        "--batch-size", "512",
        "--checkpoint",
        "outputs/checkpoints/managed_training.pt",
        
    ]

    process = subprocess.Popen(command)
    print(f"Started training process with PID: {process.pid}", flush=True)

    return process

if __name__ == "__main__":
    process = start_training()
    return_code = process.wait()
    print(f"Training process exited with code: {return_code}")