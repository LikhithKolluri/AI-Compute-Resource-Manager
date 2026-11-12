import json     
import time
import pynvml
from pathlib import Path
from prometheus_client import Gauge, start_http_server

SCHEDULER_STATE_FILE = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "scheduler_state.json"
)

gpu_utilization = Gauge(
    "ai_rm_gpu_utilization_percent",
    "Current GPU utilization percentage",
)

gpu_vram_used = Gauge(
    "ai_rm_gpu_vram_used_mb",
    "GPU memory currently used in MB",
)

gpu_vram_total = Gauge(
    "ai_rm_gpu_vram_total_mb",
    "Total GPU memory in MB",
)


scheduler_running_jobs = Gauge(
    "ai_rm_scheduler_running_jobs",
    "Number of jobs currently running",
)

scheduler_queued_jobs = Gauge(
    "ai_rm_scheduler_queued_jobs",
    "Number of jobs waiting for GPU capacity",
)

scheduler_preemptions_total = Gauge(
    "ai_rm_scheduler_preemptions_total",
    "Total number of successful scheduler preemptions",
)

scheduler_reserved_vram = Gauge(
    "ai_rm_scheduler_reserved_vram_gb",
    "Logical GPU memory reserved in GiB",
)

scheduler_available_vram = Gauge(
    "ai_rm_scheduler_available_vram_gb",
    "Logical GPU memory available in GiB",
)

scheduler_protected_vram = Gauge(
    "ai_rm_scheduler_protected_vram_gb",
    "Protected GPU capacity in GiB",
)


def update_scheduler_from_file():
    if not SCHEDULER_STATE_FILE.exists():
        return

    try:
        with SCHEDULER_STATE_FILE.open("r", encoding="utf-8") as file:
            state = json.load(file)
            
            snapshot_age = time.time() - state["updated_at"]

            if snapshot_age > 30:
                scheduler_running_jobs.set(float("nan"))
                scheduler_queued_jobs.set(float("nan"))
                scheduler_reserved_vram.set(float("nan"))
                scheduler_available_vram.set(float("nan"))
                scheduler_protected_vram.set(float("nan"))
                return

        scheduler_running_jobs.set(state["running_jobs"])
        scheduler_queued_jobs.set(state["queued_jobs"])
        scheduler_reserved_vram.set(state["reserved_vram_gb"])
        scheduler_available_vram.set(state["available_vram_gb"])
        scheduler_protected_vram.set(state["protected_vram_gb"])
        scheduler_preemptions_total.set(state.get("preemptions_total", 0))

    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Unable to read scheduler state: {exc}", flush=True)




def main():
    start_http_server(8000)
    print("GPU metrics exporter running on port 8000", flush=True)

    try:
        while True:
            nvml_initialized = False

            try:
                pynvml.nvmlInit()
                nvml_initialized = True

                gpu = pynvml.nvmlDeviceGetHandleByIndex(0)

                utilization = pynvml.nvmlDeviceGetUtilizationRates(gpu)
                memory = pynvml.nvmlDeviceGetMemoryInfo(gpu)

                gpu_utilization.set(utilization.gpu)
                gpu_vram_used.set(memory.used / (1024 ** 2))
                gpu_vram_total.set(memory.total / (1024 ** 2))

            except pynvml.NVMLError as exc:
                print(
                    f"NVML read failed: {type(exc).__name__}: {exc}",
                    flush=True
                )

                gpu_utilization.set(float("nan"))
                gpu_vram_used.set(float("nan"))
                gpu_vram_total.set(float("nan"))

            finally:
                if nvml_initialized:
                    try:
                        pynvml.nvmlShutdown()
                    except pynvml.NVMLError as exc:
                        print(f"NVML shutdown failed: {exc}", flush=True)

            update_scheduler_from_file()
            time.sleep(2)

    except KeyboardInterrupt:
        print("Stopping GPU metrics exporter")


if __name__ == "__main__":
    main()
