import csv
import threading
import time
from pathlib import Path

import pynvml



class GPUMonitor:
    def __init__(
        self,
        output_path: str,
        gpu_index: int = 0,
        interval: float = 0.5,
        append: bool = False,
        session_id: str = None
    ):
        self.output_path = Path(output_path)
        self.gpu_index = gpu_index
        self.interval = interval
        self.append = append
        self.session_id = session_id 

        self.stop_event = threading.Event()
        self.thread = None
        self.samples = []
        self.handle = None

    def start(self):
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._collect,
            daemon=True
        )
        self.thread.start()
    
    def _collect(self):
        while not self.stop_event.is_set():
            initialized = False

            try:
                pynvml.nvmlInit()
                initialized = True

                handle = pynvml.nvmlDeviceGetHandleByIndex(
                    self.gpu_index
                )

                utilization = pynvml.nvmlDeviceGetUtilizationRates(
                    handle
                )
                memory = pynvml.nvmlDeviceGetMemoryInfo(
                    handle
                )

                self.samples.append({
                    "timestamp": round(time.time(), 3),
                    "gpu_utilization_pct": utilization.gpu,
                    "vram_used_mb": round(
                        memory.used / (1024 ** 2), 1
                    ),
                    "vram_total_mb": round(
                        memory.total / (1024 ** 2), 1
                    ),
                    "session_id": self.session_id or "",
                    "gpu_index": self.gpu_index,
                })

            except pynvml.NVMLError as error:
                print(
                    f"GPU monitoring warning: {error}",
                    flush=True
                )

            finally:
                if initialized:
                    pynvml.nvmlShutdown()

            self.stop_event.wait(self.interval)
    
    def stop(self):
        self.stop_event.set()

        if self.thread:
            self.thread.join()

        if self.samples:
            self.output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            append_to_existing = (
                self.append
                and self.output_path.exists()
                and self.output_path.stat().st_size > 0
            )

            mode = "a" if append_to_existing else "w"

            with self.output_path.open(
                mode,
                newline="",
                encoding="utf-8"
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=list(self.samples[0])
                )

                if not append_to_existing:
                    writer.writeheader()

                writer.writerows(self.samples)
