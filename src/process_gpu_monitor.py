import pynvml


def get_gpu_process_memory(gpu_index=0):
    pynvml.nvmlInit()

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(
            handle
        )

        results = []

        for process in processes:
            used_bytes = process.usedGpuMemory

            if used_bytes is None or used_bytes == pynvml.NVML_VALUE_NOT_AVAILABLE:
                used_mib = None
            else:
                used_mib = round(used_bytes / (1024 ** 2), 2)

            results.append({
                "pid": process.pid,
                "gpu_memory_mib": used_mib,
            })

        return results

    finally:
        pynvml.nvmlShutdown()