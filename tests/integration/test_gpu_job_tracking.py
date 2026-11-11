import time

from src.process_manager import ProcessRegistry


def main():
    registry = ProcessRegistry()

    process = registry.launch_training(
        job_name="gpu-tracking-test",
        epochs=3,
        steps=3000,
        batch_size=2048,
        checkpoint="outputs/checkpoints/gpu_tracking_test.pt",
        stop_signal="outputs/gpu_tracking_test.flag",
    )

    try:
        while process.poll() is None:
            status = registry.get_gpu_job_status()

            
            for job in status["jobs"]:
                print(
                    f"Job: {job['job_name']} | "
                    f"PID: {job['pid']} | "
                    f"State: {job['gpu_state']} | "
                    f"Running: {job['running']} | "
                    f"GPU detected: {job['gpu_detected']} | "
                    f"GPU memory: {job['gpu_memory_mib']}",
                    flush=True,
                )

            time.sleep(1)

    finally:
        process.wait()

    print("Training process finished.")


if __name__ == "__main__":
    main()
    