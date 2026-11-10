
import time
import pynvml


def main():
    successful = 0
    failed = 0

    for i in range(10):
        initialized = False

        try:
            pynvml.nvmlInit()
            initialized = True

            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            utilization = pynvml.nvmlDeviceGetUtilizationRates(
                handle
            )
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)

            successful += 1

            print(
                f"Sample {i + 1}: "
                f"GPU={utilization.gpu}% "
                f"VRAM={memory.used / 1024**2:.1f} MB"
            )

        except pynvml.NVMLError as error:
            failed += 1
            print(
                f"Sample {i + 1} FAILED: "
                f"{type(error).__name__}: {error}"
            )

        finally:
            if initialized:
                pynvml.nvmlShutdown()

        time.sleep(0.5)

    print(f"\nSuccessful: {successful}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
