import json
from src.process_gpu_monitor import get_gpu_process_memory


def main():
    try:
        processes = get_gpu_process_memory()

        print(json.dumps({
            "monitoring_available": True,
            "processes": processes,
        }))

    except Exception as exc:
        print(json.dumps({
            "monitoring_available": False,
            "reason": str(exc),
            "processes": [],
        }))


if __name__ == "__main__":
    main()