import subprocess
import sys
import time
from pathlib import Path

from src.process_manager import ProcessRegistry

registry = ProcessRegistry() #calling class processregistry from process manager

stop_signal = Path("outputs/stop_training.flag")
checkpoint = Path("outputs/checkpoints/managed_stop.pt")

# Remove an old stop signal, if one exists.
stop_signal.unlink(missing_ok=True)

command = [
    sys.executable,
    "-m", "src.train",
    "--epochs", "20",
    "--steps", "1000",
    "--batch-size", "512",
    "--checkpoint", str(checkpoint),
]

process = subprocess.Popen(command)
registry.register("flexible-training", process)

print("Training is running:", registry.is_running("flexible-training"))

# Allow training to begin.
time.sleep(5)

if registry.is_running("flexible-training"):
    exit_code = registry.request_stop(
        "flexible-training",
        stop_signal
    )

    print("Training stopped with exit code:", exit_code)
else:
    print("Training finished before the stop request.")

print("Training is running:", registry.is_running("flexible-training"))
print("Checkpoint exists:", checkpoint.exists())