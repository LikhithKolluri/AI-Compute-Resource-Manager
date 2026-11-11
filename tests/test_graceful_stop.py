import subprocess
import sys
import time
from pathlib import Path

stop_signal = Path("outputs/stop_training.flag")
stop_signal.parent.mkdir(exist_ok=True)

# Remove stale stop signals from previous tests.
stop_signal.unlink(missing_ok=True)

command = [
    sys.executable,
    "-m",
    "src.train",
    "--epochs", "20",
    "--steps", "1000",
    "--batch-size", "512",
    "--checkpoint",
    "outputs/checkpoints/graceful_stop.pt",
]

process = subprocess.Popen(command)

print(f"Training started with PID: {process.pid}", flush=True)

# Give training time to start.
time.sleep(5)

if process.poll() is None:
    stop_signal.touch()
    print("Stop signal sent.", flush=True)
else:
    print("Training already finished before stop signal.", flush=True)

return_code = process.wait()

print(f"Training exited with code: {return_code}")