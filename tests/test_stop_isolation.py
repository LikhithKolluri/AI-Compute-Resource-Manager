from pathlib import Path

from src.process_manager import ProcessRegistry

registry = ProcessRegistry()

signal_a = Path("outputs/stop_signals/training-A.flag")
signal_b = Path("outputs/stop_signals/training-B.flag")

signal_a.parent.mkdir(parents=True, exist_ok=True)

signal_a.unlink(missing_ok=True)
signal_b.unlink(missing_ok=True)

print("\n--- STARTING TWO TRAINING PROCESSES ---")

registry.launch_training(
    job_name="training-A",
    epochs=10,
    steps=1000,
    batch_size=256,
    checkpoint="outputs/checkpoints/isolation-A.pt",
    stop_signal=str(signal_a)
)

registry.launch_training(
    job_name="training-B",
    epochs=10,
    steps=1000,
    batch_size=256,
    checkpoint="outputs/checkpoints/isolation-B.pt",
    stop_signal=str(signal_b)
)

print("\n--- STOPPING ONLY TRAINING-A ---")

exit_code = registry.request_stop(
    "training-A",
    signal_a,
    timeout=60
)

print("Training-A stop exit code:", exit_code)

print("Training-B still running:", registry.is_running("training-B"))

assert exit_code == 0
assert registry.is_running("training-B")

print("\nSUCCESS: Training-B was not interrupted!")

registry.wait_for("training-B")

print("Training-B finished normally.")