import time

from src.process_manager import ProcessRegistry

registry = ProcessRegistry()

print("\n--- LAUNCHING TRAINING ---")

registry.launch_training(
    job_name="monitor-test",
    epochs=2,
    steps=100,
    batch_size=512,
    checkpoint="outputs/checkpoints/monitor_test.pt"
)

print("\n--- CHECKING IMMEDIATELY ---")
print("Completed:", registry.check_completed())

print("\n--- WAITING FOR TRAINING TO FINISH ---")

while registry.is_running("monitor-test"):
    time.sleep(1)

print("\n--- CHECKING AFTER COMPLETION ---")
print("Completed:", registry.check_completed())
print("\n--- CHECKING AGAIN ---")
print("Completed:", registry.check_completed())