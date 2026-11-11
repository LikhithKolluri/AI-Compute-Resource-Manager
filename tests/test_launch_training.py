from src.process_manager import ProcessRegistry

registry = ProcessRegistry()

# Launch training using our new method
process = registry.launch_training(
    job_name="test-training",
    epochs=2,
    steps=100,
    batch_size=512,
    checkpoint="outputs/checkpoints/test_launch.pt",
    resume=False
)

print("Process PID:", process.pid)
print("Training running:", registry.is_running("test-training"))

# Wait for the actual process to finish
exit_code = registry.wait_for("test-training")

print("Training exit code:", exit_code)
print("Training running after completion:",
      registry.is_running("test-training"))