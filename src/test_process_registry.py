
import subprocess
import sys

from src.process_manager import ProcessRegistry

registry = ProcessRegistry()

process = subprocess.Popen([
    sys.executable,
    "-c",
    "import time; time.sleep(2)"
])

registry.register("test-job", process)

print("Is running:", registry.is_running("test-job"))

exit_code = registry.wait_for("test-job")

print("Exit code:", exit_code)
print("Is running:", registry.is_running("test-job"))
