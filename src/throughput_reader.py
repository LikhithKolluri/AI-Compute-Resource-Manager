import csv
from pathlib import Path


def read_training_throughput(job_name, session_id=None):
    path = Path("outputs/metrics") / job_name

    if session_id is not None:
        path = path / session_id

    path = path / "training_metrics.csv"

    if not path.exists():
        return None

    total_samples = 0.0
    total_seconds = 0.0

    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            duration = float(row["epoch_seconds"])
            throughput = float(row["samples_per_second"])

            if duration <= 0 or throughput < 0:
                continue

            total_samples += throughput * duration
            total_seconds += duration

    if total_seconds == 0:
        return None

    return total_samples / total_seconds