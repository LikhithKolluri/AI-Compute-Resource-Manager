
import json
from datetime import datetime
from pathlib import Path

EVENT_PATH = Path("outputs/analytics/job_events.jsonl")


def load_events(path=EVENT_PATH):
    if not Path(path).exists():
        return []

    events = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                event = json.loads(line)
                event["timestamp"] = datetime.fromisoformat(
                    event["timestamp"]
                )
                events.append(event)

    return events



def analyze_jobs(events):
    jobs = {}

    for event in events:
        job_id = event.get("job_id")

        # Ignore historical events without a job ID.
        if not job_id:
            continue

        event_type = event["event"]
        timestamp = event["timestamp"]

        if job_id not in jobs:
            jobs[job_id] = {
                "job_name": event["job_name"],
                "submitted_at": None,
                "waiting_since": None,
                "started_at": None,
                "finished_at": None,
                "queue_seconds": 0.0,
                "execution_seconds": 0.0,
                "preemptions": 0,
                "status": "UNKNOWN",
            }

        job = jobs[job_id]

        if event_type == "SUBMITTED":
            job["submitted_at"] = timestamp
            job["waiting_since"] = timestamp
            job["status"] = "QUEUED"

        elif event_type == "ADMITTED":
            if job["waiting_since"] is not None:
                job["queue_seconds"] += (
                    timestamp - job["waiting_since"]
                ).total_seconds()
                job["waiting_since"] = None

            job["started_at"] = timestamp
            job["status"] = "RUNNING"

        elif event_type == "PREEMPTED":
            if job["started_at"] is not None:
                job["execution_seconds"] += (
                    timestamp - job["started_at"]
                ).total_seconds()
                job["started_at"] = None

            job["waiting_since"] = timestamp
            job["preemptions"] += 1
            job["status"] = "QUEUED"

        elif event_type == "PAUSED":
            if job["started_at"] is not None:
                job["execution_seconds"] += (
                    timestamp - job["started_at"]
                ).total_seconds()
                job["started_at"] = None

            job["status"] = "PAUSED"

        elif event_type == "ROLLBACK":
            if job["started_at"] is not None:
                job["execution_seconds"] += (
                    timestamp - job["started_at"]
                ).total_seconds()
                job["started_at"] = None

            job["waiting_since"] = timestamp
            job["status"] = "QUEUED"

        elif event_type in ("COMPLETED", "FAILED", "REJECTED"):
            if job["started_at"] is not None:
                job["execution_seconds"] += (
                    timestamp - job["started_at"]
                ).total_seconds()
                job["started_at"] = None

            job["finished_at"] = timestamp
            job["status"] = event_type

    return jobs


def calculate_live_metrics(job, as_of=None):
    from datetime import datetime, timezone

    if as_of is None:
        as_of = datetime.now(timezone.utc)

    queue_seconds = job["queue_seconds"]
    execution_seconds = job["execution_seconds"]

    if job["status"] == "QUEUED" and job["waiting_since"] is not None:
        queue_seconds += max(
            0.0,
            (as_of - job["waiting_since"]).total_seconds()
        )

    if job["status"] == "RUNNING" and job["started_at"] is not None:
        execution_seconds += max(
            0.0,
            (as_of - job["started_at"]).total_seconds()
        )

    return {
        "queue_seconds": queue_seconds,
        "execution_seconds": execution_seconds,
    }


if __name__ == "__main__":
    events = load_events()
    results = analyze_jobs(events)

    print("\n--- SCHEDULING ANALYTICS ---")

    for job_id, metrics in results.items():
        turnaround = None

        if (
            metrics["submitted_at"] is not None
            and metrics["finished_at"] is not None
        ):
            turnaround = (
                metrics["finished_at"] - metrics["submitted_at"]
            ).total_seconds()

        print(f"\nJob: {metrics['job_name']}")
        print(f"Job ID: {job_id}")
        print(f"Status: {metrics['status']}")
        print(f"Queue time: {metrics['queue_seconds']:.3f}s")
        print(f"Execution time: {metrics['execution_seconds']:.3f}s")
        print(f"Preemptions: {metrics['preemptions']}")
        print(
            f"Turnaround time: {turnaround:.3f}s"
            if turnaround is not None
            else "Turnaround time: N/A"
        )

