
from pathlib import Path
import csv

from src.scheduling_analytics import load_events, analyze_jobs


REPORT_PATH = Path("outputs/analytics/scheduling_summary.csv")


def generate_report():
    events = load_events()
    jobs = analyze_jobs(events)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "job_id",
        "job_name",
        "status",
        "queue_seconds",
        "execution_seconds",
        "turnaround_seconds",
        "preemptions",
    ]

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for job_id, job in jobs.items():
            turnaround = None

            if job["submitted_at"] and job["finished_at"]:
                turnaround = (
                    job["finished_at"] - job["submitted_at"]
                ).total_seconds()

            writer.writerow({
                "job_id": job_id,
                "job_name": job["job_name"],
                "status": job["status"],
                "queue_seconds": round(job["queue_seconds"], 3),
                "execution_seconds": round(job["execution_seconds"], 3),
                "turnaround_seconds": (
                    round(turnaround, 3)
                    if turnaround is not None
                    else ""
                ),
                "preemptions": job["preemptions"],
            })

    print(f"Scheduling report saved: {REPORT_PATH}")
    print(f"Jobs analyzed: {len(jobs)}")


if __name__ == "__main__":
    generate_report()
