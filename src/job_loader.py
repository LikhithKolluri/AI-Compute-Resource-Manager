import json
from pathlib import Path

from src.scheduler import Job


def load_jobs(config_path):
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    jobs = []

    for job_config in config["jobs"]:
        job = Job(**job_config) #'**' operator unpacks the JSON dictionary into keyword argumentsw
        jobs.append(job)

    return jobs