
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class EventLogger:
    def __init__(
        self,
        path="outputs/analytics/job_events.jsonl",
        session_id=None
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.session_id = session_id or str(uuid4())

    def log(self, job, event, reserved_gb, reason=None):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "job_id": job.job_id,
            "job_name": job.name,
            "event": event,
            "priority": job.priority,
            "requested_vram_gb": job.vram_gb,
            "reserved_vram_gb": round(float(reserved_gb), 3),

            # Workload configuration for historical matching
            "workload": job.workload,
            "model_version": job.model_version,
            "batch_size": job.batch_size,
            "precision": job.precision,

            "reason": reason,
        }

        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
