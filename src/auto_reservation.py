
from src.workload_history import find_matching_recommendation
from src.reservation_policy import evaluate_reservation_policy


def evaluate_job_reservation(job, auto_apply=False):
    """
    Evaluate a job using actual historical GPU measurements.

    Does not modify the job or scheduler.
    """

    history = find_matching_recommendation(job)

    if history is None:
        return {
            "approved": False,
            "reason": "No matching workload history",
            "reservation_gb": job.vram_gb,
        }

    decision = evaluate_reservation_policy(
        requested_gb=job.vram_gb,
        recommended_gb=history["recommended_reservation_gb"],
        matching_sessions=history["auto_eligible_sessions"],
        auto_apply=auto_apply,
    )

    return {
        **decision,
        "matching_sessions": history["matching_sessions"],
        "auto_eligible_sessions": history["auto_eligible_sessions"],
        "recommended_gb": history["recommended_reservation_gb"],
    }
