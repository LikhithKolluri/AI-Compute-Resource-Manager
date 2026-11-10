
from src.benchmark_advisor import recommend_batch_size


def optimize_job_batch_size(job):
    """
    Apply a compatible benchmark recommendation before
    the job starts, only when explicitly enabled.
    """

    if not job.optimize_batch_size:
        print(
            f"[OPTIMIZER] {job.name}: "
            "Automatic optimization disabled."
        )
        return False

    if job.priority == 1:
        print(
            f"[OPTIMIZER] {job.name}: "
            "Skipping critical inference workload."
        )
        return False

    recommendation = recommend_batch_size(
        current_batch_size=job.batch_size,
        workload=job.workload,
        model_version=job.model_version,
        precision=job.precision,
        max_vram_gb=job.vram_gb,
    )

    if recommendation is None:
        print(
            f"[OPTIMIZER] {job.name}: "
            "No compatible benchmark available."
        )
        return False

    recommended = recommendation["recommended_batch_size"]

    if recommended <= job.batch_size:
        print(
            f"[OPTIMIZER] {job.name}: "
            "Keeping current batch size."
        )
        return False

    original = job.batch_size

    job.original_batch_size = original
    job.rollback_attempted = False

    job.batch_size = recommended

    print(
        f"[OPTIMIZER] {job.name}: "
        f"Batch size optimized {original} -> {recommended}"
    )

    return True
