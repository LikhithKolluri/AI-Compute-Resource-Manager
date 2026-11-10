from src.throughput_reader import read_training_throughput
from src.performance_regression import evaluate_performance_regression
from src.historical_baseline import find_historical_baseline

def check_optimized_job_regression(job, minimum_sessions=3,session_id=None):
    history = find_historical_baseline(job)

    if history is None:
        return {
            "evaluation_available": False,
            "regression_detected": False,
            "reason": "No matching historical baseline",
        }

    if history["matching_sessions"] < minimum_sessions:
        return {
            "evaluation_available": False,
            "regression_detected": False,
            "reason": "Insufficient historical baseline sessions",
            "matching_sessions": history["matching_sessions"],
        }

    return check_job_regression(
        job_name=job.name,
        baseline_throughput=history["baseline_throughput"],
        session_id=session_id,
    )
    
def check_job_regression(job_name, baseline_throughput, session_id=None):
    actual_throughput = read_training_throughput(job_name, session_id=session_id)

    if actual_throughput is None:
        return {
            "regression_detected": False,
            "reason": "Training metrics unavailable",
            "evaluation_available": False,
        }

    result = evaluate_performance_regression(
        baseline_throughput,
        actual_throughput,
    )

    return {
        **result,
        "evaluation_available": True,
        "job_name": job_name,
    }

def check_job_regression_from_history(job,session_id=None):
    baseline = find_historical_baseline(job)

    if baseline is None:
        return {
            "job_name": job.name,
            "evaluation_available": False,
            "regression_detected": False,
            "reason": "No sufficient matching historical baseline",
        }

    result = check_job_regression(
        job.name,
        baseline["baseline_throughput"],
        session_id=session_id
    )

    return {
        **result,
        "matching_sessions": baseline["matching_sessions"],
        "original_batch_size": baseline["original_batch_size"],
    }