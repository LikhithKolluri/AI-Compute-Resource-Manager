def evaluate_performance_regression(
    baseline_throughput: float,
    optimized_throughput: float,
    maximum_regression: float = 0.10,
):
    if baseline_throughput <= 0:
        raise ValueError("Baseline throughput must be positive")

    if optimized_throughput < 0:
        raise ValueError("Optimized throughput cannot be negative")

    if not 0 <= maximum_regression < 1:
        raise ValueError("Regression threshold must be between 0 and 1")

    change_pct = (
        (optimized_throughput - baseline_throughput)
        / baseline_throughput
    ) * 100

    regression_detected = change_pct < -(maximum_regression * 100)

    return {
        "baseline_throughput": baseline_throughput,
        "optimized_throughput": optimized_throughput,
        "change_pct": round(change_pct, 2),
        "regression_detected": regression_detected,
        "reason": (
            "Performance regression detected"
            if regression_detected
            else "Performance within acceptable range"
        ),
    }
    

def check_distributed_regression(job, baseline, optimized_throughput):
    baseline_throughput = baseline["baseline_throughput"]

    if baseline_throughput <= 0 or optimized_throughput <= 0:
        return None

    improvement_pct = (
        (optimized_throughput / baseline_throughput) - 1
    ) * 100

    regression_threshold = -5  
    regression_detected = (improvement_pct < regression_threshold)

    return {
        "evaluation_available": True,
        "regression_detected": regression_detected,
        "baseline_throughput": baseline_throughput,
        "optimized_throughput": optimized_throughput,
        "improvement_pct": improvement_pct,
        "reason": (
            f"Distributed throughput changed by "
            f"{improvement_pct:+.2f}% "
            f"({baseline_throughput:.2f} -> "
            f"{optimized_throughput:.2f} samples/s)"
        ),
    }
