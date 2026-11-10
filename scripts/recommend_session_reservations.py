
import argparse
import csv
from pathlib import Path

from src.reservation_advisor import recommend_vram_reservation


def recommend_from_comparison(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("Comparison report contains no sessions")

    recommendations = []

    for row in rows:
        peak_memory_mb = float(
            row["estimated_peak_memory_mb"]
        )

        recommendation = recommend_vram_reservation(
            estimated_peak_increase_mb=peak_memory_mb
        )

        recommendations.append({
            "session_id": row["session_id"],
            "estimated_peak_memory_mb": peak_memory_mb,
            "recommended_reservation_gb": recommendation[
                "recommended_gb"
            ],
            "safety_margin": recommendation["safety_margin"],
            "advisory_only": recommendation["advisory_only"],
        })

    return recommendations


def get_session_recommendation(
    session_id,
    csv_path="outputs/analytics/batch_comparison.csv",
):
    recommendations = recommend_from_comparison(csv_path)

    for result in recommendations:
        if result["session_id"] == session_id:
            return result["recommended_reservation_gb"]

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/analytics/batch_comparison.csv",
    )
    args = parser.parse_args()

    recommendations = recommend_from_comparison(args.input)

    print("\n--- SESSION VRAM RESERVATION RECOMMENDATIONS ---")

    for result in recommendations:
        print(f"\nSession: {result['session_id']}")
        print(
            f"Estimated peak workload VRAM: "
            f"{result['estimated_peak_memory_mb']:.2f} MB"
        )
        print(
            f"Recommended reservation: "
            f"{result['recommended_reservation_gb']:.2f} GiB"
        )
        print(f"Safety margin: {result['safety_margin']:.0%}")
        print(f"Advisory only: {result['advisory_only']}")


if __name__ == "__main__":
    main()
