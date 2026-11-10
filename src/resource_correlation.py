
from datetime import datetime, timezone

from src.gpu_analytics import load_gpu_metrics
from src.scheduling_analytics import load_events


def correlate_resources(samples, 
                        events, start_time=None,
                        end_time=None, session_id=None,
                        initial_reserved_gb=None):
    reservation_events = []

    for event in events:
        if session_id is not None and event.get("session_id") != session_id:
            continue
        if "reserved_vram_gb" not in event:
            continue

        timestamp = event["timestamp"]

        if isinstance(timestamp, datetime):
            timestamp = timestamp.timestamp()

        reservation_events.append({
            "timestamp": float(timestamp),
            "reserved_gb": float(event["reserved_vram_gb"]),
        })

    reservation_events.sort(key=lambda item: item["timestamp"])
    samples = sorted(samples, key=lambda item: item["timestamp"])

    correlated = []
    event_index = 0
    current_reservation = initial_reserved_gb

    for sample in samples:
        if session_id is not None and sample.get("session_id") != session_id:
            continue
        sample_time = float(sample["timestamp"])
        
        if start_time is not None and sample_time < start_time:
            continue

        if end_time is not None and sample_time > end_time:
            continue

        while (
            event_index < len(reservation_events)
            and reservation_events[event_index]["timestamp"] <= sample_time
        ):
            current_reservation = reservation_events[event_index]["reserved_gb"]
            event_index += 1

        if current_reservation is None:
            continue

        actual_vram_gb = sample["vram_used_mb"] / 1024

        correlated.append({
            "timestamp": sample_time,
            "gpu_utilization_pct": sample["gpu_utilization_pct"],
            "actual_vram_gb": actual_vram_gb,
            "reserved_vram_gb": current_reservation,
            "reservation_minus_usage_gb": (
                current_reservation - actual_vram_gb
            ),
        })

    return correlated


if __name__ == "__main__":
    samples = load_gpu_metrics()
    events = load_events()

    results = correlate_resources(samples, events)

    print("\n--- GPU RESERVATION CORRELATION ---")
    print(f"GPU samples: {len(samples)}")
    print(f"Matched samples: {len(results)}")

    if results:
        average_reserved = sum(
            row["reserved_vram_gb"] for row in results
        ) / len(results)

        average_actual = sum(
            row["actual_vram_gb"] for row in results
        ) / len(results)

        print(f"Average reserved VRAM: {average_reserved:.3f} GB")
        print(f"Average physical VRAM used: {average_actual:.3f} GB")
        print(
            f"Reservation minus usage: "
            f"{average_reserved - average_actual:.3f} GB"
        )
