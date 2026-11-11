"""Profile the same synthetic training step used in src.train."""
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function
from src.train import Classifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    if args.batch_size < 1 or args.warmup < 0 or args.steps < 1:
        parser.error("batch-size and steps must be positive; warmup must be nonnegative")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this Python environment")
    device = torch.device("cuda:0")
    torch.manual_seed(42)
    model = Classifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    def train_step():
        with record_function("create_synthetic_batch"):
            features = torch.randn(args.batch_size, 256, device=device)
            labels = features[:, :10].argmax(dim=1).long()
        with record_function("forward_backward_optimizer"):
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        # Do not call loss.item(): that forces a GPU-to-CPU synchronization.

    print(f"GPU: {torch.cuda.get_device_name(device)} | batch: {args.batch_size}")
    for _ in range(args.warmup):
        train_step()
    torch.cuda.synchronize()
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        for _ in range(args.steps):
            train_step()
        torch.cuda.synchronize()
    out = Path("outputs") / "profiles"
    out.mkdir(parents=True, exist_ok=True)
    trace = out / f"batch_{args.batch_size}_trace.json"
    prof.export_chrome_trace(str(trace))
    report = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20)
    summary = out / f"batch_{args.batch_size}_summary.txt"
    summary.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved {summary} and {trace}")
    print("Profiler overhead changes timing; compare operation patterns, not end-to-end throughput, with this tool.")


if __name__ == "__main__":
    main()
