import argparse
import csv
import time
import json
import torch
import math
from pathlib import Path
from torch import nn

from src.gpu_monitor import GPUMonitor


class Classifier(nn.Module):
    def __init__(self, features: int = 256, classes: int = 10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(features, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, classes),
        )

    def forward(self, inputs):
        return self.layers(inputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/training.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-signal", type=str, default="outputs/stop_training.flag")
    parser.add_argument("--job-name", type=str, default="standalone-training", help="Unique name of the training job")
    parser.add_argument("--session-id", type=str, help="Optional monitoring session ID for unique metrics storage")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--fixed-samples", type=int, default=102400)
    parser.add_argument("--simulate-failure", action="store_true")
    args = parser.parse_args()
    
    if args.epochs < 1 or args.steps < 1 or args.batch_size < 1:
        parser.error("epochs, steps, and batch-size must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable. Check NVIDIA driver and CUDA-enabled PyTorch installation.")
    if args.gpu_index < 0 or args.gpu_index >= torch.cuda.device_count():
        parser.error("Invalid GPU index")
    if args.warmup_steps < 0:
        parser.error("warmup-steps cannot be negative")
    if args.fixed_samples < 1:
        parser.error("fixed-samples must be positive")
    if args.benchmark and args.resume:
        parser.error("benchmark mode cannot resume a checkpoint")

    device = torch.device(f"cuda:{args.gpu_index}")
    torch.manual_seed(42)
    torch.cuda.set_device(device)
    print(f"Using {torch.cuda.get_device_name(device)}")
    model = Classifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    def training_step(batch_size):
        features = torch.randn(batch_size, 256, device=device)
        labels = features[:, :10].argmax(dim=1).long()

        optimizer.zero_grad(set_to_none=True)

        logits = model(features)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        return loss.detach().item()
    
    if args.benchmark:

        metrics_dir = Path("outputs/metrics") / args.job_name
        if args.session_id:
            metrics_dir = metrics_dir / args.session_id
            
        metrics_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"Benchmarking batch size {args.batch_size} "
            f"with {args.fixed_samples:,} target samples"
        )

        # Warm up CUDA kernels and the training loop.
        for _ in range(args.warmup_steps):
            training_step(args.batch_size)

        torch.cuda.synchronize(device)

        # Measure memory from this process rather than device-wide NVML usage.
        torch.cuda.reset_peak_memory_stats(device)

        steps = max(
            1,
            math.ceil(args.fixed_samples / args.batch_size)
        )

        # Monitor GPU activity during the measured workload.
        monitor = GPUMonitor(
            str(metrics_dir / "gpu_metrics.csv"),
            gpu_index=args.gpu_index,
            append=False
        )

        monitor.start()

        start = time.perf_counter()
        total_loss = 0.0

        try:
            for _ in range(steps):
                total_loss += training_step(args.batch_size)

            torch.cuda.synchronize(device)

            duration = time.perf_counter() - start

        finally:
            monitor.stop()

        processed_samples = steps * args.batch_size
        throughput = processed_samples / duration

        peak_allocated_gb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        )

        result = {
            "job_name": args.job_name,
            "batch_size": args.batch_size,
            "target_samples": args.fixed_samples,
            "processed_samples": processed_samples,
            "steps": steps,
            "warmup_steps": args.warmup_steps,
            "duration_seconds": round(duration, 4),
            "samples_per_second": round(throughput, 2),
            "peak_allocated_gb": round(peak_allocated_gb, 4),
            "average_loss": round(total_loss / steps, 5),
            "workload": "synthetic-classifier",
            "model_version": "classifier-v1",
            "precision": "fp32",
            "input_features": 256,
            "num_classes": 10,
            "gpu_name": torch.cuda.get_device_name(device),
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }

        result_path = metrics_dir / "benchmark_result.json"

        with result_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)

        print(
            f"BENCHMARK RESULT: {throughput:,.0f} samples/sec | "
            f"Peak allocated VRAM: {peak_allocated_gb:.3f} GB"
        )

        print(f"Saved benchmark result: {result_path}")

        return
    
    start_epoch = 0
    checkpoint_path = Path(args.checkpoint)

    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True
        )

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        start_epoch = checkpoint["epoch"]

        print(f"Resuming training from epoch {start_epoch + 1}")
    metrics_dir = Path("outputs/metrics") / args.job_name
    
    if args.session_id:
        metrics_dir = metrics_dir / args.session_id
        
    metrics_dir.mkdir(parents=True, exist_ok=True)

    monitor = GPUMonitor(
        str(metrics_dir / "gpu_metrics.csv"),
        gpu_index=args.gpu_index,
        append=args.resume
    )
    rows = []

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    monitor.start()
    try:
        for epoch in range(start_epoch,args.epochs):
            if args.simulate_failure and epoch == start_epoch:
                print("[TEST] Simulating training failure", flush=True)
                raise RuntimeError("Simulated training failure")
            start = time.perf_counter()
            total_loss = 0.0
            for _ in range(args.steps):
                # Synthetic classification data; generated directly on the GPU.
                features = torch.randn(args.batch_size, 256, device=device)
                labels = (features[:, :10].argmax(dim=1)).long()
                optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.detach().item()
            torch.cuda.synchronize(device)
            duration = time.perf_counter() - start
            avg_loss = total_loss / args.steps
            rows.append({"epoch": epoch + 1, "avg_loss": round(avg_loss, 5),
                         "epoch_seconds": round(duration, 3),
                         "samples_per_second": round(args.steps * args.batch_size / duration, 1)})
            print(f"Epoch {epoch + 1}: loss={avg_loss:.4f} | {duration:.2f}s | "
                  f"{rows[-1]['samples_per_second']:.0f} samples/s")
            #Saving the completed epcoh number along with the model & optimizer states.
            checkpoint_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict()
                },
                checkpoint_path
            )

            print(f"Checkpoint saved: {checkpoint_path}")
            stop_signal = Path(args.stop_signal)
            
            if stop_signal.exists():
                print("Stop signal detected. Training exiting safely.")
                stop_signal.unlink()
                break
    finally:
        torch.cuda.synchronize(device)

        peak_allocated_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )

        peak_reserved_mb = (
            torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        )

        monitor.stop()

        print(
            f"PyTorch peak allocated VRAM: "
            f"{peak_allocated_mb:.2f} MiB"
        )
        print(
            f"PyTorch peak reserved VRAM: "
            f"{peak_reserved_mb:.2f} MiB"
        )

        memory_metrics = {
            "peak_allocated_mb": round(peak_allocated_mb, 2),
            "peak_reserved_mb": round(peak_reserved_mb, 2),
            "gpu_index": args.gpu_index,
            "batch_size": args.batch_size,
        }

        memory_metrics_path = metrics_dir / "memory_metrics.json"

        with memory_metrics_path.open("w", encoding="utf-8") as f:
            json.dump(memory_metrics, f, indent=2)

        print(f"Saved memory metrics: {memory_metrics_path}")
        
        if rows:
            training_metrics_path = metrics_dir / "training_metrics.csv"

            append_metrics = (
                args.resume
                and training_metrics_path.exists()
            )

            with open(
                training_metrics_path,
                "a" if append_metrics else "w",
                newline="",
                encoding="utf-8"
            ) as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))

                if not append_metrics:
                    writer.writeheader()

                writer.writerows(rows)
    print(f"Saved {metrics_dir / 'gpu_metrics.csv'} " f"and {metrics_dir / 'training_metrics.csv'}")


if __name__ == "__main__":
    main()
