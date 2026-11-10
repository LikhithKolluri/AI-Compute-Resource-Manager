import json
import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from src.train import Classifier


def setup_distributed():
    dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint(path, model, optimizer, completed_epoch, args, world_size):
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = checkpoint_path.with_suffix(".tmp")

    checkpoint = {
        "epoch": completed_epoch,
        "model_state": model.module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "world_size": world_size,
        "batch_size": args.batch_size,
        "samples": args.samples,
        "seed": 42,
    }

    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, checkpoint_path)

    print(
        f"[CHECKPOINT] Saved completed epoch {completed_epoch}: "
        f"{checkpoint_path}",
        flush=True
    )


def load_checkpoint(path, model, optimizer, args, world_size):
    checkpoint_path = Path(path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    if (
        checkpoint["world_size"] != world_size
        or checkpoint["batch_size"] != args.batch_size
        or checkpoint["samples"] != args.samples
    ):
        raise ValueError(
            "Checkpoint configuration does not match current job"
        )

    model.module.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    start_epoch = checkpoint["epoch"]

    print(
        f"[CHECKPOINT] Resuming from epoch {start_epoch + 1}",
        flush=True
    )

    return start_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--fail-rank", type=int, default=-1)
    parser.add_argument("--fail-epoch", type=int, default=1)
    parser.add_argument("--checkpoint", default="outputs/distributed/checkpoint.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-signal", default=None)
    parser.add_argument("--metrics-path", default=None)

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.samples < 2:
        parser.error("epochs, batch-size must be positive and samples >= 2")

    rank, world_size, local_rank = setup_distributed()

    try:
        torch.manual_seed(42)

        features = torch.randn(args.samples, 256)
        labels = features[:, :10].argmax(dim=1).long()
        dataset = TensorDataset(features, labels)

        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=42
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler
        )

        model = Classifier()
        model = DDP(model)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.001
        )

        criterion = nn.CrossEntropyLoss()

        start_epoch = 0

        if args.resume:
            start_epoch = load_checkpoint(
                args.checkpoint,
                model,
                optimizer,
                args,
                world_size
            )
        
        def stop_requested():
            local_stop = int(
                args.stop_signal is not None
                and Path(args.stop_signal).exists()
            )

            stop_tensor = torch.tensor(
                [local_stop],
                dtype=torch.int32
            )

            dist.all_reduce(
                stop_tensor,
                op=dist.ReduceOp.MAX
            )

            return bool(stop_tensor.item())
        
        epoch_metrics = []
        for epoch in range(start_epoch, args.epochs):
            sampler.set_epoch(epoch)
            
            if rank == args.fail_rank and epoch + 1 == args.fail_epoch:
                raise RuntimeError(
                    f"Injected worker failure: rank={rank}, epoch={epoch + 1}"
                )

            epoch_start = time.perf_counter()
            total_loss = 0.0
            local_samples = 0

            for batch_idx, (batch_features, batch_labels) in enumerate(loader):
                local_samples += batch_features.size(0)
                optimizer.zero_grad(set_to_none=True)

                predictions = model(batch_features)
                loss = criterion(predictions, batch_labels)

                loss.backward()

                if epoch == 0:
                    gradient = torch.cat([
                        p.grad.detach().reshape(-1)
                        for p in model.module.parameters()
                        if p.grad is not None
                    ])

                    reference_gradient = gradient.clone()
                    dist.broadcast(reference_gradient, src=0)

                    gradient_difference = (
                        gradient - reference_gradient
                    ).abs().max().item()

                    if gradient_difference > 1e-6:
                        raise RuntimeError(
                            f"Gradient mismatch on rank {rank}: "
                            f"{gradient_difference}"
                        )

                if epoch == 0 and rank == 0 and batch_idx == 0:
                    print(
                        "[DDP] Gradient synchronization active",
                        flush=True
                    )

                optimizer.step()
                total_loss += loss.item()

            dist.barrier()

            if stop_requested():
                if rank == 0:
                    print(
                    "[SHUTDOWN] Stop requested. "
                    "Checkpoint saved; workers exiting.",
                    flush=True
                )

                return
            
            # Count samples processed across all workers.
            sample_tensor = torch.tensor(
                [local_samples],
                dtype=torch.int64,
            )

            dist.all_reduce(
                sample_tensor,
                op=dist.ReduceOp.SUM,
            )

            global_samples = sample_tensor.item()

            # Use the slowest worker's elapsed time.
            local_duration = time.perf_counter() - epoch_start

            duration_tensor = torch.tensor(
                [local_duration],
                dtype=torch.float64,
            )

            dist.all_reduce(
                duration_tensor,
                op=dist.ReduceOp.MAX,
            )

            duration = duration_tensor.item()
            throughput = global_samples / duration if duration > 0 else 0.0

            if rank == 0:
                print(
                    f"[DISTRIBUTED METRICS] Epoch {epoch + 1}: "
                    f"samples={global_samples}, "
                    f"throughput={throughput:.2f} samples/s",
                    flush=True,
                )
                
                epoch_metrics.append({
                    "epoch": epoch + 1,
                    "samples_processed": global_samples,
                    "duration_seconds": duration,
                    "throughput": throughput,
                })


            print(
                f"[Rank {rank}/{world_size}] "
                f"Epoch {epoch + 1}: "
                f"loss={total_loss / len(loader):.4f}, "
                f"time={duration:.2f}s",
                flush=True
            )

            # Only Rank 0 writes the shared checkpoint.
            if rank == 0:
                save_checkpoint(
                    args.checkpoint,
                    model,
                    optimizer,
                    epoch + 1,
                    args,
                    world_size
                )

            # Ensure the checkpoint is saved before proceeding.
            dist.barrier()

        # Full parameter synchronization validation.
        with torch.no_grad():
            parameters = torch.cat([
                p.detach().reshape(-1).float()
                for p in model.module.parameters()
            ])

            reference = parameters.clone()
            dist.broadcast(reference, src=0)

            max_difference = (
                (parameters - reference).abs().max().item()
            )

            differences = [None] * world_size
            dist.all_gather_object(differences, max_difference)

            synchronized = all(
                difference < 1e-6 for difference in differences
            )

            if rank == 0:
                print(
                    f"\nMaximum parameter differences: {differences}",
                    flush=True
                )
                print(
                    "FULL PARAMETER SYNCHRONIZATION:",
                    "PASSED" if synchronized else "FAILED",
                    flush=True
                )

            # All ranks must agree on success or failure.
            if not synchronized:
                raise RuntimeError(
                    "DDP model parameters are not synchronized"
                )
                
        if rank == 0:
            metrics_path = Path(
                args.metrics_path or
                "outputs/distributed/training_metrics.json"
            )
            metrics_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            total_samples = sum(
                item["samples_processed"]
                for item in epoch_metrics
            )

            total_duration = sum(
                item["duration_seconds"]
                for item in epoch_metrics
            )

            metrics = {
                "status": "COMPLETED",
                "workers": world_size,
                "batch_size_per_worker": args.batch_size,
                "dataset_samples": args.samples,
                "total_samples_processed": total_samples,
                "training_duration_seconds": total_duration,
                "samples_per_second": (
                    total_samples / total_duration
                    if total_duration > 0 else 0.0
                ),
                "epochs": epoch_metrics,
            }

            metrics_path.write_text(
                json.dumps(metrics, indent=2),
                encoding="utf-8",
            )

            print(
                f"[DISTRIBUTED METRICS] Saved to {metrics_path}",
                flush=True,
            )


    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
