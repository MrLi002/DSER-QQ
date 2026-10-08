"""Train or fine-tune DSER++ on the BS-ERGB training split."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import build_bsergb_train_dataset
from model.loss import MaskedSSIMLoss
from model.ourNet import MyNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or fine-tune DSER++ on BS-ERGB"
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="BS-ERGB training root containing one directory per scene",
    )
    parser.add_argument("--output-dir", default="train/bsergb", help="Checkpoint directory")
    parser.add_argument("--resume", default=None, help="Resume a full training checkpoint")
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Load network weights only before starting a new optimizer",
    )
    parser.add_argument(
        "--allow-partial-pretrained",
        action="store_true",
        help="Allow missing or unexpected keys when loading --pretrained",
    )
    parser.add_argument("--scenes", nargs="*", default=None, help="Optional scene allow-list")

    parser.add_argument("--bins", default=8, type=int)
    parser.add_argument("--skips", nargs="+", default=[1, 3], type=int)
    parser.add_argument("--crop-size", default=256, type=int)
    parser.add_argument("--beta", default=0.1, type=float)
    parser.add_argument("--mask-patch-size", default=32, type=int)
    parser.add_argument("--no-augmentation", action="store_true")

    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--min-lr", default=1e-6, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--grad-clip", default=1.0, type=float)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--log-interval", default=20, type=int)
    parser.add_argument("--save-every", default=1, type=int)
    parser.add_argument(
        "--max-steps-per-epoch",
        default=None,
        type=int,
        help="Optional limit useful for smoke tests",
    )

    parser.add_argument("--lambda-rec", default=1.0, type=float)
    parser.add_argument("--lambda-syn", default=1.0, type=float)
    parser.add_argument("--lambda-ref", default=1.0, type=float)
    parser.add_argument("--lambda-ps", default=1.0, type=float)
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision")
    parser.add_argument(
        "--data-parallel",
        action="store_true",
        help="Use every visible CUDA device through DataParallel",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Training device, for example cuda, cuda:0, or cpu",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state dictionary")
    for key in ("net", "model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    if not isinstance(checkpoint, dict):
        raise TypeError("Could not find a network state dictionary in checkpoint")
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }


def _torch_load(path: str) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_network_weights(
    model: torch.nn.Module,
    path: str,
    strict: bool,
) -> None:
    checkpoint = _torch_load(path)
    state_dict = _extract_state_dict(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        print(f"Missing keys: {list(incompatible.missing_keys)}")
        print(f"Unexpected keys: {list(incompatible.unexpected_keys)}")


def resume_training(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: torch.cuda.amp.GradScaler,
    path: str,
) -> Tuple[int, int]:
    checkpoint = _torch_load(path)
    if not isinstance(checkpoint, dict):
        raise TypeError("Resume checkpoint must be a dictionary")
    model.load_state_dict(_extract_state_dict(checkpoint), strict=True)
    if "optimizer" not in checkpoint or "scheduler" not in checkpoint:
        raise KeyError("Resume checkpoint must contain optimizer and scheduler states")
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("epoch", -1)) + 1, int(checkpoint.get("global_step", 0))


def bgr_to_gray(image: torch.Tensor) -> torch.Tensor:
    weights = image.new_tensor((0.114, 0.587, 0.299)).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def lpips_distance(metric: torch.nn.Module, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return metric(pred * 2.0 - 1.0, target * 2.0 - 1.0).mean()


def multiscale_l1(predictions: Iterable[torch.Tensor], gt: torch.Tensor) -> torch.Tensor:
    weights = (1.0, 0.5, 0.1)
    loss = gt.new_zeros(())
    for weight, prediction in zip(weights, predictions):
        target = F.interpolate(
            gt, size=prediction.shape[-2:], mode="bilinear", align_corners=False
        )
        loss = loss + weight * F.l1_loss(prediction, target)
    return loss


def compute_losses(
    outputs: Tuple[object, ...],
    gt: torch.Tensor,
    mask: torch.Tensor,
    perceptual: torch.nn.Module,
    structural_loss: MaskedSSIMLoss,
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    _, masked_reference, coarse, predictions, selected_mask, _ = outputs
    final_prediction = predictions[0]

    gray_gt = bgr_to_gray(gt) * mask
    gray_gt = gray_gt.repeat(1, 3, 1, 1)
    loss_rec = F.l1_loss(masked_reference, gray_gt) + lpips_distance(
        perceptual, masked_reference, gray_gt
    )
    loss_syn = F.l1_loss(coarse, gt) + lpips_distance(perceptual, coarse, gt)
    loss_ref = multiscale_l1(predictions, gt) + lpips_distance(
        perceptual, final_prediction, gt
    )
    loss_ps = structural_loss(final_prediction, gt, selected_mask)
    total = (
        args.lambda_rec * loss_rec
        + args.lambda_syn * loss_syn
        + args.lambda_ref * loss_ref
        + args.lambda_ps * loss_ps
    )
    return {
        "total": total,
        "rec": loss_rec,
        "syn": loss_syn,
        "ref": loss_ref,
        "ps": loss_ps,
    }


def save_checkpoint(
    output_dir: Path,
    filename: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    network = model.module if isinstance(model, torch.nn.DataParallel) else model
    checkpoint = {
        "net": network.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args),
    }
    destination = output_dir / filename
    temporary = output_dir / f".{filename}.tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, destination)
    return destination


def train(args: argparse.Namespace) -> None:
    if args.resume and args.pretrained:
        raise ValueError("--resume and --pretrained are mutually exclusive")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    if args.log_interval <= 0 or args.save_every <= 0:
        raise ValueError("log-interval and save-every must be positive")
    if args.max_steps_per_epoch is not None and args.max_steps_per_epoch <= 0:
        raise ValueError("max-steps-per-epoch must be positive")
    if args.crop_size <= 0 or args.mask_patch_size <= 0:
        raise ValueError("crop-size and mask-patch-size must be positive")
    if args.crop_size % 32 != 0:
        raise ValueError("crop-size must be divisible by 32")
    if args.crop_size % args.mask_patch_size != 0:
        raise ValueError("crop-size must be divisible by mask-patch-size")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires a CUDA device")

    try:
        import lpips
    except ImportError as error:
        raise ImportError(
            "Training requires lpips; install the project dependencies with "
            "`pip install -r requirements.txt`."
        ) from error

    seed_everything(args.seed)
    dataset = build_bsergb_train_dataset(
        data_root=args.data_root,
        bins=args.bins,
        skips=args.skips,
        crop_size=args.crop_size,
        scenes=args.scenes,
        augment=not args.no_augmentation,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    if len(dataloader) == 0:
        raise RuntimeError("The BS-ERGB DataLoader is empty")

    model = MyNet(args).to(device)
    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    steps_per_epoch = len(dataloader)
    if args.max_steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs * steps_per_epoch),
        eta_min=args.min_lr,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = 0
    global_step = 0
    if args.resume:
        start_epoch, global_step = resume_training(
            model, optimizer, scheduler, scaler, args.resume
        )
        print(f"Resumed {args.resume} at epoch {start_epoch}")
    elif args.pretrained:
        load_network_weights(
            model,
            args.pretrained,
            strict=not args.allow_partial_pretrained,
        )
        print(f"Loaded pretrained weights from {args.pretrained}")

    if args.data_parallel:
        if device.type != "cuda" or torch.cuda.device_count() < 2:
            raise RuntimeError("--data-parallel requires at least two visible CUDA devices")
        model = torch.nn.DataParallel(model)

    perceptual = lpips.LPIPS(net="alex").to(device).eval()
    for parameter in perceptual.parameters():
        parameter.requires_grad_(False)
    structural_loss = MaskedSSIMLoss(window_size=11).to(device)

    print(
        f"Training samples: {len(dataset)} | batches: {len(dataloader)} | "
        f"parameters: {sum(p.numel() for p in model.parameters()):,}"
    )
    output_dir = Path(args.output_dir)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = {"total": 0.0, "rec": 0.0, "syn": 0.0, "ref": 0.0, "ps": 0.0}
        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        processed_steps = 0

        for step, batch in enumerate(progress):
            if args.max_steps_per_epoch is not None and step >= args.max_steps_per_epoch:
                break
            imgs = batch["imgs"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            voxels = batch["voxels"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = model(imgs, voxels, mask, args.bins)
                losses = compute_losses(
                    outputs, gt, mask, perceptual, structural_loss, args
                )

            scaler.scale(losses["total"]).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            processed_steps += 1
            for name in running:
                running[name] += float(losses[name].detach())

            if global_step % args.log_interval == 0:
                progress.set_postfix(
                    loss=f"{losses['total'].item():.4f}",
                    rec=f"{losses['rec'].item():.3f}",
                    syn=f"{losses['syn'].item():.3f}",
                    ref=f"{losses['ref'].item():.3f}",
                    ps=f"{losses['ps'].item():.3f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

        if processed_steps == 0:
            raise RuntimeError("No training steps were processed")
        means = {name: value / processed_steps for name, value in running.items()}
        print(
            f"Epoch {epoch + 1}: total={means['total']:.6f}, "
            f"rec={means['rec']:.6f}, syn={means['syn']:.6f}, "
            f"ref={means['ref']:.6f}, ps={means['ps']:.6f}"
        )

        save_checkpoint(
            output_dir,
            "last.pth",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            global_step,
            args,
        )
        if (epoch + 1) % args.save_every == 0:
            path = save_checkpoint(
                output_dir,
                f"epoch_{epoch + 1:03d}.pth",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                global_step,
                args,
            )
            print(f"Saved checkpoint: {path}")


if __name__ == "__main__":
    train(parse_args())
