"""
train.py
========
Script training utama untuk HSCN Waste Classification.

Cara menjalankan:
    python train.py                          # konfigurasi default
    python train.py --backbone resnet101     # ganti backbone
    python train.py --epochs 100 --lr 1e-3   # ubah hyperparameter
    python train.py --resume checkpoints/best.pth  # lanjut dari checkpoint

Struktur dataset yang diharapkan:
    dataset_hscn/
    ├── train/
    │   ├── image/     (*.jpg)
    │   └── labels.json
    ├── valid/
    │   ├── image/
    │   └── labels.json
    └── test/
        ├── image/
        └── labels.json
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np

# Import modul lokal
from hierarchy import num_l1, num_l2, num_l3, L1_CLASSES, L2_ALL, L3_ALL
from dataset  import WasteHSCNDataset, build_dataloaders
from model    import HSCN
from loss     import HSCNLoss
from metrics  import HSCNMetrics


# ─── Setup Logging ────────────────────────────────────────────────────────────

def setup_logging(log_dir: str, run_name: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{run_name}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


# ─── Argument Parser ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train HSCN for Hierarchical Waste Classification"
    )

    # Dataset
    parser.add_argument("--data_dir",    type=str, default="dataset_hscn",
                        help="Root direktori dataset")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Jumlah worker DataLoader")

    # Model
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=list(HSCN.SUPPORTED_BACKBONES.keys()),
                        help="Backbone feature extractor")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Gunakan bobot ImageNet pretrained")
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Ukuran hidden layer classifier")
    parser.add_argument("--dropout",    type=float, default=0.5,
                        help="Dropout rate")

    # Training
    parser.add_argument("--epochs",      type=int,   default=80)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-3,
                        help="Learning rate untuk classifier heads")
    parser.add_argument("--backbone_lr", type=float, default=1e-4,
                        help="Learning rate untuk backbone (fine-tuning)")
    parser.add_argument("--weight_decay",type=float, default=1e-4)
    parser.add_argument("--scheduler",   type=str,   default="cosine",
                        choices=["cosine", "onecycle", "none"])
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Epoch warmup untuk OneCycleLR")
    parser.add_argument("--use_amp",     action="store_true", default=True,
                        help="Gunakan Automatic Mixed Precision (FP16)")
    parser.add_argument("--no_amp",      dest="use_amp", action="store_false")

    # Loss weights
    parser.add_argument("--lambda_l1",  type=float, default=1.0)
    parser.add_argument("--lambda_l2",  type=float, default=1.0)
    parser.add_argument("--lambda_l3",  type=float, default=0.5)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--use_class_weights", action="store_true", default=True,
                        help="Gunakan class-weighted loss untuk menangani imbalance")

    # Checkpoint & logging
    parser.add_argument("--ckpt_dir",   type=str, default="checkpoints")
    parser.add_argument("--log_dir",    type=str, default="logs")
    parser.add_argument("--run_name",   type=str, default="hscn_waste")
    parser.add_argument("--resume",     type=str, default=None,
                        help="Path ke checkpoint untuk dilanjutkan")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Simpan checkpoint setiap N epoch")

    # Misc
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--device",     type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--freeze_backbone_epochs", type=int, default=0,
                        help="Bekukan backbone N epoch pertama (transfer learning)")

    return parser.parse_args()


# ─── Seed ─────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ─── Train One Epoch ──────────────────────────────────────────────────────────

def train_epoch(
    model      : HSCN,
    loader     : torch.utils.data.DataLoader,
    optimizer  : torch.optim.Optimizer,
    loss_fn    : HSCNLoss,
    scaler     : Optional[GradScaler],
    device     : torch.device,
    scheduler  = None,
    use_amp    : bool = True,
) -> Dict[str, float]:
    """Satu epoch training. Kembalikan dict loss rata-rata."""
    model.train()

    total_loss = 0.0
    loss_accum = {}
    num_batches = 0

    for batch_idx, (imgs, l1, l2, l3) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        l1   = l1.to(device, non_blocking=True)
        l2   = l2.to(device, non_blocking=True)
        l3   = l3.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            out              = model(imgs)
            loss, loss_dict  = loss_fn(out, l1, l2, l3)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        # Scheduler step per-batch (OneCycleLR)
        if scheduler is not None and isinstance(scheduler, OneCycleLR):
            scheduler.step()

        # Akumulasi loss
        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_accum[k] = loss_accum.get(k, 0.0) + v.item()
        num_batches += 1

    # Rata-rata
    return {k: v / num_batches for k, v in loss_accum.items()}


# ─── Validation / Test ────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model   : HSCN,
    loader  : torch.utils.data.DataLoader,
    loss_fn : HSCNLoss,
    device  : torch.device,
    use_amp : bool = True,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Evaluasi model pada satu loader.

    Returns:
        loss_dict  : rata-rata loss per komponen
        metric_dict: semua metrik akurasi
    """
    model.eval()

    metrics    = HSCNMetrics()
    loss_accum = {}
    num_batches = 0

    for imgs, l1, l2, l3 in loader:
        imgs = imgs.to(device, non_blocking=True)
        l1   = l1.to(device, non_blocking=True)
        l2   = l2.to(device, non_blocking=True)
        l3   = l3.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            out             = model(imgs)
            _, loss_dict    = loss_fn(out, l1, l2, l3)

        preds = model.predict(imgs)
        metrics.update(preds, l1, l2, l3)

        for k, v in loss_dict.items():
            loss_accum[k] = loss_accum.get(k, 0.0) + v.item()
        num_batches += 1

    avg_losses  = {k: v / num_batches for k, v in loss_accum.items()}
    metric_dict = metrics.compute()
    return avg_losses, metric_dict


# ─── Build Optimizer ──────────────────────────────────────────────────────────

def build_optimizer(model: HSCN, args) -> optim.Optimizer:
    """
    Buat optimizer dengan dua parameter group:
        - backbone: lr kecil (fine-tuning)
        - heads   : lr penuh
    """
    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.clf_l1.parameters()) +
        [p for clf in model.clf_l2.values() for p in clf.parameters()] +
        [p for clf in model.clf_l3.values() for p in clf.parameters()]
    )

    param_groups = [
        {"params": backbone_params, "lr": args.backbone_lr, "name": "backbone"},
        {"params": head_params,     "lr": args.lr,          "name": "heads"},
    ]

    return optim.AdamW(param_groups, weight_decay=args.weight_decay)


# ─── Main Training Loop ───────────────────────────────────────────────────────

def main():
    args   = parse_args()
    logger = setup_logging(args.log_dir, args.run_name)
    set_seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    logger.info(f"Loading dataset dari: {args.data_dir}")
    train_loader, valid_loader, test_loader, train_ds = build_dataloaders(
        dataset_root = args.data_dir,
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
        pin_memory   = (device.type == "cuda"),
    )
    train_ds.print_stats()
    logger.info(
        f"Dataset sizes → train: {len(train_loader.dataset)}, "
        f"valid: {len(valid_loader.dataset)}, "
        f"test:  {len(test_loader.dataset)}"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info(f"Membangun HSCN dengan backbone: {args.backbone}")
    model = HSCN(
        backbone_name = args.backbone,
        pretrained    = args.pretrained,
        hidden_dim    = args.hidden_dim,
        dropout       = args.dropout,
    ).to(device)

    param_info = model.count_parameters()
    logger.info(f"Parameter total    : {param_info['total']:,}")
    logger.info(f"Parameter trainable: {param_info['trainable']:,}")

    # Bekukan backbone jika diminta
    if args.freeze_backbone_epochs > 0:
        for p in model.backbone.parameters():
            p.requires_grad = False
        logger.info(f"Backbone dibekukan untuk {args.freeze_backbone_epochs} epoch pertama")

    # ── Loss Function ─────────────────────────────────────────────────────────
    cw_l1, cw_l2, cw_l3 = None, None, None
    if args.use_class_weights:
        cw_l1 = train_ds.class_weights_l1
        cw_l2 = train_ds.class_weights_l2
        cw_l3 = train_ds.class_weights_l3
        logger.info("Menggunakan class-weighted loss")

    loss_fn = HSCNLoss(
        class_weights_l1 = cw_l1,
        class_weights_l2 = cw_l2,
        class_weights_l3 = cw_l3,
        lambda_l1        = args.lambda_l1,
        lambda_l2        = args.lambda_l2,
        lambda_l3        = args.lambda_l3,
        label_smoothing  = args.label_smoothing,
    ).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = build_optimizer(model, args)

    # ── Scheduler ─────────────────────────────────────────────────────────────
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
    elif args.scheduler == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr       = [args.backbone_lr * 10, args.lr * 10],
            steps_per_epoch = len(train_loader),
            epochs       = args.epochs,
            pct_start    = args.warmup_epochs / args.epochs,
        )

    # ── AMP Scaler ────────────────────────────────────────────────────────────
    scaler = GradScaler() if (args.use_amp and device.type == "cuda") else None

    # ── Resume dari checkpoint ────────────────────────────────────────────────
    start_epoch  = 0
    best_acc_l1  = 0.0
    history      = []

    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Melanjutkan dari checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc_l1 = ckpt.get("best_acc_l1", 0.0)
        history     = ckpt.get("history", [])
        if scheduler and "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        logger.info(f"Dilanjutkan dari epoch {start_epoch}")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TRAINING LOOP
    # ─────────────────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Mulai training: {args.epochs} epoch, batch_size={args.batch_size}")
    logger.info(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # ── Unfreeze backbone setelah warmup ──────────────────────────────────
        if epoch == args.freeze_backbone_epochs and args.freeze_backbone_epochs > 0:
            for p in model.backbone.parameters():
                p.requires_grad = True
            logger.info(f"[Epoch {epoch+1}] Backbone di-unfreeze")

        # ── Train ─────────────────────────────────────────────────────────────
        train_losses = train_epoch(
            model, train_loader, optimizer, loss_fn,
            scaler, device, scheduler if isinstance(scheduler, OneCycleLR) else None,
            use_amp=args.use_amp,
        )

        # ── Scheduler step per-epoch ──────────────────────────────────────────
        if scheduler is not None and not isinstance(scheduler, OneCycleLR):
            scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────────
        valid_losses, valid_metrics = evaluate(
            model, valid_loader, loss_fn, device, use_amp=args.use_amp
        )

        elapsed = time.time() - t0
        current_lr_head = optimizer.param_groups[1]["lr"]

        # ── Logging ───────────────────────────────────────────────────────────
        logger.info(
            f"Epoch [{epoch+1:03d}/{args.epochs}] "
            f"| Time: {elapsed:.1f}s "
            f"| LR: {current_lr_head:.2e} "
            f"| Train Loss: {train_losses.get('total', 0):.4f} "
            f"| Val Loss: {valid_losses.get('total', 0):.4f} "
            f"| Val Acc L1: {valid_metrics.get('acc_l1', 0):.4f} "
            f"| Val Acc L2: {valid_metrics.get('acc_l2', 0):.4f} "
            f"| Val Acc L3: {valid_metrics.get('acc_l3', 0):.4f} "
            f"| Val Acc Mean: {valid_metrics.get('acc_mean', 0):.4f}"
        )

        # ── Simpan history ─────────────────────────────────────────────────────
        epoch_record = {
            "epoch"         : epoch + 1,
            "train_loss"    : train_losses.get("total", 0),
            "val_loss"      : valid_losses.get("total", 0),
            "val_acc_l1"    : valid_metrics.get("acc_l1", 0),
            "val_acc_l2"    : valid_metrics.get("acc_l2", 0),
            "val_acc_l3"    : valid_metrics.get("acc_l3", 0),
            "val_acc_mean"  : valid_metrics.get("acc_mean", 0),
            "lr"            : current_lr_head,
        }
        history.append(epoch_record)

        # ── Simpan checkpoint terbaik ──────────────────────────────────────────
        val_acc = valid_metrics.get("acc_l1", 0)
        is_best = val_acc > best_acc_l1
        if is_best:
            best_acc_l1 = val_acc
            best_ckpt_path = os.path.join(args.ckpt_dir, f"{args.run_name}_best.pth")
            torch.save({
                "epoch"          : epoch,
                "model_state"    : model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler else None,
                "best_acc_l1"    : best_acc_l1,
                "valid_metrics"  : valid_metrics,
                "args"           : vars(args),
                "history"        : history,
            }, best_ckpt_path)
            logger.info(f"  ✓ Best model disimpan (acc_l1={best_acc_l1:.4f})")

        # ── Simpan checkpoint periodik ─────────────────────────────────────────
        if (epoch + 1) % args.save_every == 0:
            periodic_path = os.path.join(
                args.ckpt_dir, f"{args.run_name}_epoch{epoch+1:03d}.pth"
            )
            torch.save({
                "epoch"          : epoch,
                "model_state"    : model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler else None,
                "best_acc_l1"    : best_acc_l1,
                "args"           : vars(args),
                "history"        : history,
            }, periodic_path)

    # ─────────────────────────────────────────────────────────────────────────
    # FINAL TEST EVALUATION
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("Evaluasi akhir pada TEST SET menggunakan model terbaik")
    logger.info("="*60)

    # Muat model terbaik
    best_ckpt = os.path.join(args.ckpt_dir, f"{args.run_name}_best.pth")
    if os.path.isfile(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        logger.info(f"Model terbaik dimuat dari: {best_ckpt}")

    test_losses, test_metrics = evaluate(
        model, test_loader, loss_fn, device, use_amp=args.use_amp
    )

    # Print laporan lengkap
    logger.info(f"\nTest Loss        : {test_losses.get('total', 0):.4f}")
    logger.info(f"Test Acc L1      : {test_metrics.get('acc_l1', 0):.4f}")
    logger.info(f"Test Acc L2      : {test_metrics.get('acc_l2', 0):.4f}")
    logger.info(f"Test Acc L3      : {test_metrics.get('acc_l3', 0):.4f}")
    logger.info(f"Test Acc Mean    : {test_metrics.get('acc_mean', 0):.4f}")
    logger.info(f"Test Hier L1+L2  : {test_metrics.get('acc_hier_l1l2', 0):.4f}")
    logger.info(f"Test Hier All    : {test_metrics.get('acc_hier_all', 0):.4f}")

    # Simpan hasil ke JSON
    results_path = os.path.join(args.log_dir, f"{args.run_name}_test_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "test_losses" : test_losses,
            "test_metrics": {k: float(v) for k, v in test_metrics.items()},
            "history"     : history,
            "args"        : vars(args),
        }, f, indent=2)
    logger.info(f"\nHasil test disimpan ke: {results_path}")

    # Simpan training history
    history_path = os.path.join(args.log_dir, f"{args.run_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history disimpan ke: {history_path}")

    logger.info("\nTraining selesai!")


if __name__ == "__main__":
    main()