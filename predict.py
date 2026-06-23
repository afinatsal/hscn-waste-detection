"""
predict.py
==========
Script inference untuk HSCN Waste Classification.

Cara penggunaan:
    # Prediksi satu gambar
    python predict.py --image path/to/image.jpg --checkpoint checkpoints/hscn_waste_best.pth

    # Prediksi seluruh test set dan simpan hasilnya
    python predict.py --eval_test --checkpoint checkpoints/hscn_waste_best.pth

    # Prediksi folder gambar
    python predict.py --image_dir path/to/folder --checkpoint checkpoints/hscn_waste_best.pth
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from hierarchy import (
    L1_CLASSES, L2_ALL, L3_ALL,
    L2_SIBLINGS, L3_SIBLINGS,
    L2_TO_IDX, L3_TO_IDX,
)
from dataset import build_transforms, INPUT_SIZE
from model   import HSCN
from loss    import HSCNLoss
from metrics import HSCNMetrics


# ─── Argument Parser ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="HSCN Waste Classification Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path ke file checkpoint (.pth)")
    parser.add_argument("--image",      type=str, default=None,
                        help="Path ke satu file gambar")
    parser.add_argument("--image_dir",  type=str, default=None,
                        help="Path ke folder berisi gambar")
    parser.add_argument("--eval_test",  action="store_true",
                        help="Evaluasi pada test set")
    parser.add_argument("--data_dir",   type=str, default="dataset_hscn",
                        help="Root direktori dataset (untuk --eval_test)")
    parser.add_argument("--output",     type=str, default="predictions.json",
                        help="File output JSON")
    parser.add_argument("--device",     type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--top_k",      type=int, default=3,
                        help="Tampilkan top-K prediksi L1")
    return parser.parse_args()


# ─── Load Model ───────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> HSCN:
    """Muat model dari checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt.get("args", {})

    model = HSCN(
        backbone_name = args.get("backbone", "resnet50"),
        pretrained    = False,  # tidak perlu download saat inference
        hidden_dim    = args.get("hidden_dim", 512),
        dropout       = args.get("dropout", 0.5),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model dimuat dari: {checkpoint_path}")
    print(f"Backbone: {args.get('backbone', 'resnet50')}")
    return model


# ─── Predict Satu Gambar ──────────────────────────────────────────────────────

def predict_single(
    model      : HSCN,
    image_path : str,
    device     : torch.device,
    top_k      : int = 3,
) -> Dict:
    """
    Prediksi satu gambar, kembalikan label hierarki lengkap dengan probabilitas.
    """
    transform = build_transforms("test")

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"error": str(e), "image": image_path}

    img_tensor = transform(img).unsqueeze(0).to(device)  # (1, 3, H, W)

    with torch.no_grad():
        out  = model(img_tensor)
        pred = model.predict(img_tensor)

    # Probabilitas L1
    prob_l1 = F.softmax(out["logits_l1"], dim=-1)[0]  # (3,)

    # Top-K L1
    topk_vals, topk_ids = prob_l1.topk(min(top_k, len(L1_CLASSES)))
    l1_predictions = [
        {"class": L1_CLASSES[i.item()], "prob": float(v.item())}
        for v, i in zip(topk_vals, topk_ids)
    ]

    # Best prediction path
    best_l1_idx  = pred["pred_l1"][0].item()
    best_l1_name = L1_CLASSES[best_l1_idx]

    best_l2_idx  = pred["pred_l2"][0].item()
    best_l2_name = L2_ALL[best_l2_idx] if best_l2_idx >= 0 else None

    best_l3_idx  = pred["pred_l3"][0].item()
    best_l3_name = L3_ALL[best_l3_idx] if best_l3_idx >= 0 else None

    # Probabilitas L2 dan L3 untuk kelas terprediksi
    l2_logits  = out.get(f"logits_l2_{best_l1_name}")
    l2_probs   = F.softmax(l2_logits, dim=-1)[0] if l2_logits is not None else None

    l2_preds = []
    if l2_logits is not None and best_l1_name in L2_SIBLINGS:
        for i, cls in enumerate(L2_SIBLINGS[best_l1_name]):
            l2_preds.append({
                "class": cls,
                "prob" : float(l2_probs[i].item())
            })
        l2_preds.sort(key=lambda x: x["prob"], reverse=True)

    l3_preds = []
    if best_l2_name and best_l2_name in L3_SIBLINGS:
        l3_logits = out.get(f"logits_l3_{best_l2_name}")
        if l3_logits is not None:
            l3_probs = F.softmax(l3_logits, dim=-1)[0]
            for i, cls in enumerate(L3_SIBLINGS[best_l2_name]):
                l3_preds.append({
                    "class": cls,
                    "prob" : float(l3_probs[i].item())
                })
            l3_preds.sort(key=lambda x: x["prob"], reverse=True)

    return {
        "image"          : str(image_path),
        "prediction"     : {
            "L1" : best_l1_name,
            "L2" : best_l2_name,
            "L3" : best_l3_name,
        },
        "l1_topk"        : l1_predictions,
        "l2_distribution": l2_preds,
        "l3_distribution": l3_preds,
    }


# ─── Predict Folder ───────────────────────────────────────────────────────────

def predict_folder(
    model      : HSCN,
    folder_path: str,
    device     : torch.device,
    batch_size : int = 32,
) -> List[Dict]:
    """Prediksi semua gambar dalam satu folder."""
    from torch.utils.data import Dataset, DataLoader
    from dataset import build_transforms
    import torchvision.transforms as T

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_paths = [
        p for p in Path(folder_path).iterdir()
        if p.suffix.lower() in img_extensions
    ]

    if not img_paths:
        print(f"Tidak ada gambar di: {folder_path}")
        return []

    print(f"Memproses {len(img_paths)} gambar dari: {folder_path}")

    transform = build_transforms("test")
    results   = []

    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i + batch_size]
        tensors     = []
        valid_paths = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(transform(img))
                valid_paths.append(p)
            except Exception as e:
                print(f"[WARN] Gagal membuka {p}: {e}")
                results.append({"image": str(p), "error": str(e)})

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)

        with torch.no_grad():
            preds = model.predict(batch)

        for j, path in enumerate(valid_paths):
            l1_idx = preds["pred_l1"][j].item()
            l2_idx = preds["pred_l2"][j].item()
            l3_idx = preds["pred_l3"][j].item()
            results.append({
                "image": str(path),
                "prediction": {
                    "L1": L1_CLASSES[l1_idx] if l1_idx >= 0 else None,
                    "L2": L2_ALL[l2_idx]     if l2_idx >= 0 else None,
                    "L3": L3_ALL[l3_idx]     if l3_idx >= 0 else None,
                }
            })

        print(f"  [{i + len(batch_paths)}/{len(img_paths)}] selesai")

    return results


# ─── Evaluate Test Set ────────────────────────────────────────────────────────

def evaluate_test(
    model    : HSCN,
    data_dir : str,
    device   : torch.device,
    batch_size: int = 32,
) -> Dict:
    """Evaluasi penuh pada test set dan cetak laporan."""
    from dataset  import WasteHSCNDataset, build_dataloaders
    from metrics  import HSCNMetrics

    test_ds = WasteHSCNDataset(
        root_dir=os.path.join(data_dir, "test"),
        split="test",
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2
    )

    metrics = HSCNMetrics()
    model.eval()

    for imgs, l1, l2, l3 in test_loader:
        imgs = imgs.to(device)
        l1   = l1.to(device)
        l2   = l2.to(device)
        l3   = l3.to(device)

        with torch.no_grad():
            # predict_with_probs() → mAP + confusion matrix tersedia
            preds = model.predict_with_probs(imgs)

        metrics.update(preds, l1, l2, l3)

    results = metrics.compute()
    # format_report() mencetak mAP, acc, dan confusion matrix sekaligus
    print(metrics.format_report(results))
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = load_model(args.checkpoint, device)

    if args.image:
        # ── Prediksi satu gambar ───────────────────────────────────────────────
        result = predict_single(model, args.image, device, top_k=args.top_k)
        print("\n=== Hasil Prediksi ===")
        print(f"Gambar : {result['image']}")
        pred = result.get("prediction", {})
        print(f"L1     : {pred.get('L1', '-')}")
        print(f"L2     : {pred.get('L2', '-')}")
        print(f"L3     : {pred.get('L3', '-')}")
        print("\nTop-K L1 probabilities:")
        for p in result.get("l1_topk", []):
            print(f"  {p['class']:<15}: {p['prob']:.4f}")
        if result.get("l2_distribution"):
            print("\nL2 distribution:")
            for p in result["l2_distribution"]:
                print(f"  {p['class']:<20}: {p['prob']:.4f}")
        if result.get("l3_distribution"):
            print("\nL3 distribution:")
            for p in result["l3_distribution"]:
                print(f"  {p['class']:<25}: {p['prob']:.4f}")

        with open(args.output, "w") as f:
            json.dump([result], f, indent=2)
        print(f"\nHasil disimpan ke: {args.output}")

    elif args.image_dir:
        # ── Prediksi folder ────────────────────────────────────────────────────
        results = predict_folder(model, args.image_dir, device, args.batch_size)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nTotal: {len(results)} gambar")
        print(f"Hasil disimpan ke: {args.output}")

    elif args.eval_test:
        # ── Evaluasi test set ──────────────────────────────────────────────────
        results = evaluate_test(model, args.data_dir, device, args.batch_size)
        with open(args.output, "w") as f:
            json.dump({k: float(v) for k, v in results.items()}, f, indent=2)
        print(f"\nHasil evaluasi disimpan ke: {args.output}")

    else:
        print("Pilih salah satu: --image, --image_dir, atau --eval_test")
        sys.exit(1)


if __name__ == "__main__":
    main()