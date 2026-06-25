"""
predict.py
==========
Script inference untuk HSCN Waste Classification.

Perubahan dari versi sebelumnya:
    - predict_single() menampilkan L3 hanya jika model TIDAK memprediksi __none__
    - predict_folder() menyimpan L3 = None jika model memilih berhenti di L2
    - evaluate_test() menggunakan HSCNMetrics yang sudah mendukung __none__

Cara penggunaan:
    python predict.py --image path/to/image.jpg --checkpoint checkpoints/hscn_waste_best.pth
    python predict.py --eval_test --checkpoint checkpoints/hscn_waste_best.pth
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
    L3_NONE_LABEL,
)
from dataset import build_transforms, INPUT_SIZE
from model   import HSCN
from metrics import HSCNMetrics


# ─── Argument Parser ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="HSCN Waste Classification Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image",      type=str, default=None)
    parser.add_argument("--image_dir",  type=str, default=None)
    parser.add_argument("--eval_test",  action="store_true")
    parser.add_argument("--data_dir",   type=str, default="dataset_hscn")
    parser.add_argument("--output",     type=str, default="predictions.json")
    parser.add_argument("--device",     type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--top_k",      type=int, default=3)
    return parser.parse_args()


# ─── Load Model ───────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> HSCN:
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt.get("args", {})

    model = HSCN(
        backbone_name = args.get("backbone", "resnet50"),
        pretrained    = False,
        hidden_dim    = args.get("hidden_dim", 512),
        dropout       = args.get("dropout", 0.5),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model dimuat dari  : {checkpoint_path}")
    print(f"Backbone           : {args.get('backbone', 'resnet50')}")
    return model


# ─── Predict Satu Gambar ──────────────────────────────────────────────────────

def predict_single(
    model      : HSCN,
    image_path : str,
    device     : torch.device,
    top_k      : int = 3,
) -> Dict:
    """
    Prediksi satu gambar.

    L3 dalam hasil:
        - None  jika model memprediksi __none__ (berhenti di L2)
        - str   nama kelas L3 jika ada sub-tipe spesifik
    """
    transform = build_transforms("test")

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"error": str(e), "image": image_path}

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out  = model(img_tensor)
        pred = model.predict(img_tensor)

    # L1
    prob_l1 = F.softmax(out["logits_l1"], dim=-1)[0]
    topk_vals, topk_ids = prob_l1.topk(min(top_k, len(L1_CLASSES)))
    l1_predictions = [
        {"class": L1_CLASSES[i.item()], "prob": float(v.item())}
        for v, i in zip(topk_vals, topk_ids)
    ]

    best_l1_idx  = pred["pred_l1"][0].item()
    best_l1_name = L1_CLASSES[best_l1_idx]

    # L2
    best_l2_idx = pred["pred_l2"][0].item()
    best_l2_name = L2_ALL[best_l2_idx] if best_l2_idx >= 0 else None

    # L3 — None jika pred_l3 == -1 (model pilih __none__)
    best_l3_idx  = pred["pred_l3"][0].item()
    best_l3_name = L3_ALL[best_l3_idx] if best_l3_idx >= 0 else None

    # Distribusi L2
    l2_logits = out.get(f"logits_l2_{best_l1_name}")
    l2_preds  = []
    if l2_logits is not None and best_l1_name in L2_SIBLINGS:
        l2_probs = F.softmax(l2_logits, dim=-1)[0]
        for i, cls in enumerate(L2_SIBLINGS[best_l1_name]):
            l2_preds.append({"class": cls, "prob": float(l2_probs[i].item())})
        l2_preds.sort(key=lambda x: x["prob"], reverse=True)

    # Distribusi L3 — tampilkan termasuk __none__ agar user tahu probabilitasnya
    l3_preds = []
    if best_l2_name and best_l2_name in L3_SIBLINGS:
        l3_logits = out.get(f"logits_l3_{best_l2_name}")
        if l3_logits is not None:
            l3_probs = F.softmax(l3_logits, dim=-1)[0]
            for local_i, cls in enumerate(L3_SIBLINGS[best_l2_name]):
                display_name = "(berhenti di L2)" if cls == L3_NONE_LABEL else cls
                l3_preds.append({
                    "class": display_name,
                    "prob" : float(l3_probs[local_i].item()),
                    "is_none": cls == L3_NONE_LABEL,
                })
            l3_preds.sort(key=lambda x: x["prob"], reverse=True)

    return {
        "image"          : str(image_path),
        "prediction"     : {
            "L1": best_l1_name,
            "L2": best_l2_name,
            "L3": best_l3_name,   # None jika model pilih berhenti di L2
        },
        "l3_stopped_at_l2": best_l3_name is None and best_l2_name in L3_SIBLINGS,
        "l1_topk"         : l1_predictions,
        "l2_distribution" : l2_preds,
        "l3_distribution" : l3_preds,
    }


# ─── Predict Folder ───────────────────────────────────────────────────────────

def predict_folder(
    model      : HSCN,
    folder_path: str,
    device     : torch.device,
    batch_size : int = 32,
) -> List[Dict]:
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
            l3_idx = preds["pred_l3"][j].item()   # -1 jika __none__

            results.append({
                "image": str(path),
                "prediction": {
                    "L1": L1_CLASSES[l1_idx] if l1_idx >= 0 else None,
                    "L2": L2_ALL[l2_idx]     if l2_idx >= 0 else None,
                    "L3": L3_ALL[l3_idx]     if l3_idx >= 0 else None,
                    # None di L3 berarti model memilih berhenti di L2
                }
            })

        print(f"  [{i + len(batch_paths)}/{len(img_paths)}] selesai")

    return results


# ─── Evaluate Test Set ────────────────────────────────────────────────────────

def evaluate_test(
    model     : HSCN,
    data_dir  : str,
    device    : torch.device,
    batch_size: int = 32,
) -> Dict:
    from dataset import WasteHSCNDataset

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
            preds = model.predict(imgs)

        metrics.update(preds, l1, l2, l3)

    results = metrics.compute()
    print(metrics.format_report(results))
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    model = load_model(args.checkpoint, device)

    if args.image:
        result = predict_single(model, args.image, device, top_k=args.top_k)
        print("\n=== Hasil Prediksi ===")
        print(f"Gambar : {result['image']}")
        pred = result.get("prediction", {})
        print(f"L1     : {pred.get('L1', '-')}")
        print(f"L2     : {pred.get('L2', '-')}")

        l3_val = pred.get("L3")
        if l3_val:
            print(f"L3     : {l3_val}")
        elif result.get("l3_stopped_at_l2"):
            print(f"L3     : - (model memilih berhenti di L2)")
        else:
            print(f"L3     : - (L2 tidak memiliki sub-tipe L3)")

        print("\nTop-K L1 probabilities:")
        for p in result.get("l1_topk", []):
            print(f"  {p['class']:<15}: {p['prob']:.4f}")

        if result.get("l2_distribution"):
            print("\nL2 distribution:")
            for p in result["l2_distribution"]:
                print(f"  {p['class']:<20}: {p['prob']:.4f}")

        if result.get("l3_distribution"):
            print("\nL3 distribution (termasuk opsi berhenti di L2):")
            for p in result["l3_distribution"]:
                flag = " ← (berhenti di L2)" if p.get("is_none") else ""
                print(f"  {p['class']:<30}: {p['prob']:.4f}{flag}")

        with open(args.output, "w") as f:
            json.dump([result], f, indent=2)
        print(f"\nHasil disimpan ke: {args.output}")

    elif args.image_dir:
        results = predict_folder(model, args.image_dir, device, args.batch_size)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nTotal: {len(results)} gambar")
        print(f"Hasil disimpan ke: {args.output}")

    elif args.eval_test:
        results = evaluate_test(model, args.data_dir, device, args.batch_size)
        with open(args.output, "w") as f:
            json.dump({k: (float(v) if not isinstance(v, str) else v)
                       for k, v in results.items()}, f, indent=2)
        print(f"\nHasil evaluasi disimpan ke: {args.output}")

    else:
        print("Pilih salah satu: --image, --image_dir, atau --eval_test")
        sys.exit(1)


if __name__ == "__main__":
    main()