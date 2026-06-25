"""
detect_classify.py
==================
Visualisasi bersih mengikuti referensi:
  - Bounding box KUNING tipis
  - Label kotak terpisah di pojok kanan atas box: L1 / L2 / L3 (jika ada)
  - L3 hanya ditampilkan jika model TIDAK memprediksi __none__
  - Tidak ada summary bar, tidak ada leader line, tidak ada elemen lain
  - Label di LUAR box (di atas atau di bawah jika mepet atas)
"""

import os, sys
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "hscn_waste"))
from hierarchy import L1_CLASSES, L2_SIBLINGS, L3_SIBLINGS, L3_NONE_LABEL
from dataset import build_transforms
from model import HSCN
from ultralytics import YOLO

# ─── Path ────────────────────────────────────────────────────────────────────
IMAGE_PATH      = "/Users/afinatsal/Desktop/skripsi/testImage/realwaste_glass_000017.jpg"
YOLO_WEIGHTS    = "/Users/afinatsal/Desktop/skripsi/waste_detection_results_yolol/weights/best.pt"
HSCN_CHECKPOINT = "waste_hscn_results_efficientnet_b3/weights/hscn_waste_efficientnet_b3_best.pth"
OUTPUT_PATH     = "hasil_deteksi.jpg"
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BOX_COLOR    = (255, 220, 0)
BOX_LW       = 2
LABEL_GAP    = 4
LABEL_PAD_X  = 8
LABEL_PAD_Y  = 5
LABEL_BG     = (20, 22, 28, 220)
LABEL_RADIUS = 4
FONT_SIZE    = 13


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def load_font(size=13, bold=False):
    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def twh(draw, text, font):
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0], bb[3] - bb[1]


def round_rect(draw, xy, r, fill=None, outline=None, lw=1):
    try:
        draw.rounded_rectangle(xy, radius=r, fill=fill, outline=outline, width=lw)
    except Exception:
        draw.rectangle(xy, fill=fill, outline=outline, width=lw)


def draw_label_box(draw, x, y, text, font):
    tw, th = twh(draw, text, font)
    w = tw + LABEL_PAD_X * 2
    h = th + LABEL_PAD_Y * 2
    round_rect(draw, [x, y, x + w, y + h],
               r=LABEL_RADIUS, fill=LABEL_BG, outline=(*BOX_COLOR, 200), lw=1)
    draw.text((x + LABEL_PAD_X, y + LABEL_PAD_Y), text, font=font,
              fill=(240, 240, 240, 255))
    return w, h


def get_best(logits):
    probs = F.softmax(logits, dim=-1)
    conf, idx = probs.max(dim=-1)
    return int(idx.item()), float(conf.item())


# ── Load models ──────────────────────────────────────────────────────────────
yolo_model = YOLO(YOLO_WEIGHTS)

ckpt      = torch.load(HSCN_CHECKPOINT, map_location=device)
ckpt_args = ckpt.get("args", {})
hscn_model = HSCN(
    backbone_name=ckpt_args.get("backbone", "efficientnet_b3"),
    pretrained=False,
    hidden_dim=ckpt_args.get("hidden_dim", 512),
    dropout=ckpt_args.get("dropout", 0.5),
).to(device)
hscn_model.load_state_dict(ckpt["model_state"])
hscn_model.eval()
transform = build_transforms("test")

# ── Proses gambar ─────────────────────────────────────────────────────────────
base_img  = Image.open(IMAGE_PATH).convert("RGBA")
img_w, img_h = base_img.size
overlay   = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
draw      = ImageDraw.Draw(overlay)

font_bold = load_font(FONT_SIZE, bold=True)
font_reg  = load_font(FONT_SIZE, bold=False)

results   = yolo_model.predict(source=IMAGE_PATH, verbose=False)[0]
boxes     = results.boxes
num_boxes = 0 if boxes is None else len(boxes)
print(f"Jumlah objek terdeteksi: {num_boxes}")

if boxes is not None and len(boxes) > 0:
    boxes_xyxy = boxes.xyxy.cpu().numpy()
    boxes_conf = boxes.conf.cpu().numpy() if boxes.conf is not None else None

    for i, box in enumerate(boxes_xyxy, start=1):
        x1, y1, x2, y2 = [float(v) for v in box]
        x1i = clamp(int(round(x1)), 0, img_w - 1)
        y1i = clamp(int(round(y1)), 0, img_h - 1)
        x2i = clamp(int(round(x2)), x1i + 1, img_w)
        y2i = clamp(int(round(y2)), y1i + 1, img_h)

        # ── HSCN inference ───────────────────────────────────────────────────
        crop = base_img.convert("RGB").crop((x1i, y1i, x2i, y2i))
        img_tensor = transform(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            out  = hscn_model(img_tensor)
            pred = hscn_model.predict(img_tensor)

        # L1
        l1_idx  = pred["pred_l1"][0].item()
        l1_name = L1_CLASSES[l1_idx]

        # L2
        l2_logits       = out[f"logits_l2_{l1_name}"][0]
        l2_idx, l2_conf = get_best(l2_logits)
        l2_name         = L2_SIBLINGS[l1_name][l2_idx]

        # L3 — HANYA tampilkan jika:
        #   1. L2 punya L3 siblings
        #   2. Model TIDAK memprediksi __none__ (local index 0)
        l3_name = None
        l3_conf = None
        if l2_name in L3_SIBLINGS:
            l3_logits        = out[f"logits_l3_{l2_name}"][0]
            l3_local_idx, l3_conf = get_best(l3_logits)

            if l3_local_idx == 0:
                # Model memilih __none__ → berhenti di L2
                l3_name = None
            else:
                # Model memilih sub-tipe L3 spesifik
                l3_name = L3_SIBLINGS[l2_name][l3_local_idx]

        yolo_conf = float(boxes_conf[i - 1]) if boxes_conf is not None else None

        # Log hasil
        log_str = f"  - Box {i}: {l1_name} > {l2_name}"
        if l3_name:
            log_str += f" > {l3_name} (conf={l3_conf:.2f})"
        else:
            log_str += " [berhenti di L2]"
        if yolo_conf:
            log_str += f" | det={yolo_conf:.2f}"
        print(log_str)

        # ── Bounding box kuning tipis ─────────────────────────────────────────
        draw.rectangle([x1i, y1i, x2i, y2i],
                       outline=(*BOX_COLOR, 255), width=BOX_LW)

        # ── Susun label kotak ─────────────────────────────────────────────────
        labels = [
            f"L1: {l1_name.replace('_', ' ')}",
            f"L2: {l2_name.replace('_', ' ')}",
        ]
        if l3_name:
            labels.append(f"L3: {l3_name.replace('_', ' ')}")
        # Jika tidak ada l3_name, hanya tampilkan L1 dan L2

        # Ukur dimensi label
        label_heights = []
        label_widths  = []
        for txt in labels:
            tw_v, th_v = twh(draw, txt, font_bold)
            lw_v = tw_v + LABEL_PAD_X * 2
            lh_v = th_v + LABEL_PAD_Y * 2
            label_widths.append(lw_v)
            label_heights.append(lh_v)

        total_h = sum(label_heights) + LABEL_GAP * (len(labels) - 1)
        max_w   = max(label_widths)

        lx = x2i - max_w
        lx = clamp(lx, 0, img_w - max_w)

        if y1i - total_h - LABEL_GAP >= 0:
            ly = y1i - total_h - LABEL_GAP
        else:
            ly = y2i + LABEL_GAP

        for txt, lh in zip(labels, label_heights):
            draw_label_box(draw, lx, ly, txt, font_bold)
            ly += lh + LABEL_GAP

# ── Composite & simpan ────────────────────────────────────────────────────────
final = Image.alpha_composite(base_img, overlay).convert("RGB")
final.save(OUTPUT_PATH, quality=95, subsampling=0)
print(f"\nHasil disimpan ke: {OUTPUT_PATH}")