"""
dataset.py
==========
PyTorch Dataset untuk HSCN Waste Classification.

Membaca labels.json yang berformat:
    [{"image": "train_0001.jpg", "L1": "Recyclable", "L2": "Plastic", "L3": "Plastic_Bottle"}, ...]

Menghasilkan per-sample:
    - image tensor  : (3, H, W)  — setelah transform
    - label_l1      : int  (index ke L1_CLASSES)
    - label_l2      : int  (index ke L2_ALL; -1 jika tidak ada)
    - label_l3      : int  (index ke L3_ALL; -1 jika tidak ada / leaf di L2)
"""

import os
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from hierarchy import (
    L1_TO_IDX, L2_TO_IDX, L3_TO_IDX,
    L2_NO_L3, L3_SIBLINGS,
    num_l1, num_l2, num_l3,
)

# ─── Normalisasi ImageNet (standar untuk backbone pra-latih) ──────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─── Ukuran input standar ─────────────────────────────────────────────────────
INPUT_SIZE = 224  # sesuai paper (bounding box warped ke 224×224)


def build_transforms(split: str = "train") -> T.Compose:
    """
    Augmentasi ringan untuk split 'train', transform minimal untuk 'val'/'test'.
    Disesuaikan dengan gambar crop sampah berukuran rata-rata ~250×250 px.
    """
    if split == "train":
        return T.Compose([
            T.Resize((INPUT_SIZE + 32, INPUT_SIZE + 32)),   # sedikit lebih besar
            T.RandomCrop(INPUT_SIZE),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.2, hue=0.05),
            T.RandomRotation(degrees=15),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize((INPUT_SIZE, INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


class WasteHSCNDataset(Dataset):
    """
    Dataset sampah dengan label hirarki 3 level untuk HSCN.

    Args:
        root_dir : path ke folder split, mis. 'dataset_hscn/train'
        split    : 'train' | 'valid' | 'test'
        transform: torchvision transform (jika None, gunakan default)
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[T.Compose] = None,
    ):
        self.root_dir   = Path(root_dir)
        self.split      = split
        self.image_dir  = self.root_dir / "image"
        self.labels_path = self.root_dir / "labels.json"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image dir not found: {self.image_dir}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")

        with open(self.labels_path, "r") as f:
            raw = json.load(f)

        # Normalisasi entry: semua key ke title-case dengan underscore
        self.samples = []
        for entry in raw:
            sample = self._parse_entry(entry)
            if sample is not None:
                self.samples.append(sample)

        self.transform = transform if transform else build_transforms(split)

        # Hitung distribusi kelas untuk class-weighted loss
        self._compute_class_weights()

    def _parse_entry(self, entry: dict) -> Optional[dict]:
        """
        Parsing satu entry labels.json → dict dengan kunci standar.
        Mengembalikan None jika L1 tidak dikenali.
        """
        img_name = entry.get("image", "")
        l1_raw   = entry.get("L1", "")
        l2_raw   = entry.get("L2", "")
        l3_raw   = entry.get("L3", "")

        # Normalisasi: ganti spasi → underscore, capitalize setiap kata
        def norm(s):
            if not s:
                return ""
            return s.strip().replace(" ", "_")

        l1 = norm(l1_raw)
        l2 = norm(l2_raw)
        l3 = norm(l3_raw)

        # Validasi L1
        if l1 not in L1_TO_IDX:
            print(f"[WARN] L1='{l1}' tidak dikenali, entry dilewati: {entry}")
            return None

        label_l1 = L1_TO_IDX[l1]
        label_l2 = L2_TO_IDX.get(l2, -1)
        label_l3 = L3_TO_IDX.get(l3, -1)

        return {
            "image"   : img_name,
            "label_l1": label_l1,
            "label_l2": label_l2,
            "label_l3": label_l3,
            "l1_str"  : l1,
            "l2_str"  : l2,
            "l3_str"  : l3,
        }

    def _compute_class_weights(self):
        """
        Hitung bobot per kelas pada setiap level untuk mengatasi imbalance.
        Metode: inverse frequency (diclamp ke [0.1, 10]).
        """
        counts_l1 = np.zeros(num_l1(), dtype=np.float32)
        counts_l2 = np.zeros(num_l2(), dtype=np.float32)
        counts_l3 = np.zeros(num_l3(), dtype=np.float32)

        for s in self.samples:
            counts_l1[s["label_l1"]] += 1
            if s["label_l2"] >= 0:
                counts_l2[s["label_l2"]] += 1
            if s["label_l3"] >= 0:
                counts_l3[s["label_l3"]] += 1

        def safe_inv(counts):
            total = counts.sum()
            w = np.where(counts > 0, total / (len(counts) * counts + 1e-6), 0)
            w = np.clip(w, 0.1, 10.0)
            return torch.tensor(w, dtype=torch.float32)

        self.class_weights_l1 = safe_inv(counts_l1)
        self.class_weights_l2 = safe_inv(counts_l2)
        self.class_weights_l3 = safe_inv(counts_l3)

        self.counts_l1 = counts_l1
        self.counts_l2 = counts_l2
        self.counts_l3 = counts_l3

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, int]:
        sample = self.samples[idx]

        img_path = self.image_dir / sample["image"]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback: buat gambar hitam jika file rusak
            print(f"[WARN] Gagal membuka {img_path}: {e}")
            img = Image.fromarray(np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8))

        img = self.transform(img)

        return (
            img,
            sample["label_l1"],
            sample["label_l2"],
            sample["label_l3"],
        )

    def print_stats(self):
        """Tampilkan statistik distribusi label."""
        from hierarchy import L1_CLASSES, L2_ALL, L3_ALL
        print(f"\n{'='*50}")
        print(f"Dataset [{self.split}]  |  Total: {len(self.samples)} sampel")
        print(f"{'='*50}")
        print("L1 distribution:")
        for i, name in enumerate(L1_CLASSES):
            print(f"  {name:<15}: {int(self.counts_l1[i]):>5}")
        print("L2 distribution:")
        for i, name in enumerate(L2_ALL):
            if self.counts_l2[i] > 0:
                print(f"  {name:<20}: {int(self.counts_l2[i]):>5}")
        print("L3 distribution:")
        for i, name in enumerate(L3_ALL):
            if self.counts_l3[i] > 0:
                print(f"  {name:<25}: {int(self.counts_l3[i]):>5}")


def build_dataloaders(
    dataset_root: str = "dataset_hscn",
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Buat DataLoader untuk train, valid, dan test split.

    Returns:
        train_loader, valid_loader, test_loader
    """
    train_ds = WasteHSCNDataset(
        root_dir=os.path.join(dataset_root, "train"),
        split="train",
    )
    valid_ds = WasteHSCNDataset(
        root_dir=os.path.join(dataset_root, "valid"),
        split="valid",
    )
    test_ds = WasteHSCNDataset(
        root_dir=os.path.join(dataset_root, "test"),
        split="test",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader, test_loader, train_ds


if __name__ == "__main__":
    # Uji coba cepat
    ds = WasteHSCNDataset("dataset_hscn/train", split="train")
    ds.print_stats()
    img, l1, l2, l3 = ds[0]
    print(f"\nSample[0]: img={img.shape}, L1={l1}, L2={l2}, L3={l3}")
