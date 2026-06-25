"""
loss.py
=======
Loss function HSCN: Hierarchical Sibling Cross-Entropy.

Perubahan dari versi sebelumnya:
    - Loss L3 sekarang juga dihitung untuk sampel yang TIDAK punya anotasi L3
      spesifik (ditandai dengan sentinel -(label_l2 + 10) dari dataset.py).
    - Sentinel ini dikonversi ke local index 0 (__none__) sebelum menghitung loss.
    - Dengan begitu model belajar untuk memprediksi "__none__" saat tidak ada
      sub-tipe L3 yang cocok.

Encoding label_l3 dari dataset.py:
    -1              → L2 tidak punya L3 siblings → skip loss L3 sama sekali
    -(l2_idx + 10)  → L2 punya L3 siblings tapi L3 tidak dianotasi → __none__ (local 0)
    >= 0            → global index L3 yang valid → konversi ke local index
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from hierarchy import (
    L1_CLASSES, L2_SIBLINGS, L3_SIBLINGS,
    L2_ALL, L3_ALL,
    L2_TO_IDX, L3_TO_IDX, L2_PARENT,
    L3_NONE_LABEL, L3_LOCAL_IDX,
)


class HSCNLoss(nn.Module):
    """
    Hierarchical Sibling Cross-Entropy Loss.

    Args:
        class_weights_l1 : (3,)          tensor bobot kelas L1
        class_weights_l2 : (num_l2,)     tensor bobot kelas L2
        class_weights_l3 : (num_l3,)     tensor bobot kelas L3 (excl. __none__)
        lambda_l1        : bobot loss L1 dalam total loss
        lambda_l2        : bobot loss L2 dalam total loss
        lambda_l3        : bobot loss L3 dalam total loss
        label_smoothing  : nilai smoothing untuk CrossEntropy (0.0 = off)
    """

    def __init__(
        self,
        class_weights_l1 : Optional[torch.Tensor] = None,
        class_weights_l2 : Optional[torch.Tensor] = None,
        class_weights_l3 : Optional[torch.Tensor] = None,
        lambda_l1        : float = 1.0,
        lambda_l2        : float = 1.0,
        lambda_l3        : float = 0.5,
        label_smoothing  : float = 0.1,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.lambda_l3 = lambda_l3

        if class_weights_l1 is not None:
            self.register_buffer("cw_l1", class_weights_l1)
        else:
            self.register_buffer("cw_l1", None)

        self._cw_l2 = class_weights_l2
        self._cw_l3 = class_weights_l3

        self.label_smoothing = label_smoothing

        # Precompute: untuk setiap L2 sibling-set, berapa jumlah kelas L3
        # TERMASUK __none__ (index 0)?
        self._l3_num_classes = {
            l2: len(children)
            for l2, children in L3_SIBLINGS.items()
        }

        # Precompute: global index L3 → local index dalam sibling-set-nya
        # (untuk kelas L3 yang valid, bukan __none__)
        self._l3_global_to_local = {}
        for l2_name, children in L3_SIBLINGS.items():
            for local_i, cls in enumerate(children):
                if cls != L3_NONE_LABEL:
                    global_i = L3_TO_IDX[cls]
                    self._l3_global_to_local[(l2_name, global_i)] = local_i

    def _ce(self, logits: torch.Tensor, targets: torch.Tensor,
            weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """CrossEntropyLoss dengan optional class weights dan label smoothing."""
        return F.cross_entropy(
            logits, targets,
            weight=weight,
            label_smoothing=self.label_smoothing,
        )

    def _decode_l3_label(
        self,
        label_l3_batch: torch.Tensor,
        label_l2_batch: torch.Tensor,
        l2_name: str,
        l2_global_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode label_l3 untuk satu L2 sibling-set.

        Mengembalikan:
            mask        : boolean tensor (B,) — sampel yang relevan untuk sibling-set ini
            local_labels: long tensor (mask.sum(),) — local index dalam sibling-set L3

        Aturan decoding:
            label_l3 >= 0               → kelas L3 valid, konversi ke local index
            label_l3 == -(l2_idx + 10)  → __none__ (local index 0)
            label_l3 == -1              → skip (L2 tidak punya L3)
        """
        none_sentinel = -(l2_global_idx + 10)
        device = label_l3_batch.device

        # Mask: sampel yang L2-nya adalah l2_name ini
        mask_l2 = (label_l2_batch == l2_global_idx)

        # Mask: sampel yang punya label L3 valid (global >= 0) atau __none__ sentinel
        mask_has_l3 = (label_l3_batch >= 0) | (label_l3_batch == none_sentinel)

        # Gabungkan: hanya sampel L2 ini yang punya info L3
        mask = mask_l2 & mask_has_l3

        if mask.sum() == 0:
            return mask, torch.tensor([], dtype=torch.long, device=device)

        # Konversi ke local label
        selected_l3 = label_l3_batch[mask]
        local_labels = torch.zeros(mask.sum(), dtype=torch.long, device=device)

        for i, lbl in enumerate(selected_l3):
            lbl_val = lbl.item()
            if lbl_val == none_sentinel:
                # __none__ → local index 0
                local_labels[i] = 0
            elif lbl_val >= 0:
                # Kelas L3 valid → konversi ke local index
                key = (l2_name, lbl_val)
                local_labels[i] = self._l3_global_to_local.get(key, 0)

        return mask, local_labels

    def forward(
        self,
        model_out  : Dict[str, torch.Tensor],
        label_l1   : torch.Tensor,   # (B,) — semua valid
        label_l2   : torch.Tensor,   # (B,) — -1 jika tidak ada L2
        label_l3   : torch.Tensor,   # (B,) — -1, -(l2_idx+10), atau >= 0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Hitung total HSCN loss.

        Returns:
            total_loss : scalar tensor
            loss_dict  : dict dengan rincian loss per level & sibling-set
        """
        device    = label_l1.device
        loss_dict = {}

        # ── 1. Loss L1 ────────────────────────────────────────────────────────
        cw_l1 = self.cw_l1.to(device) if self.cw_l1 is not None else None
        loss_l1 = self._ce(model_out["logits_l1"], label_l1, weight=cw_l1)
        loss_dict["loss_l1"] = loss_l1

        # ── 2. Loss L2 (per L1 sibling-set) ───────────────────────────────────
        total_l2     = torch.tensor(0.0, device=device)
        num_l2_terms = 0

        for l1_name, l2_children in L2_SIBLINGS.items():
            l1_idx    = L1_CLASSES.index(l1_name)
            logit_key = f"logits_l2_{l1_name}"
            logits    = model_out[logit_key]   # (B, |l2_children|)

            mask_l1  = (label_l1 == l1_idx)
            mask_l2v = (label_l2 >= 0)
            mask     = mask_l1 & mask_l2v

            if mask.sum() == 0:
                continue

            l2_global_indices = [L2_TO_IDX[c] for c in l2_children]
            global_to_local   = {g: loc for loc, g in enumerate(l2_global_indices)}

            local_labels = torch.tensor(
                [global_to_local.get(label_l2[i].item(), 0)
                 for i in mask.nonzero(as_tuple=True)[0]],
                device=device, dtype=torch.long,
            )

            cw_l2_sub = None
            if self._cw_l2 is not None:
                cw_l2_sub = self._cw_l2[l2_global_indices].to(device)

            loss_l2_sub = self._ce(logits[mask], local_labels, weight=cw_l2_sub)
            loss_dict[f"loss_l2_{l1_name}"] = loss_l2_sub
            total_l2 = total_l2 + loss_l2_sub
            num_l2_terms += 1

        loss_l2 = total_l2 / max(num_l2_terms, 1)
        loss_dict["loss_l2"] = loss_l2

        # ── 3. Loss L3 (per L2 sibling-set dalam Recyclable) ──────────────────
        # PERUBAHAN UTAMA: sekarang juga menghitung loss untuk sampel __none__
        # (sentinel negatif), sehingga model belajar kapan harus berhenti di L2.

        total_l3     = torch.tensor(0.0, device=device)
        num_l3_terms = 0

        for l2_name, l3_children in L3_SIBLINGS.items():
            l2_global_idx = L2_TO_IDX[l2_name]
            logit_key     = f"logits_l3_{l2_name}"
            logits        = model_out[logit_key]   # (B, |l3_children| incl. __none__)

            mask, local_labels = self._decode_l3_label(
                label_l3, label_l2, l2_name, l2_global_idx
            )

            if mask.sum() == 0:
                continue

            # Class weights: untuk __none__ (index 0) tidak ada di cw_l3
            # (karena cw_l3 hanya untuk kelas nyata), jadi kita tidak
            # gunakan class weights untuk L3 agar tetap sederhana.
            # Alternatif: bisa dihitung terpisah jika diperlukan.
            loss_l3_sub = self._ce(logits[mask], local_labels, weight=None)
            loss_dict[f"loss_l3_{l2_name}"] = loss_l3_sub
            total_l3    = total_l3 + loss_l3_sub
            num_l3_terms += 1

        loss_l3 = total_l3 / max(num_l3_terms, 1)
        loss_dict["loss_l3"] = loss_l3

        # ── 4. Total weighted loss ─────────────────────────────────────────────
        total_loss = (
            self.lambda_l1 * loss_l1 +
            self.lambda_l2 * loss_l2 +
            self.lambda_l3 * loss_l3
        )
        loss_dict["total"] = total_loss

        return total_loss, loss_dict


# ─── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    from hierarchy import num_l1, num_l2, num_l3, L3_SIBLINGS, L2_TO_IDX

    B = 8

    # Jumlah kelas L3 per sibling-set TERMASUK __none__
    model_out = {
        "logits_l1"            : torch.randn(B, num_l1()),
        "logits_l2_Organic"    : torch.randn(B, 1),
        "logits_l2_Recyclable" : torch.randn(B, 5),
        "logits_l2_Hazardous"  : torch.randn(B, 2),
        "logits_l3_Plastic"    : torch.randn(B, 4),   # __none__ + 3 kelas = 4
        "logits_l3_Metal"      : torch.randn(B, 2),   # __none__ + 1 kelas = 2
        "logits_l3_Paper"      : torch.randn(B, 2),
        "logits_l3_Cardboard"  : torch.randn(B, 2),
        "logits_l3_Glass"      : torch.randn(B, 2),
    }

    # Simulasi label:
    # label_l2: Plastic=0, Metal=1, Paper=2, Cardboard=3, Glass=4, Food_Waste=5, ...
    # (urutan sesuai L2_ALL: Food_Waste=0, Plastic=1, Metal=2, Paper=3, Cardboard=4, Glass=5, Battery=6, E_Waste=7)
    from hierarchy import L2_ALL, L3_ALL, L2_TO_IDX
    print("L2_ALL:", L2_ALL)
    print("L3_ALL:", L3_ALL)

    plastic_idx   = L2_TO_IDX["Plastic"]
    glass_idx     = L2_TO_IDX["Glass"]
    food_idx      = L2_TO_IDX["Food_Waste"]

    label_l1 = torch.tensor([1, 1, 1, 1, 0, 1, 1, 1])   # semua Recyclable kecuali idx 4 (Organic)
    label_l2 = torch.tensor([
        plastic_idx,           # Plastic dengan L3
        plastic_idx,           # Plastic tanpa L3 → sentinel
        glass_idx,             # Glass dengan L3
        glass_idx,             # Glass tanpa L3 → sentinel
        food_idx,              # Food_Waste (tidak ada L3) → -1
        plastic_idx,           # Plastic dengan L3
        glass_idx,             # Glass tanpa L3 → sentinel
        plastic_idx,           # Plastic tanpa L3 → sentinel
    ])

    from hierarchy import L3_TO_IDX
    plastic_bottle_idx = L3_TO_IDX["Plastic_Bottle"]
    glass_bottle_idx   = L3_TO_IDX["Glass_Bottle"]

    # Sentinel untuk __none__
    plastic_none = -(plastic_idx + 10)
    glass_none   = -(glass_idx + 10)

    label_l3 = torch.tensor([
        plastic_bottle_idx,   # Plastic → Plastic_Bottle
        plastic_none,         # Plastic → __none__
        glass_bottle_idx,     # Glass → Glass_Bottle
        glass_none,           # Glass → __none__
        -1,                   # Food_Waste → skip
        plastic_bottle_idx,   # Plastic → Plastic_Bottle
        glass_none,           # Glass → __none__
        plastic_none,         # Plastic → __none__
    ])

    loss_fn   = HSCNLoss()
    total, ld = loss_fn(model_out, label_l1, label_l2, label_l3)

    print(f"\nTotal loss: {total.item():.4f}")
    for k, v in ld.items():
        print(f"  {k}: {v.item():.4f}")