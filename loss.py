"""
loss.py
=======
Loss function HSCN: Hierarchical Sibling Cross-Entropy.

Sesuai paper HSCN, loss dihitung dengan cara:
    1. Gunakan SoftMax TERPISAH untuk setiap sibling-set di setiap level.
    2. L2 loss hanya dihitung untuk sampel yang memiliki label L2 valid.
    3. L3 loss hanya dihitung untuk sampel yang memiliki label L3 valid
       (partial annotation handling).
    4. Total loss = λ1*L(L1) + λ2*L(L2) + λ3*L(L3)

Setiap level menggunakan CrossEntropyLoss dengan class weights
untuk menangani ketidakseimbangan kelas.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from hierarchy import (
    L1_CLASSES, L2_SIBLINGS, L3_SIBLINGS,
    L2_ALL, L3_ALL,
    L2_TO_IDX, L3_TO_IDX, L2_PARENT,
)


class HSCNLoss(nn.Module):
    """
    Hierarchical Sibling Cross-Entropy Loss.

    Args:
        class_weights_l1 : (3,)          tensor bobot kelas L1
        class_weights_l2 : (num_l2,)     tensor bobot kelas L2
        class_weights_l3 : (num_l3,)     tensor bobot kelas L3
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
        lambda_l3        : float = 0.5,   # lebih rendah karena partial annotation
        label_smoothing  : float = 0.1,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.lambda_l3 = lambda_l3

        # Daftarkan weights sebagai buffer agar berpindah device otomatis
        if class_weights_l1 is not None:
            self.register_buffer("cw_l1", class_weights_l1)
        else:
            self.register_buffer("cw_l1", None)

        # L2 & L3 weights — simpan sebagai dict of buffers
        # Karena nn.Module tidak support ModuleDict of buffers langsung,
        # kita simpan sebagai atribut nn.Parameter dengan requires_grad=False
        self._cw_l2 = class_weights_l2  # (num_l2,) | None
        self._cw_l3 = class_weights_l3  # (num_l3,) | None

        self.label_smoothing = label_smoothing

    def _ce(self, logits: torch.Tensor, targets: torch.Tensor,
            weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """CrossEntropyLoss dengan optional class weights dan label smoothing."""
        return F.cross_entropy(
            logits, targets,
            weight=weight,
            label_smoothing=self.label_smoothing,
        )

    def forward(
        self,
        model_out  : Dict[str, torch.Tensor],
        label_l1   : torch.Tensor,  # (B,) — semua valid
        label_l2   : torch.Tensor,  # (B,) — -1 jika tidak ada L2
        label_l3   : torch.Tensor,  # (B,) — -1 jika tidak ada L3
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Hitung total HSCN loss.

        Returns:
            total_loss : scalar tensor
            loss_dict  : dict dengan rincian loss per level & sibling-set
        """
        device     = label_l1.device
        loss_dict  = {}

        # ── 1. Loss L1 ────────────────────────────────────────────────────────
        cw_l1 = self.cw_l1.to(device) if self.cw_l1 is not None else None
        loss_l1 = self._ce(model_out["logits_l1"], label_l1, weight=cw_l1)
        loss_dict["loss_l1"] = loss_l1

        # ── 2. Loss L2 (per L1 sibling-set) ───────────────────────────────────
        # Untuk setiap l1 sibling set, hitung loss hanya pada sampel
        # yang benar-benar berasal dari L1 tersebut dan memiliki label L2 valid.

        total_l2 = torch.tensor(0.0, device=device)
        num_l2_terms = 0

        for l1_name, l2_children in L2_SIBLINGS.items():
            l1_idx    = L1_CLASSES.index(l1_name)
            logit_key = f"logits_l2_{l1_name}"
            logits    = model_out[logit_key]  # (B, |l2_children|)

            # Mask: sampel dengan L1 == l1_idx DAN L2 valid
            mask_l1    = (label_l1 == l1_idx)
            mask_l2v   = (label_l2 >= 0)
            mask       = mask_l1 & mask_l2v

            if mask.sum() == 0:
                continue

            # Konversi label_l2 global → local index dalam sibling-set
            # (index dalam l2_children list untuk l1_name ini)
            l2_global_indices = [L2_TO_IDX[c] for c in l2_children]
            # Buat mapping global_idx → local_idx
            global_to_local = {g: loc for loc, g in enumerate(l2_global_indices)}

            local_labels = torch.tensor(
                [global_to_local.get(label_l2[i].item(), 0) for i in mask.nonzero(as_tuple=True)[0]],
                device=device,
                dtype=torch.long,
            )

            # Class weights untuk sibling-set ini (ambil dari cw_l2)
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
        # [FIX] Sebelumnya loss L3 HANYA dihitung untuk sampel yang punya
        # label_l3 valid (>=0) — sampel yang memang "berhenti di L2" (label_l3
        # == -1) tidak pernah ikut training di level L3 sama sekali. Akibatnya
        # classifier L3 tidak pernah belajar konsep "tidak usah lanjut ke L3",
        # dan softmax-nya (apalagi yang cuma 1 child seperti Glass/Metal/Paper)
        # selalu mengeluarkan child itu dengan prob ~1.0.
        #
        # Sekarang: SEMUA sampel yang label_l2-nya cocok dengan grup ini diikutkan
        # (tidak lagi memfilter label_l3 >= 0). Target lokalnya:
        #   - jika label_l3 menunjuk salah satu child di grup ini → index child itu
        #   - jika label_l3 == -1 (tidak ada anotasi L3)           → index kelas STOP

        total_l3    = torch.tensor(0.0, device=device)
        num_l3_terms = 0

        for l2_name, l3_children in L3_SIBLINGS.items():
            l2_global_idx = L2_TO_IDX[l2_name]
            logit_key     = f"logits_l3_{l2_name}"
            logits        = model_out[logit_key]  # (B, |l3_children| + 1) — +1 = STOP

            # Mask: semua sampel dengan L2 == l2_global_idx (cukup label L2 valid)
            mask = (label_l2 == l2_global_idx)

            if mask.sum() == 0:
                continue

            l3_global_indices = [L3_TO_IDX[c] for c in l3_children]
            global_to_local   = {g: loc for loc, g in enumerate(l3_global_indices)}
            stop_local_idx    = len(l3_children)   # indeks lokal kelas STOP (terakhir)

            idxs = mask.nonzero(as_tuple=True)[0]
            local_labels = torch.tensor(
                [
                    global_to_local.get(label_l3[i].item(), stop_local_idx)
                    for i in idxs
                ],
                device=device,
                dtype=torch.long,
            )

            # Class weights untuk sibling-set ini (ambil dari cw_l3) + bobot kelas STOP
            cw_l3_sub = None
            if self._cw_l3 is not None:
                cw_real = self._cw_l3[l3_global_indices].to(device)
                # Bobot kelas STOP belum punya statistik tersendiri di dataset,
                # jadi dinetralkan dengan rata-rata bobot child di grup ini.
                stop_w  = cw_real.mean().unsqueeze(0)
                cw_l3_sub = torch.cat([cw_real, stop_w], dim=0)

            loss_l3_sub = self._ce(logits[mask], local_labels, weight=cw_l3_sub)
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
    from hierarchy import num_l1, num_l2, num_l3

    B = 8
    # Simulasi output model — setiap grup L3 sekarang punya +1 kelas STOP
    model_out = {
        "logits_l1"            : torch.randn(B, num_l1()),
        "logits_l2_Organic"    : torch.randn(B, 1),
        "logits_l2_Recyclable" : torch.randn(B, 5),
        "logits_l2_Hazardous"  : torch.randn(B, 2),
        "logits_l3_Plastic"    : torch.randn(B, 3 + 1),   # 3 child + STOP
        "logits_l3_Metal"      : torch.randn(B, 1 + 1),   # 1 child + STOP
        "logits_l3_Paper"      : torch.randn(B, 1 + 1),
        "logits_l3_Cardboard"  : torch.randn(B, 1 + 1),
        "logits_l3_Glass"      : torch.randn(B, 1 + 1),
    }

    # Simulasi label (campuran dengan partial annotation)
    label_l1 = torch.tensor([1, 1, 0, 2, 1, 1, 0, 2])  # Recyclable, Organic, Hazardous
    label_l2 = torch.tensor([0, 1, 0, 0, 2, 3, -1, 1])  # sebagian valid
    # label_l3 == -1 berarti "berhenti di L2" (mis. sample index 4 & 5: L2 valid
    # tapi tidak ada anotasi L3 → harus dilatih sebagai kelas STOP, bukan dibuang)
    label_l3 = torch.tensor([0, -1, -1, -1, -1, -1, -1, -1])

    # Sesuaikan: L2 untuk Recyclable dimulai index 1 di L2_ALL
    from hierarchy import L2_ALL, L3_ALL
    print("L2_ALL:", L2_ALL)
    print("L3_ALL:", L3_ALL)

    loss_fn    = HSCNLoss()
    total, ld  = loss_fn(model_out, label_l1, label_l2, label_l3)

    print(f"\nTotal loss: {total.item():.4f}")
    for k, v in ld.items():
        print(f"  {k}: {v.item():.4f}")