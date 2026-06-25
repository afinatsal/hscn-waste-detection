"""
model.py
========
Hierarchical Sibling Classification Network (HSCN) untuk klasifikasi sampah.

Perubahan dari versi sebelumnya:
    - L3 classifiers sekarang memiliki +1 kelas untuk "__none__" (index 0).
    - predict() dan predict_with_probs() TIDAK menetapkan pred_l3 jika
      prediksi L3 adalah "__none__" (pred_l3[b] = -1).
    - Ini membuat model bisa "memilih berhenti di L2" saat tidak yakin ada L3.

Struktur Classifier per level (UPDATED):
    L1 : 1 SoftMax classifier   → 3 kelas
    L2 : 3 SoftMax classifiers  → sesuai jumlah sibling per L1
    L3 : 5 SoftMax classifiers  → sesuai jumlah sibling per L2 + 1 (__none__)
         Plastic     → 4 kelas  (__none__, Plastic_Bottle, Plastic_Cup, Plastic_Bag_Film)
         Metal       → 2 kelas  (__none__, Metal_Can)
         Paper       → 2 kelas  (__none__, Paper_Sheet)
         Cardboard   → 2 kelas  (__none__, Cardboard_Box)
         Glass       → 2 kelas  (__none__, Glass_Bottle)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Optional, Tuple

from hierarchy import (
    L1_CLASSES, L2_SIBLINGS, L3_SIBLINGS,
    L2_TO_IDX, L3_TO_IDX,
    L3_NONE_LABEL,
    num_l1, num_l2, num_l3,
    num_l3_with_none,
)


# ─── Classifier Module ────────────────────────────────────────────────────────

class SiblingClassifier(nn.Module):
    """
    Classifier sederhana untuk satu sibling set.

    Architecture:
        Dropout → FC(feat_dim → hidden) → BN → ReLU → Dropout → FC(hidden → num_classes)
    """

    def __init__(self, feat_dim: int, num_classes: int, hidden_dim: int = 512,
                 dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── HSCN Model ───────────────────────────────────────────────────────────────

class HSCN(nn.Module):
    """
    Hierarchical Sibling Classification Network (HSCN).

    Args:
        backbone_name : nama backbone torchvision
        pretrained    : gunakan bobot ImageNet pre-trained
        hidden_dim    : ukuran hidden layer di setiap SiblingClassifier
        dropout       : dropout rate
        freeze_bn     : bekukan BatchNorm backbone
    """

    SUPPORTED_BACKBONES = {
        "resnet50"          : (models.resnet50,         models.ResNet50_Weights.IMAGENET1K_V2,       2048),
        "resnet101"         : (models.resnet101,        models.ResNet101_Weights.IMAGENET1K_V2,      2048),
        "resnet18"          : (models.resnet18,         models.ResNet18_Weights.IMAGENET1K_V1,       512),
        "efficientnet_b3"   : (models.efficientnet_b3,  models.EfficientNet_B3_Weights.IMAGENET1K_V1, 1536),
        "convnext_small"    : (models.convnext_small,   models.ConvNeXt_Small_Weights.IMAGENET1K_V1,  768),
        "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V2, 960),
    }

    def __init__(
        self,
        backbone_name : str   = "resnet50",
        pretrained    : bool  = True,
        hidden_dim    : int   = 512,
        dropout       : float = 0.5,
        freeze_bn     : bool  = False,
    ):
        super().__init__()

        if backbone_name not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"backbone '{backbone_name}' tidak didukung. "
                f"Pilihan: {list(self.SUPPORTED_BACKBONES.keys())}"
            )

        model_fn, weights, feat_dim = self.SUPPORTED_BACKBONES[backbone_name]
        self.feat_dim      = feat_dim
        self.backbone_name = backbone_name

        # ── 1. Backbone ───────────────────────────────────────────────────────
        backbone = model_fn(weights=weights if pretrained else None)
        self.backbone = self._strip_classifier(backbone, backbone_name)

        if freeze_bn:
            self._freeze_batchnorm(self.backbone)

        # ── 2. Global Average Pooling ─────────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── 3. L1 Classifier ──────────────────────────────────────────────────
        self.clf_l1 = SiblingClassifier(feat_dim, num_l1(), hidden_dim, dropout)

        # ── 4. L2 Classifiers ─────────────────────────────────────────────────
        self.clf_l2 = nn.ModuleDict({
            l1: SiblingClassifier(feat_dim, len(children), hidden_dim, dropout)
            for l1, children in L2_SIBLINGS.items()
        })

        # ── 5. L3 Classifiers (TERMASUK kelas __none__ di index 0) ────────────
        # num_l3_with_none(l2) = len(L3_SIBLINGS[l2]) termasuk __none__
        self.clf_l3 = nn.ModuleDict({
            l2: SiblingClassifier(feat_dim, num_l3_with_none(l2), hidden_dim, dropout)
            for l2 in L3_SIBLINGS.keys()
        })

    # ── Backbone helpers ──────────────────────────────────────────────────────

    def _strip_classifier(self, model: nn.Module, name: str) -> nn.Module:
        if name.startswith("resnet"):
            return nn.Sequential(*list(model.children())[:-2])
        elif name.startswith("efficientnet"):
            model.classifier = nn.Identity()
            return model.features
        elif name.startswith("convnext"):
            model.classifier = nn.Identity()
            return model.features
        elif name.startswith("mobilenet"):
            model.classifier = nn.Identity()
            return model.features
        else:
            raise ValueError(f"Tidak tahu cara strip backbone: {name}")

    def _freeze_batchnorm(self, module: nn.Module):
        for m in module.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    # ── Forward ───────────────────────────────────────────────────────────────

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.backbone(x)
        feat     = self.gap(feat_map)
        feat     = feat.flatten(1)
        return feat

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass penuh.

        Returns:
            dict dengan kunci:
                'logits_l1'         : (B, 3)
                'logits_l2_<L1>'    : (B, |L2_siblings|)
                'logits_l3_<L2>'    : (B, |L3_siblings_incl_none|)
        """
        feat = self.extract_features(x)
        out  = {}

        out["logits_l1"] = self.clf_l1(feat)

        for l1_name, clf in self.clf_l2.items():
            out[f"logits_l2_{l1_name}"] = clf(feat)

        for l2_name, clf in self.clf_l3.items():
            out[f"logits_l3_{l2_name}"] = clf(feat)

        return out

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Prediksi hierarki penuh.

        Logika L3:
            - Jika argmax L3 == 0 (__none__) → pred_l3[b] = -1
              (artinya objek ini tidak memiliki sub-tipe L3)
            - Jika argmax L3 > 0 → pred_l3[b] = global index L3 di L3_ALL

        Returns:
            pred_l1 : (B,)  — index prediksi L1
            pred_l2 : (B,)  — index prediksi L2 dalam L2_ALL
            pred_l3 : (B,)  — index prediksi L3 dalam L3_ALL, -1 jika __none__
            prob_l1 : (B, 3)
        """
        out    = self.forward(x)
        B      = x.size(0)
        device = x.device

        prob_l1 = F.softmax(out["logits_l1"], dim=-1)
        pred_l1 = prob_l1.argmax(dim=-1)

        pred_l2 = torch.full((B,), -1, dtype=torch.long, device=device)
        pred_l3 = torch.full((B,), -1, dtype=torch.long, device=device)

        for b in range(B):
            l1_idx  = pred_l1[b].item()
            l1_name = L1_CLASSES[l1_idx]

            # ── L2 ────────────────────────────────────────────────────────────
            l2_logits = out[f"logits_l2_{l1_name}"][b]
            l2_local  = l2_logits.argmax().item()
            l2_name   = L2_SIBLINGS[l1_name][l2_local]
            pred_l2[b] = L2_TO_IDX[l2_name]

            # ── L3 ────────────────────────────────────────────────────────────
            if l2_name in L3_SIBLINGS:
                l3_logits = out[f"logits_l3_{l2_name}"][b]
                l3_local  = l3_logits.argmax().item()

                if l3_local == 0:
                    # __none__: model memilih berhenti di L2
                    pred_l3[b] = -1
                else:
                    # Kelas L3 nyata: konversi local → global
                    l3_name    = L3_SIBLINGS[l2_name][l3_local]
                    pred_l3[b] = L3_TO_IDX[l3_name]

        return {
            "pred_l1": pred_l1,
            "pred_l2": pred_l2,
            "pred_l3": pred_l3,
            "prob_l1": prob_l1,
        }

    @torch.no_grad()
    def predict_with_probs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Prediksi hierarki penuh + joint probability per level.

        Joint probability:
            p_l1(c)  = softmax_l1(c)
            p_l2(c)  = softmax_l2_local(c) * p_l1(parent)
            p_l3(c)  = softmax_l3_local(c) * p_l2(parent)
                       → __none__ tidak dimasukkan ke prob_l3_joint
                       → probability L3 nyata sudah memperhitungkan prob __none__

        Returns:
            pred_l1       : (B,)
            pred_l2       : (B,)
            pred_l3       : (B,)  — -1 jika __none__ diprediksi
            prob_l1       : (B, num_l1)
            prob_l2_joint : (B, num_l2)
            prob_l3_joint : (B, num_l3)  — hanya kelas nyata, excl. __none__
        """
        out    = self.forward(x)
        B      = x.size(0)
        device = x.device

        # ── L1 ────────────────────────────────────────────────────────────────
        prob_l1 = F.softmax(out["logits_l1"], dim=-1)
        pred_l1 = prob_l1.argmax(dim=-1)

        # ── Joint probability L2 ──────────────────────────────────────────────
        num_l2_total  = len(L2_TO_IDX)
        prob_l2_joint = torch.zeros(B, num_l2_total, device=device)

        for l1_idx, l1_name in enumerate(L1_CLASSES):
            p_l1_this = prob_l1[:, l1_idx]
            l2_logits = out[f"logits_l2_{l1_name}"]
            l2_local  = F.softmax(l2_logits, dim=-1)
            for local_i, l2_name in enumerate(L2_SIBLINGS[l1_name]):
                g = L2_TO_IDX[l2_name]
                prob_l2_joint[:, g] = l2_local[:, local_i] * p_l1_this

        pred_l2 = torch.full((B,), -1, dtype=torch.long, device=device)
        for b in range(B):
            l1_name   = L1_CLASSES[pred_l1[b].item()]
            l2_logits = out[f"logits_l2_{l1_name}"][b]
            l2_local  = l2_logits.argmax().item()
            l2_name   = L2_SIBLINGS[l1_name][l2_local]
            pred_l2[b] = L2_TO_IDX[l2_name]

        # ── Joint probability L3 ──────────────────────────────────────────────
        # Hanya kelas L3 nyata (bukan __none__) yang masuk ke prob_l3_joint.
        # Karena index 0 = __none__, kelas nyata dimulai dari index 1 (local).
        num_l3_total  = len(L3_TO_IDX)
        prob_l3_joint = torch.zeros(B, num_l3_total, device=device)

        for l2_name, l3_children in L3_SIBLINGS.items():
            l2_global_idx = L2_TO_IDX[l2_name]
            p_l2_this     = prob_l2_joint[:, l2_global_idx]
            l3_logits     = out[f"logits_l3_{l2_name}"]
            l3_local_probs = F.softmax(l3_logits, dim=-1)  # (B, n_incl_none)

            # Skip index 0 (__none__), mulai dari 1
            for local_i, cls_name in enumerate(l3_children):
                if cls_name == L3_NONE_LABEL:
                    continue
                global_i = L3_TO_IDX[cls_name]
                prob_l3_joint[:, global_i] = l3_local_probs[:, local_i] * p_l2_this

        pred_l3 = torch.full((B,), -1, dtype=torch.long, device=device)
        for b in range(B):
            l2_idx = pred_l2[b].item()
            if l2_idx < 0:
                continue
            # Cari nama L2
            l2_name = next((n for n, i in L2_TO_IDX.items() if i == l2_idx), None)
            if l2_name and l2_name in L3_SIBLINGS:
                l3_logits = out[f"logits_l3_{l2_name}"][b]
                l3_local  = l3_logits.argmax().item()
                if l3_local == 0:
                    pred_l3[b] = -1   # __none__
                else:
                    l3_name    = L3_SIBLINGS[l2_name][l3_local]
                    pred_l3[b] = L3_TO_IDX[l3_name]

        return {
            "pred_l1"      : pred_l1,
            "pred_l2"      : pred_l2,
            "pred_l3"      : pred_l3,
            "prob_l1"      : prob_l1,
            "prob_l2_joint": prob_l2_joint,
            "prob_l3_joint": prob_l3_joint,
        }

    def get_trainable_params(self, backbone_lr_scale: float = 0.1):
        backbone_params = list(self.backbone.parameters())
        head_params = (
            list(self.clf_l1.parameters()) +
            [p for clf in self.clf_l2.values() for p in clf.parameters()] +
            [p for clf in self.clf_l3.values() for p in clf.parameters()] +
            list(self.gap.parameters())
        )
        return [
            {"params": backbone_params, "lr_scale": backbone_lr_scale},
            {"params": head_params,     "lr_scale": 1.0},
        ]

    def count_parameters(self) -> Dict[str, int]:
        def count(module):
            return sum(p.numel() for p in module.parameters())

        result = {
            "backbone": count(self.backbone),
            "clf_l1"  : count(self.clf_l1),
        }
        for k, v in self.clf_l2.items():
            result[f"clf_l2_{k}"] = count(v)
        for k, v in self.clf_l3.items():
            result[f"clf_l3_{k}"] = count(v)
        result["total"]     = count(self)
        result["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return result


# ─── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    model = HSCN(backbone_name="resnet50", pretrained=False)

    dummy = torch.randn(4, 3, 224, 224)
    out   = model(dummy)

    print("=== HSCN Output Keys ===")
    for k, v in out.items():
        print(f"  {k}: {v.shape}")

    pred = model.predict(dummy)
    print("\n=== Predictions ===")
    for k, v in pred.items():
        print(f"  {k}: {v}")
    print("(pred_l3 == -1 berarti model memilih berhenti di L2 / __none__)")

    # Verifikasi jumlah kelas L3
    print("\n=== L3 Classifier Output Sizes ===")
    from hierarchy import L3_SIBLINGS, L3_NONE_LABEL
    for l2, children in L3_SIBLINGS.items():
        print(f"  clf_l3[{l2}]: {len(children)} kelas {children}")

    params = model.count_parameters()
    print("\n=== Parameter Count ===")
    for k, v in params.items():
        print(f"  {k:<25}: {v:>12,}")