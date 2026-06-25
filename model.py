"""
model.py
========
Hierarchical Sibling Classification Network (HSCN) untuk klasifikasi sampah.

Mengimplementasikan arsitektur HSCN sesuai paper:
    "Hierarchical Multi-Label Object Detection Framework for Remote Sensing Images"
    Shin et al., Remote Sensing 2020.

Prinsip utama HSCN:
    1. Satu backbone bersama (shared feature extractor).
    2. Classifier TERPISAH untuk SETIAP sibling-set di setiap level.
       → Menghindari kompetisi SoftMax lintas kelas yang tidak se-level.
    3. Setiap classifier hanya dilatih dengan sampel yang memiliki anotasi
       pada level tersebut (partial annotation handling).
    4. Prediksi final = argmax dari output classifier yang relevan per level.

Struktur Classifier per level:
    L1 : 1 SoftMax classifier   → 3 kelas  (Organic, Recyclable, Hazardous)
    L2 : 3 SoftMax classifiers  → masing-masing untuk sibling set tiap L1
         Organic     → 1 kelas  (Food_Waste)
         Recyclable  → 5 kelas  (Plastic, Metal, Paper, Cardboard, Glass)
         Hazardous   → 2 kelas  (Battery, E_Waste)
    L3 : 5 SoftMax classifiers  → masing-masing untuk sibling set tiap L2-Recyclable
         Plastic     → 3 kelas  (Plastic_Bottle, Plastic_Cup, Plastic_Bag_Film)
         Metal       → 1 kelas  (Metal_Can)
         Paper       → 1 kelas  (Paper_Sheet)
         Cardboard   → 1 kelas  (Cardboard_Box)
         Glass       → 1 kelas  (Glass_Bottle)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Optional, Tuple

from hierarchy import (
    L1_CLASSES, L2_SIBLINGS, L3_SIBLINGS,
    L2_TO_IDX, L3_TO_IDX,
    num_l1, num_l2, num_l3,
)


# ─── Classifier Module ────────────────────────────────────────────────────────

class SiblingClassifier(nn.Module):
    """
    Classifier sederhana untuk satu sibling set.
    Sesuai paper: "fully dense" (fully connected layers).

    Architecture:
        Dropout → FC(feat_dim → hidden) → ReLU → Dropout → FC(hidden → num_classes)
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
        """x: (B, feat_dim) → logits: (B, num_classes)"""
        return self.net(x)


# ─── HSCN Model ───────────────────────────────────────────────────────────────

class HSCN(nn.Module):
    """
    Hierarchical Sibling Classification Network (HSCN).

    Args:
        backbone_name : nama backbone torchvision ('resnet50', 'resnet101',
                        'efficientnet_b3', 'convnext_small', dsb.)
        pretrained    : gunakan bobot ImageNet pre-trained
        hidden_dim    : ukuran hidden layer di setiap SiblingClassifier
        dropout       : dropout rate
        freeze_bn     : bekukan BatchNorm backbone (berguna untuk dataset kecil)
    """

    SUPPORTED_BACKBONES = {
        "resnet50"        : (models.resnet50,         models.ResNet50_Weights.IMAGENET1K_V2,   2048),
        "resnet101"       : (models.resnet101,        models.ResNet101_Weights.IMAGENET1K_V2,  2048),
        "resnet18"        : (models.resnet18,         models.ResNet18_Weights.IMAGENET1K_V1,   512),
        "efficientnet_b3" : (models.efficientnet_b3,  models.EfficientNet_B3_Weights.IMAGENET1K_V1, 1536),
        "convnext_small"  : (models.convnext_small,   models.ConvNeXt_Small_Weights.IMAGENET1K_V1, 768),
        "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V2, 960),
    }

    def __init__(
        self,
        backbone_name : str  = "resnet50",
        pretrained    : bool = True,
        hidden_dim    : int  = 512,
        dropout       : float = 0.5,
        freeze_bn     : bool = False,
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

        # ── 1. Build backbone (hapus head classifier bawaan) ──────────────────
        if pretrained:
            backbone = model_fn(weights=weights)
        else:
            backbone = model_fn(weights=None)

        self.backbone = self._strip_classifier(backbone, backbone_name)

        if freeze_bn:
            self._freeze_batchnorm(self.backbone)

        # ── 2. Global Average Pooling + Flatten ───────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── 3. L1 Classifier (satu sibling set: root) ─────────────────────────
        # Sibling set: {Organic, Recyclable, Hazardous}
        self.clf_l1 = SiblingClassifier(feat_dim, num_l1(), hidden_dim, dropout)

        # ── 4. L2 Classifiers (satu per L1 sibling set) ───────────────────────
        # Organic → [Food_Waste]                    (1 kelas)
        # Recyclable → [Plastic, Metal, Paper, Cardboard, Glass]  (5 kelas)
        # Hazardous → [Battery, E_Waste]            (2 kelas)
        self.clf_l2 = nn.ModuleDict({
            l1: SiblingClassifier(feat_dim, len(children), hidden_dim, dropout)
            for l1, children in L2_SIBLINGS.items()
        })

        # ── 5. L3 Classifiers (satu per L2 sibling set di Recyclable) ─────────
        # Plastic   → [Plastic_Bottle, Plastic_Cup, Plastic_Bag_Film, STOP]  (3+1 kelas)
        # Metal     → [Metal_Can, STOP]       (1+1 kelas)
        # Paper     → [Paper_Sheet, STOP]     (1+1 kelas)
        # Cardboard → [Cardboard_Box, STOP]   (1+1 kelas)
        # Glass     → [Glass_Bottle, STOP]    (1+1 kelas)
        #
        # [FIX] Setiap sibling-set L3 mendapat SATU kelas tambahan "STOP"
        # (indeks lokal terakhir = len(children)). Kelas ini mewakili
        # "anotasi berhenti di L2, tidak ada L3". Tanpa kelas ini, softmax
        # dipaksa selalu memilih salah satu child L3 walaupun sampel memang
        # tidak punya label L3 (mis. foto kaca tanpa bentuk botol yang jelas).
        # Ini terutama krusial untuk Metal/Paper/Cardboard/Glass yang aslinya
        # cuma punya 1 child — tanpa STOP, softmax 1-kelas itu SELALU bernilai
        # 1.0 sehingga model 100% "wajib" memprediksi L3.
        self.clf_l3 = nn.ModuleDict({
            l2: SiblingClassifier(feat_dim, len(children) + 1, hidden_dim, dropout)
            for l2, children in L3_SIBLINGS.items()
        })

    # ── Backbone helper methods ────────────────────────────────────────────────

    def _strip_classifier(self, model: nn.Module, name: str) -> nn.Module:
        """Hapus head classifier dari backbone, sisakan feature extractor saja."""
        if name.startswith("resnet"):
            # ResNet: hapus avgpool + fc
            return nn.Sequential(*list(model.children())[:-2])
        elif name.startswith("efficientnet"):
            # EfficientNet: hapus classifier
            model.classifier = nn.Identity()
            return model.features  # kembalikan hanya bagian features
        elif name.startswith("convnext"):
            # ConvNeXt: hapus head
            model.classifier = nn.Identity()
            return model.features
        elif name.startswith("mobilenet"):
            # MobileNetV3: hapus classifier
            model.classifier = nn.Identity()
            return model.features
        else:
            raise ValueError(f"Tidak tahu cara strip backbone: {name}")

    def _freeze_batchnorm(self, module: nn.Module):
        """Bekukan semua BatchNorm layers (set ke eval mode permanen)."""
        for m in module.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    # ── Forward ───────────────────────────────────────────────────────────────

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ekstrak feature vector bersama dari backbone.

        Returns:
            feat: (B, feat_dim)
        """
        feat_map = self.backbone(x)          # (B, C, H', W')
        feat     = self.gap(feat_map)        # (B, C, 1, 1)
        feat     = feat.flatten(1)           # (B, C)
        return feat

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass penuh.

        Args:
            x : (B, 3, H, W) — batch gambar

        Returns:
            dict dengan kunci:
                'logits_l1'         : (B, 3)          — logits L1
                'logits_l2_<L1>'    : (B, |siblings|) — logits L2 per L1 parent
                'logits_l3_<L2>'    : (B, |siblings|) — logits L3 per L2 parent
        """
        feat = self.extract_features(x)   # (B, feat_dim)

        out = {}

        # L1
        out["logits_l1"] = self.clf_l1(feat)

        # L2 — satu classifier per L1 sibling-set
        for l1_name, clf in self.clf_l2.items():
            out[f"logits_l2_{l1_name}"] = clf(feat)

        # L3 — satu classifier per L2 sibling-set (Recyclable subtypes)
        for l2_name, clf in self.clf_l3.items():
            out[f"logits_l3_{l2_name}"] = clf(feat)

        return out

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Prediksi hierarki penuh menggunakan conditional probability.

        Strategi (mengikuti YOLO9000 / paper HSCN):
            P(L2_i | L1_j) × P(L1_j)  → argmax per sibling-set L2
            P(L3_k | L2_i) × P(L2_i | L1_j) × P(L1_j) → argmax per sibling-set L3

        Returns:
            {
              'pred_l1'     : (B,) — index prediksi L1
              'pred_l2'     : (B,) — index prediksi L2 dalam L2_ALL (-1 jika tidak relevan)
              'pred_l3'     : (B,) — index prediksi L3 dalam L3_ALL (-1 jika tidak ada L3)
              'prob_l1'     : (B, 3)
            }
        """
        from hierarchy import (
            L1_CLASSES, L2_SIBLINGS, L3_SIBLINGS,
            L2_TO_IDX, L3_TO_IDX,
        )

        out  = self.forward(x)
        B    = x.size(0)

        prob_l1  = F.softmax(out["logits_l1"], dim=-1)  # (B, 3)
        pred_l1  = prob_l1.argmax(dim=-1)               # (B,)

        # L2: pilih argmax dari classifier L2 yang relevan per sampel
        pred_l2  = torch.full((B,), -1, dtype=torch.long, device=x.device)
        pred_l3  = torch.full((B,), -1, dtype=torch.long, device=x.device)

        for b_idx in range(B):
            l1_idx  = pred_l1[b_idx].item()
            l1_name = L1_CLASSES[l1_idx]

            # ── L2 ────────────────────────────────────────────────────────────
            l2_logits = out[f"logits_l2_{l1_name}"][b_idx]  # (|L2_siblings|,)
            l2_local  = l2_logits.argmax().item()
            l2_name   = L2_SIBLINGS[l1_name][l2_local]
            pred_l2[b_idx] = L2_TO_IDX[l2_name]

            # ── L3 (hanya jika L2 memiliki L3 children) ───────────────────────
            if l2_name in L3_SIBLINGS:
                l3_logits  = out[f"logits_l3_{l2_name}"][b_idx]   # (|children|+1,) — +1 = STOP
                n_children = len(L3_SIBLINGS[l2_name])
                l3_local   = l3_logits.argmax().item()
                # [FIX] indeks lokal terakhir (== n_children) adalah kelas STOP.
                # Jika model memilih STOP, biarkan pred_l3 tetap -1 (berhenti di L2).
                if l3_local < n_children:
                    l3_name = L3_SIBLINGS[l2_name][l3_local]
                    pred_l3[b_idx] = L3_TO_IDX[l3_name]

        return {
            "pred_l1": pred_l1,
            "pred_l2": pred_l2,
            "pred_l3": pred_l3,
            "prob_l1": prob_l1,
        }

    @torch.no_grad()
    def predict_with_probs(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Prediksi hierarki penuh + joint probability per level sesuai Eq. 5 paper.

        Joint probability (hierarkis):
            p_l1(c)     = softmax_l1(c)                    — prob L1
            p_l2(c)     = softmax_l2_local(c) * p_l1(par)  — prob joint L2
            p_l3(c)     = softmax_l3_local(c) * p_l2(par)  — prob joint L3

        Joint prob digunakan untuk menghitung mAP (sesuai paper).

        Returns:
            pred_l1       : (B,)  — argmax L1
            pred_l2       : (B,)  — argmax L2 dalam L2_ALL
            pred_l3       : (B,)  — argmax L3 dalam L3_ALL (-1 jika model memilih STOP di L2)
            prob_l1       : (B, num_l1)
            prob_l2_joint : (B, num_l2)  — joint prob semua kelas L2
            prob_l3_joint : (B, num_l3)  — joint prob semua kelas L3 (tidak termasuk massa STOP)
            prob_stop_l3  : (B,)  — probabilitas model memilih berhenti di L2 (tidak lanjut ke L3)
        """
        from hierarchy import (
            L1_CLASSES, L2_SIBLINGS, L3_SIBLINGS,
            L2_TO_IDX, L3_TO_IDX,
        )

        out    = self.forward(x)
        B      = x.size(0)
        device = x.device

        # ── L1 ────────────────────────────────────────────────────────────────
        prob_l1 = F.softmax(out["logits_l1"], dim=-1)   # (B, 3)
        pred_l1 = prob_l1.argmax(dim=-1)                # (B,)

        # ── Joint probability matrix L2 — shape (B, num_l2) ──────────────────
        num_l2_total = len(L2_TO_IDX)
        prob_l2_joint = torch.zeros(B, num_l2_total, device=device)

        for l1_idx, l1_name in enumerate(L1_CLASSES):
            p_l1_this = prob_l1[:, l1_idx]              # (B,) prob L1 = l1_name
            l2_logits  = out[f"logits_l2_{l1_name}"]    # (B, |siblings|)
            l2_local   = F.softmax(l2_logits, dim=-1)   # (B, |siblings|)
            for local_idx, l2_name in enumerate(L2_SIBLINGS[l1_name]):
                global_idx = L2_TO_IDX[l2_name]
                # joint: P(L2=c) = P(L2=c | L1=l1) * P(L1=l1)
                prob_l2_joint[:, global_idx] = l2_local[:, local_idx] * p_l1_this

        pred_l2 = torch.full((B,), -1, dtype=torch.long, device=device)
        for b in range(B):
            l1_name   = L1_CLASSES[pred_l1[b].item()]
            l2_logits = out[f"logits_l2_{l1_name}"][b]
            l2_local  = l2_logits.argmax().item()
            l2_name   = L2_SIBLINGS[l1_name][l2_local]
            pred_l2[b] = L2_TO_IDX[l2_name]

        # ── Joint probability matrix L3 — shape (B, num_l3) ──────────────────
        # [FIX] Setiap grup L3 kini punya 1 logit ekstra "STOP" di indeks
        # terakhir. Softmax dihitung atas (children + STOP), tapi hanya
        # probabilitas milik child NYATA yang dimasukkan ke prob_l3_joint
        # — probabilitas yang "lari" ke STOP otomatis tidak dihitung sebagai
        # milik salah satu child (sesuai semantik mAP per kelas L3 nyata).
        num_l3_total  = len(L3_TO_IDX)
        prob_l3_joint = torch.zeros(B, num_l3_total, device=device)

        for l2_name, l3_children in L3_SIBLINGS.items():
            l2_global_idx = L2_TO_IDX[l2_name]
            p_l2_this     = prob_l2_joint[:, l2_global_idx]    # (B,)
            l3_logits     = out[f"logits_l3_{l2_name}"]        # (B, |children|+1)
            l3_full       = F.softmax(l3_logits, dim=-1)       # (B, |children|+1), termasuk prob STOP
            n_children    = len(l3_children)
            l3_local      = l3_full[:, :n_children]            # (B, |children|) — exclude STOP
            for local_idx, l3_name in enumerate(l3_children):
                global_idx = L3_TO_IDX[l3_name]
                prob_l3_joint[:, global_idx] = l3_local[:, local_idx] * p_l2_this

        pred_l3      = torch.full((B,), -1, dtype=torch.long, device=device)
        prob_stop_l3 = torch.zeros(B, device=device)  # prob model memilih "berhenti di L2"

        for b in range(B):
            l2_idx = pred_l2[b].item()
            if l2_idx < 0:
                continue
            # cari nama l2 dari global idx
            l2_name = None
            for name, idx in L2_TO_IDX.items():
                if idx == l2_idx:
                    l2_name = name
                    break
            if l2_name and l2_name in L3_SIBLINGS:
                l3_logits  = out[f"logits_l3_{l2_name}"][b]    # (|children|+1,)
                n_children = len(L3_SIBLINGS[l2_name])
                full_probs = F.softmax(l3_logits, dim=-1)
                prob_stop_l3[b] = full_probs[n_children]        # prob kelas STOP

                l3_local = l3_logits.argmax().item()
                # [FIX] jika argmax jatuh ke indeks STOP, jangan paksa pilih
                # salah satu child — biarkan pred_l3 tetap -1 (berhenti di L2).
                if l3_local < n_children:
                    l3_name = L3_SIBLINGS[l2_name][l3_local]
                    pred_l3[b] = L3_TO_IDX[l3_name]

        return {
            "pred_l1"      : pred_l1,
            "pred_l2"      : pred_l2,
            "pred_l3"      : pred_l3,
            "prob_l1"      : prob_l1,
            "prob_l2_joint": prob_l2_joint,
            "prob_l3_joint": prob_l3_joint,
            "prob_stop_l3" : prob_stop_l3,
        }

    def get_trainable_params(self, backbone_lr_scale: float = 0.1):
        """
        Kembalikan parameter groups untuk optimizer dengan learning rate
        yang lebih kecil untuk backbone (fine-tuning) dan lr penuh untuk
        classifier heads.

        Args:
            backbone_lr_scale: faktor pengali lr untuk backbone
        """
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
        """Hitung jumlah parameter per komponen."""
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
        result["total"] = count(self)
        result["trainable"] = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
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

    params = model.count_parameters()
    print("\n=== Parameter Count ===")
    for k, v in params.items():
        print(f"  {k:<25}: {v:>12,}")