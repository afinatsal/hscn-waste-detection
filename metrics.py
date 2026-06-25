"""
metrics.py
==========
Evaluasi performa HSCN pada setiap level hirarki.

Perubahan dari versi sebelumnya:
    - Akurasi L3 dihitung dengan dua cara:
        1. acc_l3_specific : akurasi hanya pada sampel yang punya L3 spesifik
           (ground truth != -1, bukan __none__)
        2. acc_l3_stop     : akurasi "keputusan berhenti" — apakah model
           benar memprediksi -1 (berhenti di L2) untuk sampel yang memang
           tidak punya L3 spesifik
        3. acc_l3          : akurasi gabungan (semua sampel yang punya L2
           dengan L3 siblings)

Encoding pred_l3 dari model.predict():
    -1   → model memilih __none__ (berhenti di L2)
    >= 0 → model memilih kelas L3 spesifik (index di L3_ALL)

Encoding true_l3 dari dataset:
    -1              → L2 tidak punya L3 siblings (Food_Waste, Battery, E_Waste)
    -(l2_idx + 10)  → __none__ sentinel (L2 punya L3 tapi tidak ada sub-tipe)
    >= 0            → index global L3 di L3_ALL
"""

import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

import torch

from hierarchy import (
    L1_CLASSES, L2_ALL, L3_ALL,
    L2_SIBLINGS, L3_SIBLINGS,
    L2_TO_IDX, L3_TO_IDX,
    L3_NONE_LABEL,
)


def decode_true_l3(true_l3_val: int, true_l2_val: int) -> Optional[int]:
    """
    Decode label L3 dari dataset ke nilai yang bisa dibandingkan dengan pred_l3.

    Returns:
        None  → sampel ini tidak relevan untuk evaluasi L3 (L2 tidak punya L3)
        -1    → ground truth adalah __none__ (berhenti di L2)
        >= 0  → index global L3 spesifik
    """
    if true_l3_val == -1:
        # L2 tidak punya L3 siblings → tidak relevan
        return None
    elif true_l3_val < -1:
        # Sentinel -(l2_idx + 10) → ground truth adalah __none__
        return -1
    else:
        # Kelas L3 spesifik
        return true_l3_val


class HSCNMetrics:
    """
    Akumulator metrik untuk evaluasi HSCN.

    Usage:
        metrics = HSCNMetrics()
        for batch in dataloader:
            preds = model.predict(imgs)
            metrics.update(preds, label_l1, label_l2, label_l3)
        results = metrics.compute()
        metrics.reset()
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._pred_l1 = []
        self._true_l1 = []
        self._pred_l2 = []
        self._true_l2 = []
        self._pred_l3 = []
        self._true_l3 = []   # raw dari dataset (bisa berisi sentinel)

    def update(
        self,
        preds   : Dict[str, torch.Tensor],
        true_l1 : torch.Tensor,
        true_l2 : torch.Tensor,
        true_l3 : torch.Tensor,
    ):
        """
        Akumulasi prediksi dan ground truth satu batch.

        Args:
            preds   : output dari model.predict()
            true_l1 : (B,) label L1
            true_l2 : (B,) label L2  (-1 = tidak ada)
            true_l3 : (B,) label L3  (-1 = tidak ada, -(l2+10) = __none__, >= 0 = valid)
        """
        self._pred_l1.extend(preds["pred_l1"].cpu().numpy().tolist())
        self._true_l1.extend(true_l1.cpu().numpy().tolist())
        self._pred_l2.extend(preds["pred_l2"].cpu().numpy().tolist())
        self._true_l2.extend(true_l2.cpu().numpy().tolist())
        self._pred_l3.extend(preds["pred_l3"].cpu().numpy().tolist())
        self._true_l3.extend(true_l3.cpu().numpy().tolist())

    def compute(self) -> Dict[str, float]:
        """
        Hitung semua metrik dari akumulasi data.

        Returns:
            dict berisi semua metrik skalar.
        """
        pred_l1 = np.array(self._pred_l1)
        true_l1 = np.array(self._true_l1)
        pred_l2 = np.array(self._pred_l2)
        true_l2 = np.array(self._true_l2)
        pred_l3 = np.array(self._pred_l3)
        true_l3_raw = np.array(self._true_l3)

        # Decode true_l3: sentinel → -1 (none), -1 → None (tidak relevan)
        true_l3_decoded = np.array([
            decode_true_l3(int(t3), int(t2))
            for t3, t2 in zip(true_l3_raw, true_l2)
        ], dtype=object)

        results = {}
        N = len(pred_l1)

        # ── L1 Accuracy ───────────────────────────────────────────────────────
        results["acc_l1"] = float((pred_l1 == true_l1).mean())

        for i, name in enumerate(L1_CLASSES):
            mask = (true_l1 == i)
            if mask.sum() > 0:
                results[f"acc_l1_{name}"] = float((pred_l1[mask] == true_l1[mask]).mean())

        # ── L2 Accuracy ───────────────────────────────────────────────────────
        mask_l2v = (true_l2 >= 0)
        if mask_l2v.sum() > 0:
            results["acc_l2"] = float((pred_l2[mask_l2v] == true_l2[mask_l2v]).mean())
            for i, name in enumerate(L2_ALL):
                mask_cls = mask_l2v & (true_l2 == i)
                if mask_cls.sum() > 0:
                    results[f"acc_l2_{name}"] = float(
                        (pred_l2[mask_cls] == true_l2[mask_cls]).mean()
                    )
        else:
            results["acc_l2"] = 0.0

        # ── L3 Accuracy ───────────────────────────────────────────────────────
        # Sampel relevan: true_l3_decoded != None
        # (artinya L2-nya punya L3 siblings, baik yang punya L3 spesifik maupun __none__)
        relevant_mask = np.array([v is not None for v in true_l3_decoded])

        if relevant_mask.sum() > 0:
            pred_l3_rel   = pred_l3[relevant_mask]
            true_l3_rel   = np.array([v for v in true_l3_decoded if v is not None], dtype=int)

            # acc_l3: akurasi keseluruhan (termasuk sampel __none__)
            results["acc_l3"] = float((pred_l3_rel == true_l3_rel).mean())

            # acc_l3_specific: hanya sampel dengan L3 spesifik (ground truth >= 0)
            specific_mask = (true_l3_rel >= 0)
            if specific_mask.sum() > 0:
                results["acc_l3_specific"] = float(
                    (pred_l3_rel[specific_mask] == true_l3_rel[specific_mask]).mean()
                )
                # Per-class L3 accuracy
                for i, name in enumerate(L3_ALL):
                    mask_cls = specific_mask & (true_l3_rel == i)
                    if mask_cls.sum() > 0:
                        results[f"acc_l3_{name}"] = float(
                            (pred_l3_rel[mask_cls] == true_l3_rel[mask_cls]).mean()
                        )
            else:
                results["acc_l3_specific"] = 0.0

            # acc_l3_stop: akurasi "keputusan berhenti di L2"
            # Sampel yang ground truth-nya __none__ (-1 setelah decode)
            none_mask = (true_l3_rel == -1)
            if none_mask.sum() > 0:
                # Model benar jika pred_l3 == -1 (juga memilih berhenti)
                results["acc_l3_stop"] = float(
                    (pred_l3_rel[none_mask] == -1).mean()
                )
            else:
                results["acc_l3_stop"] = float("nan")

        else:
            results["acc_l3"]          = 0.0
            results["acc_l3_specific"] = 0.0
            results["acc_l3_stop"]     = float("nan")

        # ── Hierarchical Accuracy ─────────────────────────────────────────────
        mask_l2v_idx = np.where(mask_l2v)[0]
        if len(mask_l2v_idx) > 0:
            correct_l1l2 = (
                (pred_l1[mask_l2v] == true_l1[mask_l2v]) &
                (pred_l2[mask_l2v] == true_l2[mask_l2v])
            )
            results["acc_hier_l1l2"] = float(correct_l1l2.mean())

        rel_idx = np.where(relevant_mask)[0]
        if len(rel_idx) > 0:
            true_l3_rel_arr = np.array(
                [v for v in true_l3_decoded if v is not None], dtype=int
            )
            correct_all = (
                (pred_l1[rel_idx] == true_l1[rel_idx]) &
                (pred_l2[rel_idx] == true_l2[rel_idx]) &
                (pred_l3[rel_idx] == true_l3_rel_arr)
            )
            results["acc_hier_all"] = float(correct_all.mean())

        # ── Summary ───────────────────────────────────────────────────────────
        accs_weighted = [results["acc_l1"]]
        if mask_l2v.sum() > 0:
            accs_weighted.append(results["acc_l2"])
        if relevant_mask.sum() > 0:
            accs_weighted.append(results["acc_l3"])
        results["acc_mean"] = float(np.mean(accs_weighted))

        results["n_total"]         = N
        results["n_l2_valid"]      = int(mask_l2v.sum())
        results["n_l3_relevant"]   = int(relevant_mask.sum())
        results["n_l3_specific"]   = int(sum(
            1 for v in true_l3_decoded if v is not None and v >= 0
        ))
        results["n_l3_none"]       = int(sum(
            1 for v in true_l3_decoded if v is not None and v == -1
        ))

        return results

    def format_report(self, results: Optional[Dict] = None) -> str:
        if results is None:
            results = self.compute()

        lines = ["=" * 60, "HSCN Evaluation Report", "=" * 60]

        lines.append(f"Samples total          : {results.get('n_total', 0)}")
        lines.append(f"Samples w/ L2          : {results.get('n_l2_valid', 0)}")
        lines.append(f"Samples w/ L3 (rel)    : {results.get('n_l3_relevant', 0)}")
        lines.append(f"  - L3 spesifik        : {results.get('n_l3_specific', 0)}")
        lines.append(f"  - L3 none (stop L2)  : {results.get('n_l3_none', 0)}")
        lines.append("")

        lines.append("── Level Accuracy ──────────────────────────────────────")
        lines.append(f"  L1 Accuracy          : {results.get('acc_l1', 0):.4f}")
        lines.append(f"  L2 Accuracy          : {results.get('acc_l2', 0):.4f}")
        lines.append(f"  L3 Accuracy (all)    : {results.get('acc_l3', 0):.4f}")
        lines.append(f"  L3 Acc (specific)    : {results.get('acc_l3_specific', 0):.4f}")
        stop_acc = results.get('acc_l3_stop', float('nan'))
        stop_str = f"{stop_acc:.4f}" if not (isinstance(stop_acc, float) and np.isnan(stop_acc)) else "N/A"
        lines.append(f"  L3 Acc (stop@L2)     : {stop_str}")
        lines.append(f"  Mean Accuracy        : {results.get('acc_mean', 0):.4f}")
        lines.append("")

        lines.append("── Hierarchical Accuracy ───────────────────────────────")
        lines.append(f"  L1+L2                : {results.get('acc_hier_l1l2', 0):.4f}")
        lines.append(f"  L1+L2+L3             : {results.get('acc_hier_all', 0):.4f}")
        lines.append("")

        lines.append("── Per-Class L1 ────────────────────────────────────────")
        for name in L1_CLASSES:
            k = f"acc_l1_{name}"
            if k in results:
                lines.append(f"  {name:<15}: {results[k]:.4f}")

        lines.append("")
        lines.append("── Per-Class L2 ────────────────────────────────────────")
        for name in L2_ALL:
            k = f"acc_l2_{name}"
            if k in results:
                lines.append(f"  {name:<20}: {results[k]:.4f}")

        lines.append("")
        lines.append("── Per-Class L3 (specific only) ────────────────────────")
        for name in L3_ALL:
            k = f"acc_l3_{name}"
            if k in results:
                lines.append(f"  {name:<25}: {results[k]:.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)