"""
metrics.py
==========
Evaluasi performa HSCN pada setiap level hirarki.

Metrik yang dihitung:
    - Accuracy per level (L1, L2, L3)
    - Accuracy hierarkis (benar di SEMUA level yang tersedia)
    - Per-class accuracy per level
    - Confusion matrix (opsional)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch

from hierarchy import (
    L1_CLASSES, L2_ALL, L3_ALL,
    L2_SIBLINGS, L3_SIBLINGS,
    L2_TO_IDX, L3_TO_IDX,
)


class HSCNMetrics:
    """
    Akumulator metrik untuk evaluasi HSCN.

    Usage:
        metrics = HSCNMetrics()
        for batch in dataloader:
            ...
            preds = model.predict(imgs)
            metrics.update(preds, label_l1, label_l2, label_l3)
        results = metrics.compute()
        metrics.reset()
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._pred_l1  = []
        self._true_l1  = []
        self._pred_l2  = []
        self._true_l2  = []
        self._pred_l3  = []
        self._true_l3  = []

    def update(
        self,
        preds    : Dict[str, torch.Tensor],
        true_l1  : torch.Tensor,
        true_l2  : torch.Tensor,
        true_l3  : torch.Tensor,
    ):
        """
        Akumulasi prediksi dan ground truth satu batch.

        Args:
            preds   : output dari model.predict()
            true_l1 : (B,) label L1
            true_l2 : (B,) label L2  (-1 = tidak ada)
            true_l3 : (B,) label L3  (-1 = tidak ada)
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
        true_l3 = np.array(self._true_l3)

        results = {}

        # ── L1 Accuracy ───────────────────────────────────────────────────────
        results["acc_l1"] = float((pred_l1 == true_l1).mean())

        # Per-class L1 accuracy
        for i, name in enumerate(L1_CLASSES):
            mask = (true_l1 == i)
            if mask.sum() > 0:
                results[f"acc_l1_{name}"] = float((pred_l1[mask] == true_l1[mask]).mean())

        # ── L2 Accuracy (hanya sampel dengan L2 valid) ────────────────────────
        mask_l2v = (true_l2 >= 0)
        if mask_l2v.sum() > 0:
            results["acc_l2"] = float(
                (pred_l2[mask_l2v] == true_l2[mask_l2v]).mean()
            )
            # Per-class L2 accuracy
            for i, name in enumerate(L2_ALL):
                mask_cls = mask_l2v & (true_l2 == i)
                if mask_cls.sum() > 0:
                    results[f"acc_l2_{name}"] = float(
                        (pred_l2[mask_cls] == true_l2[mask_cls]).mean()
                    )
        else:
            results["acc_l2"] = 0.0

        # ── L3 Accuracy (hanya sampel dengan L3 valid) ────────────────────────
        mask_l3v = (true_l3 >= 0)
        if mask_l3v.sum() > 0:
            results["acc_l3"] = float(
                (pred_l3[mask_l3v] == true_l3[mask_l3v]).mean()
            )
            # Per-class L3 accuracy
            for i, name in enumerate(L3_ALL):
                mask_cls = mask_l3v & (true_l3 == i)
                if mask_cls.sum() > 0:
                    results[f"acc_l3_{name}"] = float(
                        (pred_l3[mask_cls] == true_l3[mask_cls]).mean()
                    )
        else:
            results["acc_l3"] = 0.0

        # ── Hierarchical Accuracy ─────────────────────────────────────────────
        # Benar di L1 DAN L2 (untuk sampel yang punya L2)
        if mask_l2v.sum() > 0:
            correct_l1_and_l2 = (
                (pred_l1[mask_l2v] == true_l1[mask_l2v]) &
                (pred_l2[mask_l2v] == true_l2[mask_l2v])
            )
            results["acc_hier_l1l2"] = float(correct_l1_and_l2.mean())

        # Benar di L1, L2, DAN L3 (untuk sampel yang punya L3)
        if mask_l3v.sum() > 0:
            correct_all = (
                (pred_l1[mask_l3v] == true_l1[mask_l3v]) &
                (pred_l2[mask_l3v] == true_l2[mask_l3v]) &
                (pred_l3[mask_l3v] == true_l3[mask_l3v])
            )
            results["acc_hier_all"] = float(correct_all.mean())

        # ── Summary metric (weighted average) ─────────────────────────────────
        n_l2_valid = int(mask_l2v.sum())
        n_l3_valid = int(mask_l3v.sum())
        n_total    = len(pred_l1)

        # Mean accuracy: bobot berdasarkan jumlah sampel yang relevan
        accs_weighted = [results["acc_l1"]]
        if n_l2_valid > 0: accs_weighted.append(results["acc_l2"])
        if n_l3_valid > 0: accs_weighted.append(results["acc_l3"])
        results["acc_mean"] = float(np.mean(accs_weighted))

        # Sample counts (berguna untuk logging)
        results["n_total"]    = n_total
        results["n_l2_valid"] = n_l2_valid
        results["n_l3_valid"] = n_l3_valid

        return results

    def format_report(self, results: Optional[Dict] = None) -> str:
        """Format metrik menjadi teks laporan yang mudah dibaca."""
        if results is None:
            results = self.compute()

        lines = ["=" * 55, "HSCN Evaluation Report", "=" * 55]

        # Top-level summary
        lines.append(f"Samples total   : {results.get('n_total', 0)}")
        lines.append(f"Samples w/ L2   : {results.get('n_l2_valid', 0)}")
        lines.append(f"Samples w/ L3   : {results.get('n_l3_valid', 0)}")
        lines.append("")

        lines.append("── Level Accuracy ──────────────────────────────────")
        lines.append(f"  L1 Accuracy    : {results.get('acc_l1', 0):.4f}")
        lines.append(f"  L2 Accuracy    : {results.get('acc_l2', 0):.4f}")
        lines.append(f"  L3 Accuracy    : {results.get('acc_l3', 0):.4f}")
        lines.append(f"  Mean Accuracy  : {results.get('acc_mean', 0):.4f}")
        lines.append("")
        lines.append("── Hierarchical Accuracy ───────────────────────────")
        lines.append(f"  L1+L2          : {results.get('acc_hier_l1l2', 0):.4f}")
        lines.append(f"  L1+L2+L3       : {results.get('acc_hier_all', 0):.4f}")
        lines.append("")
        lines.append("── Per-Class L1 ────────────────────────────────────")
        for name in L1_CLASSES:
            k = f"acc_l1_{name}"
            if k in results:
                lines.append(f"  {name:<15}: {results[k]:.4f}")

        lines.append("")
        lines.append("── Per-Class L2 ────────────────────────────────────")
        for name in L2_ALL:
            k = f"acc_l2_{name}"
            if k in results:
                lines.append(f"  {name:<20}: {results[k]:.4f}")

        lines.append("")
        lines.append("── Per-Class L3 ────────────────────────────────────")
        for name in L3_ALL:
            k = f"acc_l3_{name}"
            if k in results:
                lines.append(f"  {name:<25}: {results[k]:.4f}")

        lines.append("=" * 55)
        return "\n".join(lines)
