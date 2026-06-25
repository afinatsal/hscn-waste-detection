"""
metrics.py
==========
Evaluasi performa HSCN pada setiap level hirarki.

Metrik utama (sesuai paper Shin et al., 2020):
    - mAP per level (L1, L2, L3):
        AP dihitung per kelas (one-vs-rest) dari confidence/probability score,
        lalu di-rata-rata jadi mAP — sesuai "level-wise mAP at L1/L2/L3"
        Table 2 & 3 paper.
        Probability hierarkis mengikuti Eq. 5 paper:
            p_HSCN(j,L) = p(j,L) x prod_{l=1}^{L-1} p(par(j),l)
        (softmax lokal dikali semua leluhur — disuplai oleh predict_with_probs)

    - Confusion matrix:
        Ditampilkan hanya di format_report() (akhir evaluasi per model).
        Baris = true, kolom = predicted (argmax).
"""

import numpy as np
from typing import Dict, List, Optional

import torch

from hierarchy import (
    L1_CLASSES, L2_ALL, L3_ALL,
    L2_SIBLINGS, L3_SIBLINGS,
    L2_TO_IDX, L3_TO_IDX,
)


# ─── AP helper ────────────────────────────────────────────────────────────────

def _ap_from_scores(y_true, y_score):
    if y_true.sum() == 0:
        return float("nan")
    order     = np.argsort(-y_score)
    y_true    = y_true[order]
    tp_cum    = np.cumsum(y_true)
    fp_cum    = np.cumsum(1 - y_true)
    total_p   = y_true.sum()
    precision = tp_cum / (tp_cum + fp_cum + 1e-12)
    recall    = tp_cum / (total_p + 1e-12)
    precision = np.concatenate([[1.0], precision])
    recall    = np.concatenate([[0.0], recall])
    return float(np.sum(np.diff(recall) * precision[1:]))


def _map_from_scores(true_labels, score_matrix, num_classes):
    ap_per_class = []
    valid_aps    = []
    for c in range(num_classes):
        ap = _ap_from_scores((true_labels == c).astype(np.float32), score_matrix[:, c])
        ap_per_class.append(ap)
        if not np.isnan(ap):
            valid_aps.append(ap)
    return {"mAP": float(np.mean(valid_aps)) if valid_aps else 0.0,
            "AP_per_class": ap_per_class}


# ─── Confusion matrix helper ──────────────────────────────────────────────────

def _confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def _format_confusion_matrix(cm, class_names):
    col_w  = max(max(len(c) for c in class_names), 6) + 2
    header = " " * col_w + "".join(f"{c:>{col_w}}" for c in class_names)
    lines  = [header]
    for i, row_name in enumerate(class_names):
        row = f"{row_name:>{col_w}}" + "".join(
            f"{cm[i, j]:>{col_w}}" for j in range(len(class_names))
        )
        lines.append(row)
    return "\n".join(lines)


# ─── HSCNMetrics ──────────────────────────────────────────────────────────────

class HSCNMetrics:
    """
    Akumulator metrik HSCN.

    Jika update() menerima output predict_with_probs() (ada kunci prob_l1,
    prob_l2_joint, prob_l3_joint) → hitung mAP + acc.
    Jika hanya output predict() lama → hitung acc saja.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._pred_l1 = []; self._true_l1 = []
        self._pred_l2 = []; self._true_l2 = []
        self._pred_l3 = []; self._true_l3 = []
        self._prob_l1       = []
        self._prob_l2_joint = []
        self._prob_l3_joint = []
        self._has_probs = False

    def update(self, preds, true_l1, true_l2, true_l3):
        self._pred_l1.extend(preds["pred_l1"].cpu().numpy().tolist())
        self._true_l1.extend(true_l1.cpu().numpy().tolist())
        self._pred_l2.extend(preds["pred_l2"].cpu().numpy().tolist())
        self._true_l2.extend(true_l2.cpu().numpy().tolist())
        self._pred_l3.extend(preds["pred_l3"].cpu().numpy().tolist())
        self._true_l3.extend(true_l3.cpu().numpy().tolist())

        if "prob_l1" in preds:
            self._prob_l1.append(preds["prob_l1"].cpu().numpy())
            self._has_probs = True
        if "prob_l2_joint" in preds:
            self._prob_l2_joint.append(preds["prob_l2_joint"].cpu().numpy())
        if "prob_l3_joint" in preds:
            self._prob_l3_joint.append(preds["prob_l3_joint"].cpu().numpy())

    def compute(self):
        pred_l1 = np.array(self._pred_l1); true_l1 = np.array(self._true_l1)
        pred_l2 = np.array(self._pred_l2); true_l2 = np.array(self._true_l2)
        pred_l3 = np.array(self._pred_l3); true_l3 = np.array(self._true_l3)
        mask_l2v = (true_l2 >= 0)
        mask_l3v = (true_l3 >= 0)
        results  = {}

        # ── Accuracy ──────────────────────────────────────────────────────────
        results["acc_l1"] = float((pred_l1 == true_l1).mean())
        for i, name in enumerate(L1_CLASSES):
            m = (true_l1 == i)
            if m.sum() > 0:
                results[f"acc_l1_{name}"] = float((pred_l1[m] == true_l1[m]).mean())

        if mask_l2v.sum() > 0:
            results["acc_l2"] = float((pred_l2[mask_l2v] == true_l2[mask_l2v]).mean())
            for i, name in enumerate(L2_ALL):
                m = mask_l2v & (true_l2 == i)
                if m.sum() > 0:
                    results[f"acc_l2_{name}"] = float((pred_l2[m] == true_l2[m]).mean())
        else:
            results["acc_l2"] = 0.0

        if mask_l3v.sum() > 0:
            results["acc_l3"] = float((pred_l3[mask_l3v] == true_l3[mask_l3v]).mean())
            for i, name in enumerate(L3_ALL):
                m = mask_l3v & (true_l3 == i)
                if m.sum() > 0:
                    results[f"acc_l3_{name}"] = float((pred_l3[m] == true_l3[m]).mean())
        else:
            results["acc_l3"] = 0.0

        # ── [BARU] Stop-decision accuracy (mekanisme STOP di L3) ────────────────
        # Mengukur apakah model bisa membedakan "harus lanjut ke L3" vs
        # "harus berhenti di L2", terbatas pada sampel yang L2-nya termasuk
        # grup yang MEMILIKI children L3 (Plastic/Metal/Paper/Cardboard/Glass).
        # Ini metrik tambahan — tidak menggantikan acc_l3/mAP_l3 di atas.
        l3_capable_idx = np.array([L2_TO_IDX[l2] for l2 in L3_SIBLINGS.keys()])
        mask_l3_capable = mask_l2v & np.isin(true_l2, l3_capable_idx)
        if mask_l3_capable.sum() > 0:
            true_stop = (true_l3[mask_l3_capable] < 0)
            pred_stop = (pred_l3[mask_l3_capable] < 0)
            results["acc_l3_stop_decision"] = float((true_stop == pred_stop).mean())
            if true_stop.sum() > 0:
                # Dari sampel yang seharusnya berhenti di L2, berapa % terdeteksi benar
                results["recall_l3_stop"] = float(pred_stop[true_stop].mean())
            if (~true_stop).sum() > 0:
                # Dari sampel yang seharusnya lanjut ke L3, berapa % yang benar terdeteksi "lanjut"
                results["recall_l3_continue"] = float((~pred_stop[~true_stop]).mean())

        if mask_l2v.sum() > 0:
            results["acc_hier_l1l2"] = float((
                (pred_l1[mask_l2v] == true_l1[mask_l2v]) &
                (pred_l2[mask_l2v] == true_l2[mask_l2v])
            ).mean())
        if mask_l3v.sum() > 0:
            results["acc_hier_all"] = float((
                (pred_l1[mask_l3v] == true_l1[mask_l3v]) &
                (pred_l2[mask_l3v] == true_l2[mask_l3v]) &
                (pred_l3[mask_l3v] == true_l3[mask_l3v])
            ).mean())

        # ── mAP per level (paper metric) ──────────────────────────────────────
        if self._has_probs:
            if self._prob_l1:
                pm = np.concatenate(self._prob_l1, axis=0)
                r  = _map_from_scores(true_l1, pm, len(L1_CLASSES))
                results["mAP_l1"] = r["mAP"]
                for i, name in enumerate(L1_CLASSES):
                    ap = r["AP_per_class"][i]
                    if not np.isnan(ap):
                        results[f"AP_l1_{name}"] = float(ap)

            if self._prob_l2_joint and mask_l2v.sum() > 0:
                pm = np.concatenate(self._prob_l2_joint, axis=0)
                r  = _map_from_scores(true_l2[mask_l2v], pm[mask_l2v], len(L2_ALL))
                results["mAP_l2"] = r["mAP"]
                for i, name in enumerate(L2_ALL):
                    ap = r["AP_per_class"][i]
                    if not np.isnan(ap):
                        results[f"AP_l2_{name}"] = float(ap)

            if self._prob_l3_joint and mask_l3v.sum() > 0:
                pm = np.concatenate(self._prob_l3_joint, axis=0)
                r  = _map_from_scores(true_l3[mask_l3v], pm[mask_l3v], len(L3_ALL))
                results["mAP_l3"] = r["mAP"]
                for i, name in enumerate(L3_ALL):
                    ap = r["AP_per_class"][i]
                    if not np.isnan(ap):
                        results[f"AP_l3_{name}"] = float(ap)

            level_maps = [results[k] for k in ("mAP_l1", "mAP_l2", "mAP_l3") if k in results]
            if level_maps:
                results["mAP_mean"] = float(np.mean(level_maps))

        results["n_total"]    = int(len(pred_l1))
        results["n_l2_valid"] = int(mask_l2v.sum())
        results["n_l3_valid"] = int(mask_l3v.sum())
        return results

    def compute_confusion_matrices(self):
        pred_l1 = np.array(self._pred_l1); true_l1 = np.array(self._true_l1)
        pred_l2 = np.array(self._pred_l2); true_l2 = np.array(self._true_l2)
        pred_l3 = np.array(self._pred_l3); true_l3 = np.array(self._true_l3)
        cms = {"cm_l1": _confusion_matrix(true_l1, pred_l1, len(L1_CLASSES))}
        mask_l2v = (true_l2 >= 0)
        if mask_l2v.sum() > 0:
            cms["cm_l2"] = _confusion_matrix(true_l2[mask_l2v], pred_l2[mask_l2v], len(L2_ALL))
        mask_l3v = (true_l3 >= 0)
        if mask_l3v.sum() > 0:
            cms["cm_l3"] = _confusion_matrix(true_l3[mask_l3v], pred_l3[mask_l3v], len(L3_ALL))
        return cms

    def format_report(self, results=None):
        """Laporan lengkap: mAP, acc, dan confusion matrix."""
        if results is None:
            results = self.compute()

        lines = ["=" * 60, "HSCN Evaluation Report", "=" * 60]
        lines += [
            f"Samples total   : {results.get('n_total', 0)}",
            f"Samples w/ L2   : {results.get('n_l2_valid', 0)}",
            f"Samples w/ L3   : {results.get('n_l3_valid', 0)}", "",
        ]

        # mAP — metrik utama sesuai paper
        if "mAP_l1" in results:
            lines += [
                "── mAP per Level  [paper metric] ────────────────────────",
                f"  mAP L1         : {results.get('mAP_l1', 0):.4f}",
                f"  mAP L2         : {results.get('mAP_l2', 0):.4f}",
                f"  mAP L3         : {results.get('mAP_l3', 0):.4f}",
                f"  mAP Mean       : {results.get('mAP_mean', 0):.4f}", "",
                "── AP per Class (L1) ────────────────────────────────────",
            ]
            for name in L1_CLASSES:
                k = f"AP_l1_{name}"
                if k in results:
                    lines.append(f"  {name:<15}: {results[k]:.4f}")
            lines += ["", "── AP per Class (L2) ────────────────────────────────────"]
            for name in L2_ALL:
                k = f"AP_l2_{name}"
                if k in results:
                    lines.append(f"  {name:<20}: {results[k]:.4f}")
            lines += ["", "── AP per Class (L3) ────────────────────────────────────"]
            for name in L3_ALL:
                k = f"AP_l3_{name}"
                if k in results:
                    lines.append(f"  {name:<25}: {results[k]:.4f}")
            lines.append("")

        # Accuracy
        lines += [
            "── Accuracy per Level ───────────────────────────────────",
            f"  L1 Accuracy    : {results.get('acc_l1', 0):.4f}",
            f"  L2 Accuracy    : {results.get('acc_l2', 0):.4f}",
            f"  L3 Accuracy    : {results.get('acc_l3', 0):.4f}", "",
            "── Hierarchical Accuracy ────────────────────────────────",
            f"  L1+L2          : {results.get('acc_hier_l1l2', 0):.4f}",
            f"  L1+L2+L3       : {results.get('acc_hier_all', 0):.4f}", "",
        ]

        # [BARU] Stop-decision (mekanisme STOP di L3)
        if "acc_l3_stop_decision" in results:
            lines += [
                "── L3 Stop-Decision  [BARU — kapan model berhenti di L2] ─",
                f"  Stop-decision acc : {results.get('acc_l3_stop_decision', 0):.4f}",
                f"  Recall (stop)     : {results.get('recall_l3_stop', float('nan')):.4f}",
                f"  Recall (lanjut)   : {results.get('recall_l3_continue', float('nan')):.4f}",
                "",
            ]

        # Confusion matrices
        cms = self.compute_confusion_matrices()
        lines += ["── Confusion Matrix L1 ──────────────────────────────────",
                  _format_confusion_matrix(cms["cm_l1"], L1_CLASSES), ""]
        if "cm_l2" in cms:
            lines += ["── Confusion Matrix L2 ──────────────────────────────────",
                      _format_confusion_matrix(cms["cm_l2"], L2_ALL), ""]
        if "cm_l3" in cms:
            lines += ["── Confusion Matrix L3 ──────────────────────────────────",
                      _format_confusion_matrix(cms["cm_l3"], L3_ALL), ""]

        lines.append("=" * 60)
        return "\n".join(lines)