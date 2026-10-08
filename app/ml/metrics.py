from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)), 1)
    return cm


def accuracy_from_cm(cm: np.ndarray) -> float:
    total = int(cm.sum())
    return float(np.trace(cm) / total) if total else 0.0


def per_class_prf(cm: np.ndarray):
    """Returns (precision, recall, f1, support) arrays; index i == class i."""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)
    precision = np.divide(tp, np.maximum(tp + fp, 1.0))
    recall = np.divide(tp, np.maximum(tp + fn, 1.0))
    f1 = np.divide(2 * precision * recall, np.maximum(precision + recall, 1e-12))
    return precision, recall, f1, support.astype(np.int64)


def _binary_auc(pos_mask: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based (Mann-Whitney U) AUC with tie-corrected average ranks."""
    n = len(scores)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks_sorted = np.arange(1, n + 1, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks_sorted[i:j] = (i + 1 + j) / 2.0  # average 1-based rank of the tied block
        i = j
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    n_pos = int(pos_mask.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos_mask].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def roc_auc_ovr(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> np.ndarray:
    aucs = np.full(num_classes, np.nan)
    y_true = np.asarray(y_true, dtype=np.int64)
    for c in range(num_classes):
        aucs[c] = _binary_auc(y_true == c, probs[:, c])
    return aucs


def latency_stats_ms(times_ms: list[float]) -> dict:
    if not times_ms:
        return {"count": 0}
    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean_ms": round(float(arr.mean()), 3),
        "min_ms": round(float(arr.min()), 3),
        "max_ms": round(float(arr.max()), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
    }


def classification_report(y_true, y_pred, class_names: list[str], probs: np.ndarray | None = None) -> dict:
    num_classes = len(class_names)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = confusion_matrix(y_true, y_pred, num_classes)
    precision, recall, f1, support = per_class_prf(cm)
    total = int(cm.sum())

    w_precision = float(np.dot(precision, support) / max(total, 1))
    w_recall = float(np.dot(recall, support) / max(total, 1))
    w_f1 = float(np.dot(f1, support) / max(total, 1))

    aucs = roc_auc_ovr(y_true, probs, num_classes) if probs is not None else None
    per_class: dict[str, dict] = {}
    for i, name in enumerate(class_names):
        entry = {
            "precision": round(float(precision[i]), 6),
            "recall": round(float(recall[i]), 6),
            "f1": round(float(f1[i]), 6),
            "support": int(support[i]),
        }
        if aucs is not None:
            entry["roc_auc"] = None if np.isnan(aucs[i]) else round(float(aucs[i]), 6)
        per_class[name] = entry

    report = {
        "num_rows": total,
        "accuracy": round(accuracy_from_cm(cm), 6),
        "macro_avg": {
            "precision": round(float(precision.mean()), 6),
            "recall": round(float(recall.mean()), 6),
            "f1": round(float(f1.mean()), 6),
        },
        "weighted_avg": {
            "precision": round(w_precision, 6),
            "recall": round(w_recall, 6),
            "f1": round(w_f1, 6),
        },
        "per_class": per_class,
        "confusion_matrix": {"labels": class_names, "matrix": cm.tolist()},
    }
    if aucs is not None:
        valid = aucs[~np.isnan(aucs)]
        report["roc_auc_macro"] = round(float(valid.mean()), 6) if valid.size else None
    return report