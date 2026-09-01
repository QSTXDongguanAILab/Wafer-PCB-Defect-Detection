"""PCB 假点过滤的评价指标。

不要用 accuracy 汇报这个任务:业务是不对称的。
    漏检(真缺陷被判成假点放行)= 不良品流到客户,是事故
    误报(假点被判成缺陷)      = 多一次人工复检,是成本
所以准入门槛是「NG 召回」,收益指标是「假点过滤率」,
汇报口径固定为:NG 召回 ≥ 目标水位时,能自动放行掉多少假点。
"""
from __future__ import annotations

import numpy as np

from pcb.labels import LABELS


def operating_curve(y_ng: np.ndarray, ok_prob: np.ndarray) -> list[dict]:
    """扫放行阈值,给出 (阈值, NG召回, 假点过滤率) 曲线。

    y_ng: 1=真缺陷 0=假点;ok_prob: 模型给「假点」的概率。
    放行规则:ok_prob >= 阈值 才放行。
    """
    y_ng = np.asarray(y_ng).astype(int)
    ok_prob = np.asarray(ok_prob, dtype=float)
    n_ng = int((y_ng == 1).sum())
    n_ok = int((y_ng == 0).sum())
    out: list[dict] = []
    for thr in np.concatenate([np.linspace(0.0, 1.0, 101), np.array([1.0001])]):
        released = ok_prob >= thr
        missed = int((released & (y_ng == 1)).sum())  # 漏检
        filtered = int((released & (y_ng == 0)).sum())  # 成功放行的假点
        out.append(
            {
                "threshold": round(float(thr), 4),
                "ng_recall": 1.0 if n_ng == 0 else round(1.0 - missed / n_ng, 4),
                "filter_rate": 0.0 if n_ok == 0 else round(filtered / n_ok, 4),
                "missed_ng": missed,
                "released_ok": filtered,
            }
        )
    return out


def operating_point(
    y_ng: np.ndarray, ok_prob: np.ndarray, target_recall: float = 0.99
) -> dict:
    """满足 NG 召回 ≥ target_recall 的前提下,过滤率最高的那个阈值。

    曲线里包含「阈值高到什么都不放行」这个退化点,所以召回目标总能达到 ——
    真正的信号是那时的 filter_rate:等于 0 就说明这个模型在要求的召回水位下毫无收益,
    别指望靠调阈值救回来。
    """
    curve = operating_curve(y_ng, ok_prob)
    feasible = [c for c in curve if c["ng_recall"] >= target_recall]
    best = max(feasible, key=lambda c: (c["filter_rate"], -c["threshold"]))
    return {**best, "target_recall": target_recall, "useful": best["filter_rate"] > 0}


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int = len(LABELS)) -> np.ndarray:
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)):
        m[t, p] += 1
    return m


def format_confusion(m: np.ndarray, labels: tuple[str, ...] = LABELS) -> str:
    """终端里能看的混淆矩阵(行=真值,列=预测)。"""
    width = max(len(x) for x in labels) + 1
    head = " " * width + "".join(f"{i:>5d}" for i in range(len(labels)))
    lines = [head]
    for i, name in enumerate(labels):
        lines.append(f"{name:<{width}}" + "".join(f"{v:>5d}" for v in m[i]))
    lines.append("列序号即类别顺序:" + " ".join(f"{i}={n}" for i, n in enumerate(labels)))
    return "\n".join(lines)


def per_class_recall(m: np.ndarray, labels: tuple[str, ...] = LABELS) -> dict[str, float]:
    out: dict[str, float] = {}
    for i, name in enumerate(labels):
        total = int(m[i].sum())
        out[name] = round(float(m[i, i] / total), 4) if total else 0.0
    return out
