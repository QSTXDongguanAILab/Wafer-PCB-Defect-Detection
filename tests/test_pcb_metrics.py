"""假点过滤指标的单元测试:重点是「NG 召回优先」这条业务口径。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcb.metrics import confusion_matrix, operating_point, per_class_recall


def test_perfect_separation():
    # 假点(0)的 ok_prob 高,NG(1)的低 —— 完全可分
    y_ng = np.array([0, 0, 0, 1, 1, 1])
    ok_prob = np.array([0.99, 0.98, 0.97, 0.02, 0.03, 0.01])
    op = operating_point(y_ng, ok_prob, target_recall=1.0)
    assert op["useful"]
    assert op["ng_recall"] == 1.0
    assert op["filter_rate"] == 1.0
    assert op["missed_ng"] == 0


def test_recall_constraint_costs_filter_rate():
    # 有一个 NG 的 ok_prob 高达 0.9,要保住 100% 召回就必须把阈值顶上去,
    # 代价是 0.9 以下的假点全都放不掉
    y_ng = np.array([0, 0, 0, 0, 1, 1])
    ok_prob = np.array([0.95, 0.85, 0.80, 0.60, 0.90, 0.05])
    op = operating_point(y_ng, ok_prob, target_recall=1.0)
    assert op["missed_ng"] == 0
    assert op["threshold"] > 0.90
    assert op["filter_rate"] == 0.25  # 只有 0.95 那个假点能放行


def test_worthless_model_reports_zero_gain():
    """NG 的假点概率比所有假点都高时,保住召回就一个也放行不了 —— 收益必须报 0,不能粉饰。"""
    y_ng = np.array([0, 0, 1])
    ok_prob = np.array([0.4, 0.5, 0.9])
    op = operating_point(y_ng, ok_prob, target_recall=1.0)
    assert op["missed_ng"] == 0
    assert op["filter_rate"] == 0.0
    assert op["useful"] is False


def test_confusion_and_recall():
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 0])
    m = confusion_matrix(y_true, y_pred, n=10)
    assert m[0, 0] == 1 and m[0, 1] == 1
    assert m[1, 1] == 2
    r = per_class_recall(m)
    assert r["假点"] == 0.5
    assert r["基材划痕"] == 1.0
    assert r["基材异物"] == 0.0
