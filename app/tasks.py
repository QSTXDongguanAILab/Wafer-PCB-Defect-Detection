"""两条业务线的元信息:类别表、接入阶段、数据现状。

前端的 PCB / 光伏入口全部由本模块驱动 —— 类别只在 pcb/labels.py 与 wafer/labels.py
定义一次,页面渲染时拉 /tasks,避免出现「前端写死一套类名、后端另一套」的错位。
"""
from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from pcb import labels as pcb_labels
from wafer import labels as wafer_labels


def _pcb_dataset() -> dict:
    """扫一遍 PCB 原始数据目录,把「还差多少标注」这件事摆在页面上。"""
    s = get_settings()
    root = s.resolve(s.pcb.dataset_root)
    if not root.exists():
        return {"available": False, "root": str(root), "hint": "原始数据目录不存在"}
    try:
        from pcb.dataset import collect_pairs, describe

        snap = describe(collect_pairs(root))
        snap["available"] = True
        snap["root"] = str(root)
        return snap
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "root": str(root), "hint": f"{type(exc).__name__}: {exc}"}


def _wafer_dataset() -> dict:
    s = get_settings()
    root = s.resolve(s.wafer.dataset_root)
    if not root.exists():
        return {"available": False, "root": str(root), "hint": "原始数据目录不存在"}
    try:
        from wafer.voc import collect_voc, describe

        labeled = root / "测试集"
        snap = describe(collect_voc(labeled if labeled.exists() else root))
        unlabeled = root / "训练集"
        snap["unlabeled_images"] = (
            len(list(unlabeled.glob("*.png"))) if unlabeled.exists() else 0
        )
        snap["available"] = True
        snap["root"] = str(root)
        return snap
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "root": str(root), "hint": f"{type(exc).__name__}: {exc}"}


def pcb_task() -> dict:
    s = get_settings()
    weights = s.resolve(s.pcb.model_path)
    ready = weights.exists()
    return {
        "task": "pcb",
        "title": "PCB 假点过滤(终检 AVI 复判)",
        "kind": "classification",
        "ready": ready,
        "model_path": s.pcb.model_path,
        "stage": "基线实验" if not ready else "已有权重",
        "classes": [
            {
                "id": pcb_labels.LABEL_TO_ID[name],
                "name": name,
                "is_ok": pcb_labels.is_ok(name),
                "part": pcb_labels.part_of(name),
                "note": "AVI 误报,判为假点即放行" if pcb_labels.is_ok(name) else None,
            }
            for name in pcb_labels.LABELS
        ],
        "dataset": _pcb_dataset(),
        "notes": [
            "定位:插在 AVI 设备与人工复检之间 —— PCB板 → AVI → 本系统分选 → 人工复检 → OK/NG。",
            "输入是成对 100×100 ROI:<名>.jpg 为待检图,<名>_T.jpg 为同位置标准模板图。",
            f"放行阈值 release_min_prob={s.pcb.release_min_prob}:只有高置信度的「假点」自动放行,"
            "其余一律转人工。漏检是事故,误报只是成本。",
            "汇报口径:NG 召回 ≥99% 时的假点过滤率,不看 accuracy。",
            "训练/验证按板号分组切分,同一块板的 ROI 不跨 train/val,否则指标虚高。",
        ],
    }


def wafer_task() -> dict:
    s = get_settings()
    weights = s.resolve(s.wafer.model_path)
    ready = weights.exists()
    return {
        "task": "wafer",
        "title": "光伏硅片缺陷检测(切片后分选)",
        "kind": "detection",
        "ready": ready,
        "model_path": s.wafer.model_path,
        "stage": "数据准备" if not ready else "已有权重",
        "classes": [
            {
                "id": wafer_labels.CODE_TO_ID[code],
                "name": code,
                "is_ok": False,
                "part": None,
                "note": (
                    "样本个位数,暂不参与训练"
                    if code in wafer_labels.RARE_CODES
                    else wafer_labels.CODE_NAMES.get(code)
                ),
            }
            for code in wafer_labels.CODES
        ],
        "dataset": _wafer_dataset(),
        "notes": [
            "对象是硅片(wafer),不是组件;工序:粘晶 → 切片 → 脱胶 → 插片 → 清洗 → 分选。",
            "标注是 Pascal-VOC XML,一张图常含多个不同代码的框,训练标签以 XML 为准,目录名只当线索。",
            "缺陷代码↔中文名对照表甲方未提供,必须去问数据方;拿到前不要自行猜译。",
            "训练集 510 张 640×640 灰度图没有 XML —— 这边同样卡在标注上。",
        ],
    }


def all_tasks() -> list[dict]:
    return [pcb_task(), wafer_task()]
