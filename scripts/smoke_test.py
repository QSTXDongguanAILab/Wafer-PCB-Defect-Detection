"""冒烟测试:不依赖权重,验证配置/数据扫描/DB/接口能起来。

    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILED: list[str] = []


def check(name: str, fn) -> None:
    try:
        detail = fn()
        print(f"[OK]   {name}" + (f" — {detail}" if detail else ""))
    except Exception as exc:  # noqa: BLE001
        FAILED.append(name)
        print(f"[FAIL] {name} — {type(exc).__name__}: {exc}")


def main() -> None:
    from app.config import load_settings

    s = load_settings()
    check("配置加载", lambda: f"{s.app_name_en} device={s.device} port={s.port}")

    def db():
        from app.db import count_records, init_db, stats

        init_db(s)
        stats()
        return f"records={count_records()}"

    check("SQLite 初始化", db)

    def pcb_labels():
        from pcb.labels import LABELS, canonical, decide

        assert canonical("基材异物-中等") == "基材异物"
        assert decide("假点", 0.95, 0.90) == "pass"
        assert decide("假点", 0.80, 0.90) == "review"
        assert decide("焊盘氧化", 0.10, 0.90) == "ng"
        return f"{len(LABELS)} 类"

    check("PCB 类别与判级", pcb_labels)

    def pcb_scan():
        from pcb.dataset import collect_pairs, describe, group_split, split_labeled

        root = s.resolve(s.pcb.dataset_root)
        if not root.exists():
            return "原始数据不在位,跳过"
        snap = describe(collect_pairs(root))
        labeled, _ = split_labeled(collect_pairs(root))
        train, val = group_split(labeled, s.pcb.val_ratio, s.pcb.seed)
        overlap = {p.board for p in train} & {p.board for p in val}
        assert not overlap, f"板号泄漏: {overlap}"
        return f"{snap['pairs']} 对 / 已标注 {snap['labeled']} / 板 {snap['boards']} / 切分无泄漏"

    check("PCB 数据扫描与分组切分", pcb_scan)

    def pcb_tensor():
        import numpy as np

        from pcb.loader import INPUT_MODES, build_input, channels_for

        a = np.zeros((8, 8, 3), dtype=np.float32)
        for mode in INPUT_MODES:
            x = build_input(a, a, mode)
            assert x.shape == (channels_for(mode), 8, 8), (mode, x.shape)
        return "四种输入表示通道数正确"

    check("PCB 输入表示", pcb_tensor)

    def pcb_forward():
        import torch

        from pcb.model import build_model

        m = build_model(s.pcb.input_mode)
        from pcb.loader import channels_for

        y = m(torch.zeros(2, channels_for(s.pcb.input_mode), s.pcb.img_size, s.pcb.img_size))
        assert y[0].shape == (2, 2) and y[1].shape[1] == 10
        n = sum(p.numel() for p in m.parameters())
        return f"{s.pcb.input_mode} 前向通,参数 {n/1e6:.2f}M"

    check("PCB 模型前向", pcb_forward)

    def wafer_scan():
        from wafer.voc import collect_voc, describe

        root = s.resolve(s.wafer.dataset_root) / "测试集"
        if not root.exists():
            return "原始数据不在位,跳过"
        snap = describe(collect_voc(root))
        assert not snap["unknown_codes"], f"未登记的代码: {snap['unknown_codes']}"
        return f"{snap['xml_files']} 份 XML / {snap['boxes']} 框 / 硅片 {snap['wafers']}"

    check("硅片 VOC 解析", wafer_scan)

    def api():
        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app) as c:
            h = c.get("/health")
            assert h.status_code == 200, h.text
            t = c.get("/tasks")
            assert t.status_code == 200, t.text
            names = [x["task"] for x in t.json()]
            assert names == ["pcb", "wafer"], names
            r = c.get("/records")
            assert r.status_code == 200, r.text
            st = c.get("/stats")
            assert st.status_code == 200, st.text
        return f"/health /tasks{names} /records /stats 全部 200"

    check("API 起服务", api)

    print()
    if FAILED:
        print(f"失败 {len(FAILED)} 项: {FAILED}")
        raise SystemExit(1)
    print("全部通过")


if __name__ == "__main__":
    main()
