"""数据体检:两条业务线的数据现状一次看完,不训练不写盘。

    python scripts/data_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import load_settings  # noqa: E402


def main() -> None:
    s = load_settings()
    report: dict = {}

    root = s.resolve(s.pcb.dataset_root)
    print(f"=== PCB  {root} ===")
    if root.exists():
        from pcb.dataset import collect_pairs, describe, group_split, split_labeled

        pairs = collect_pairs(root)
        snap = describe(pairs)
        labeled, _ = split_labeled(pairs)
        train, val = group_split(labeled, s.pcb.val_ratio, s.pcb.seed)
        snap["group_split"] = {
            "train_pairs": len(train),
            "val_pairs": len(val),
            "train_boards": len({p.board for p in train}),
            "val_boards": len({p.board for p in val}),
            "board_overlap": len({p.board for p in train} & {p.board for p in val}),
        }
        report["pcb"] = snap
        print(json.dumps(snap, ensure_ascii=False, indent=2))
    else:
        print("数据目录不存在")

    wroot = s.resolve(s.wafer.dataset_root)
    print(f"\n=== 光伏硅片  {wroot} ===")
    if wroot.exists():
        from wafer.voc import collect_voc, describe as wdescribe

        labeled_dir = wroot / "测试集"
        snap = wdescribe(collect_voc(labeled_dir if labeled_dir.exists() else wroot))
        unl = wroot / "训练集"
        snap["unlabeled_images"] = len(list(unl.glob("*.png"))) if unl.exists() else 0
        report["wafer"] = snap
        print(json.dumps(snap, ensure_ascii=False, indent=2))
    else:
        print("数据目录不存在")

    out = s.artifacts_path / "data_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告: {out}")


if __name__ == "__main__":
    main()
