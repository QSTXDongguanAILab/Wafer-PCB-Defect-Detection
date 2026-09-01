"""用已训权重给未标注的 PCB 样本做预标注,人工只做纠正。

训练集那 1149 对散图从零手分是整个项目最贵的一块人工。用测试集训出的基线模型
先预分类,按「模型有多确定」排序,人工从最不确定的开始核对 —— 同样工时能覆盖更多样本。

用法:
    python scripts/prelabel_pcb.py                       # 输出 CSV 清单
    python scripts/prelabel_pcb.py --move-to <目录>       # 按预测类别复制进类目录(仍需人工核对)
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import load_settings  # noqa: E402
from pcb.dataset import collect_pairs, split_labeled  # noqa: E402
from pcb.infer import ModelNotReady, get_predictor  # noqa: E402


def main() -> None:
    s = load_settings()
    ap = argparse.ArgumentParser(description="PCB 未标注样本预标注")
    ap.add_argument("--root", default=None, help="默认取 config.yaml 的 pcb.dataset_root")
    ap.add_argument("--out", default="artifacts/pcb/prelabel.csv")
    ap.add_argument("--move-to", default=None, help="按预测类别把成对图复制到该目录下的 <类名>/")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 对(调试用)")
    args = ap.parse_args()

    root = Path(args.root) if args.root else s.resolve(s.pcb.dataset_root)
    _labeled, unlabeled = split_labeled(collect_pairs(root))
    if args.limit:
        unlabeled = unlabeled[: args.limit]
    if not unlabeled:
        print(f"{root} 下没有未标注样本。")
        return

    predictor = get_predictor()
    try:
        predictor.load()
    except ModelNotReady as exc:
        print(f"错误:{exc}")
        raise SystemExit(1) from exc

    out_path = s.resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, pair in enumerate(unlabeled, 1):
        r = predictor.predict_pair(pair)
        rows.append(
            {
                "stem": pair.stem,
                "board": pair.board,
                "avi_item": pair.avi_item,
                "pred_label": r["label"],
                "ok_prob": r["ok_prob"],
                "confidence": r["confidence"],
                "status": r["status"],
                "has_template": int(pair.has_template),
                "test_path": str(pair.test_path),
            }
        )
        if i % 100 == 0:
            print(f"  {i}/{len(unlabeled)}")

    # 按「离 0.5 最近」升序:模型最犹豫的排前面,人工先核对这些收益最高
    rows.sort(key=lambda r: abs(r["ok_prob"] - 0.5))
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n共 {len(rows)} 对,清单: {out_path}")
    print("已按不确定度升序排列,人工从第一行开始核对效率最高。")

    if args.move_to:
        dest_root = Path(args.move_to)
        for r in rows:
            test = Path(r["test_path"])
            tmpl = test.with_name(test.stem + "_T" + test.suffix)
            d = dest_root / r["pred_label"]
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(test, d / test.name)
            if tmpl.exists():
                shutil.copy2(tmpl, d / tmpl.name)
        print(f"已按预测类别复制到 {dest_root}(是预标注,不是真值,必须人工核对)")


if __name__ == "__main__":
    main()
