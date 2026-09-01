"""硅片缺陷检测训练(YOLO)。

先跑 `python -m wafer.prepare --write` 生成 data/wafer/yolo/data.yaml。

用法:
    python -m wafer.train --epochs 100
    python -m wafer.train --epochs 2 --imgsz 320   # 冒烟

现状提醒:标注数据只有 537 张(测试集),BYW/KYW/XHB 三类样本个位数已被剔除,
训练集那 510 张没有 XML。这个量级只够跑通流程和拿基线,不够交付。
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from app.config import get_settings


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description="硅片缺陷检测训练")
    ap.add_argument("--weights", default="yolo11n.pt", help="预训练权重")
    ap.add_argument("--epochs", type=int, default=s.wafer.epochs)
    ap.add_argument("--imgsz", type=int, default=s.wafer.img_size)
    ap.add_argument("--batch", type=int, default=s.wafer.batch_size)
    ap.add_argument("--device", default=s.device)
    args = ap.parse_args()

    data_yaml = s.resolve(s.wafer.yolo_dir) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"缺少 {data_yaml},先运行:python -m wafer.prepare --write"
        )

    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=s.wafer.seed,
        project=str(s.project_root / "runs" / "wafer"),
        name="detect",
        exist_ok=True,
    )

    best = s.project_root / "runs" / "wafer" / "detect" / "weights" / "best.pt"
    dest = s.resolve(s.wafer.model_path)
    if best.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dest)
        print(f"[wafer] 权重已复制到 {dest}")
    else:
        print(f"[wafer] 警告:未找到 {best}")


if __name__ == "__main__":
    main()
