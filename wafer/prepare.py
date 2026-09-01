"""硅片数据准备:VOC → YOLO 目录结构 + data.yaml。

用法:
    python -m wafer.prepare            # 只看数据概况,不落盘
    python -m wafer.prepare --write    # 生成 data/wafer/yolo/

切分按硅片号分组:同一片硅片切出的多张图不能一边训一边验。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import yaml

from app.config import get_settings
from wafer.labels import CODES, NUM_CLASSES, RARE_CODES
from wafer.voc import VocSample, collect_voc, describe, to_yolo_lines


def group_split(
    samples: list[VocSample], val_ratio: float, seed: int
) -> tuple[list[VocSample], list[VocSample]]:
    groups: dict[str, list[VocSample]] = {}
    for s in samples:
        groups.setdefault(s.wafer_id, []).append(s)
    ordered = sorted(
        groups.items(),
        key=lambda kv: hashlib.sha256(f"{seed}:{kv[0]}".encode()).hexdigest(),
    )
    target = int(round(len(samples) * val_ratio))
    val: list[VocSample] = []
    train: list[VocSample] = []
    for _wid, items in ordered:
        (val if len(val) < target else train).extend(items)
    return train, val


def write_yolo(samples: list[VocSample], out_dir: Path, val_ratio: float, seed: int) -> dict:
    usable = [s for s in samples if s.image_path and s.width > 0]
    train, val = group_split(usable, val_ratio, seed)

    for split in ("train", "val"):
        for kind in ("images", "labels"):
            d = out_dir / kind / split
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    empty = {"train": 0, "val": 0}
    for split, items in (("train", train), ("val", val)):
        for s in items:
            # 目录名进文件名:不同目录存在同名硅片图,直接拷会互相覆盖
            stem = f"{s.folder_code}_{s.image_path.stem}"
            shutil.copy2(s.image_path, out_dir / "images" / split / f"{stem}{s.image_path.suffix}")
            lines = to_yolo_lines(s, RARE_CODES)
            (out_dir / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            counts[split] += 1
            if not lines:
                empty[split] += 1

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(out_dir),
                "train": "images/train",
                "val": "images/val",
                "nc": NUM_CLASSES,
                "names": list(CODES),
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return {
        "out_dir": str(out_dir),
        "data_yaml": str(data_yaml),
        "skipped_no_image": len(samples) - len(usable),
        "counts": counts,
        "empty_labels": empty,
        "train_wafers": len({s.wafer_id for s in train}),
        "val_wafers": len({s.wafer_id for s in val}),
        "skipped_codes": sorted(RARE_CODES),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="硅片 VOC → YOLO 数据准备")
    ap.add_argument("--write", action="store_true", help="真正落盘,不加则只打印概况")
    args = ap.parse_args()

    s = get_settings()
    root = s.resolve(s.wafer.dataset_root)
    labeled_root = root / "测试集"
    unlabeled_root = root / "训练集"

    samples = collect_voc(labeled_root if labeled_root.exists() else root)
    snapshot = describe(samples)
    n_unlabeled = (
        len([p for p in unlabeled_root.glob("*.png")]) if unlabeled_root.exists() else 0
    )
    snapshot["unlabeled_images"] = n_unlabeled
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    if n_unlabeled:
        print(
            f"\n注意:{unlabeled_root.name}/ 有 {n_unlabeled} 张图没有 XML 标注,"
            "现阶段无法参与训练 —— 硅片这边同样卡在标注上。"
        )

    if not args.write:
        print("\n未加 --write,只做概况统计。")
        return

    out = write_yolo(samples, s.resolve(s.wafer.yolo_dir), s.wafer.val_ratio, s.wafer.seed)
    print("\n" + json.dumps(out, ensure_ascii=False, indent=2))
    report = s.artifacts_path / "wafer"
    report.mkdir(parents=True, exist_ok=True)
    (report / "prepare_report.json").write_text(
        json.dumps({"snapshot": snapshot, "output": out}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"报告: {report / 'prepare_report.json'}")


if __name__ == "__main__":
    main()
