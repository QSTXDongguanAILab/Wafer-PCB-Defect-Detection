"""Pascal-VOC XML 解析与 YOLO 标签转换(硅片缺陷检测)。

VOC 里的坐标是像素绝对值 xmin/ymin/xmax/ymax;YOLO 要的是归一化 cx,cy,w,h。
实测标注存在越界(xmin=1、ymax=640 贴边)和 truncated=1 的框,统一裁到图内。
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from wafer.labels import CODE_TO_ID

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass(frozen=True)
class Box:
    code: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    truncated: bool = False
    difficult: bool = False


@dataclass(frozen=True)
class VocSample:
    xml_path: Path
    image_path: Path | None
    width: int
    height: int
    folder_code: str  # 所在目录名,即甲方标的「主缺陷代码」
    boxes: tuple[Box, ...]

    @property
    def wafer_id(self) -> str:
        """同一片硅片会切出多张图,切分时按它分组,防泄漏。

        文件名形如 34090654224052_3.png → 硅片号 34090654224052。
        """
        stem = self.xml_path.stem
        return stem.rsplit("_", 1)[0] if "_" in stem else stem


def _find_image(xml_path: Path, filename: str | None) -> Path | None:
    if filename:
        cand = xml_path.parent / filename
        if cand.exists():
            return cand
    for suffix in IMAGE_SUFFIXES:
        cand = xml_path.with_suffix(suffix)
        if cand.exists():
            return cand
    return None


def parse_voc(xml_path: Path) -> VocSample:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(float(size.findtext("width", "0"))) if size is not None else 0
    height = int(float(size.findtext("height", "0"))) if size is not None else 0

    boxes: list[Box] = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        boxes.append(
            Box(
                code=(obj.findtext("name") or "").strip().upper(),
                xmin=float(bb.findtext("xmin", "0")),
                ymin=float(bb.findtext("ymin", "0")),
                xmax=float(bb.findtext("xmax", "0")),
                ymax=float(bb.findtext("ymax", "0")),
                truncated=obj.findtext("truncated", "0") == "1",
                difficult=obj.findtext("difficult", "0") == "1",
            )
        )
    return VocSample(
        xml_path=xml_path,
        image_path=_find_image(xml_path, root.findtext("filename")),
        width=width,
        height=height,
        folder_code=xml_path.parent.name.strip().upper(),
        boxes=tuple(boxes),
    )


def collect_voc(root: Path) -> list[VocSample]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在: {root}")
    return [parse_voc(p) for p in sorted(root.rglob("*.xml"))]


def to_yolo_lines(sample: VocSample, skip_codes: frozenset[str] = frozenset()) -> list[str]:
    """转 YOLO 标签行;越界坐标裁回图内,退化成零面积的框丢弃。"""
    if sample.width <= 0 or sample.height <= 0:
        return []
    lines: list[str] = []
    for b in sample.boxes:
        if b.code in skip_codes or b.code not in CODE_TO_ID:
            continue
        x1 = max(0.0, min(b.xmin, b.xmax))
        y1 = max(0.0, min(b.ymin, b.ymax))
        x2 = min(float(sample.width), max(b.xmin, b.xmax))
        y2 = min(float(sample.height), max(b.ymin, b.ymax))
        bw, bh = x2 - x1, y2 - y1
        if bw <= 1 or bh <= 1:
            continue
        cx = (x1 + x2) / 2 / sample.width
        cy = (y1 + y2) / 2 / sample.height
        lines.append(
            f"{CODE_TO_ID[b.code]} {cx:.6f} {cy:.6f} "
            f"{bw / sample.width:.6f} {bh / sample.height:.6f}"
        )
    return lines


def describe(samples: list[VocSample]) -> dict:
    codes = Counter(b.code for s in samples for b in s.boxes)
    files = Counter(code for s in samples for code in {b.code for b in s.boxes})
    return {
        "xml_files": len(samples),
        "with_image": sum(1 for s in samples if s.image_path),
        "missing_image": sum(1 for s in samples if not s.image_path),
        "boxes": int(sum(codes.values())),
        "wafers": len({s.wafer_id for s in samples}),
        "unknown_codes": sorted(c for c in codes if c not in CODE_TO_ID),
        "by_code_boxes": dict(codes.most_common()),
        "by_code_files": dict(files.most_common()),
        "by_folder": dict(Counter(s.folder_code for s in samples).most_common()),
        "sizes": dict(Counter(f"{s.width}x{s.height}" for s in samples).most_common()),
    }
