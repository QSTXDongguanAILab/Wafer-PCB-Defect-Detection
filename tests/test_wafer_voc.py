"""硅片 VOC 解析与 YOLO 转换的单元测试。"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wafer.labels import CODE_TO_ID, RARE_CODES
from wafer.voc import collect_voc, describe, parse_voc, to_yolo_lines

XML = """<annotation>
  <folder>{folder}</folder>
  <filename>{name}.png</filename>
  <size><width>{w}</width><height>{h}</height><depth>1</depth></size>
  <segmented>0</segmented>
{objects}
</annotation>
"""

OBJ = """  <object>
    <name>{code}</name><truncated>{tr}</truncated><difficult>0</difficult>
    <bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox>
  </object>"""


def _write(directory: Path, name: str, boxes, w: int = 640, h: int = 640) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    Image.new("L", (w, h), 0).save(directory / f"{name}.png")
    objects = "\n".join(
        OBJ.format(code=c, x1=x1, y1=y1, x2=x2, y2=y2, tr=tr) for c, x1, y1, x2, y2, tr in boxes
    )
    path = directory / f"{name}.xml"
    path.write_text(
        XML.format(folder=directory.name, name=name, w=w, h=h, objects=objects), encoding="utf-8"
    )
    return path


def test_parse_and_wafer_id(tmp_path: Path):
    path = _write(tmp_path / "BX", "34090654224052_3", [("X", 100, 100, 200, 200, 0)])
    s = parse_voc(path)
    assert s.width == 640 and s.height == 640
    assert s.folder_code == "BX"
    assert s.image_path is not None and s.image_path.suffix == ".png"
    assert s.wafer_id == "34090654224052"  # 同片多图按它分组
    assert len(s.boxes) == 1 and s.boxes[0].code == "X"


def test_multi_code_in_one_file(tmp_path: Path):
    """一张图里混多个代码是常态,目录名只是主代码。"""
    path = _write(
        tmp_path / "BX",
        "W1_0",
        [("X", 10, 10, 50, 50, 0), ("HBB", 60, 60, 80, 80, 0), ("BX", 90, 90, 120, 130, 0)],
    )
    s = parse_voc(path)
    assert {b.code for b in s.boxes} == {"X", "HBB", "BX"}
    lines = to_yolo_lines(s)
    assert len(lines) == 3
    assert lines[0].split()[0] == str(CODE_TO_ID["X"])


def test_yolo_normalization_and_clipping(tmp_path: Path):
    # 越界坐标(x2 > 图宽)必须裁回图内,否则 YOLO 训练会拿到 >1 的归一化值
    path = _write(tmp_path / "X", "W2_0", [("X", 0, 0, 1280, 320, 1)], w=640, h=640)
    (line,) = to_yolo_lines(parse_voc(path))
    cid, cx, cy, bw, bh = line.split()
    assert cid == str(CODE_TO_ID["X"])
    assert abs(float(cx) - 0.5) < 1e-6 and abs(float(bw) - 1.0) < 1e-6
    assert abs(float(cy) - 0.25) < 1e-6 and abs(float(bh) - 0.5) < 1e-6


def test_degenerate_box_dropped(tmp_path: Path):
    path = _write(tmp_path / "X", "W3_0", [("X", 100, 100, 100, 100, 0)])
    assert to_yolo_lines(parse_voc(path)) == []


def test_rare_codes_skipped(tmp_path: Path):
    rare = sorted(RARE_CODES)[0]
    path = _write(tmp_path / "X", "W4_0", [(rare, 10, 10, 50, 50, 0), ("X", 60, 60, 90, 90, 0)])
    assert len(to_yolo_lines(parse_voc(path), RARE_CODES)) == 1


def test_describe_flags_unknown_code(tmp_path: Path):
    _write(tmp_path / "X", "W5_0", [("ZZZ", 10, 10, 50, 50, 0)])
    snap = describe(collect_voc(tmp_path))
    assert snap["unknown_codes"] == ["ZZZ"]
    assert to_yolo_lines(parse_voc(tmp_path / "X" / "W5_0.xml")) == []
