"""PCB 数据扫描 / 切分 / 判级 的单元测试(用临时目录造数据,不依赖真实数据集)。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcb.dataset import (
    collect_pairs,
    describe,
    group_split,
    model_split,
    parse_stem,
    split_labeled,
)
from pcb.labels import canonical, decide, is_ok, label_id, part_of


def _img(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), (128, 128, 128)).save(path)


def _pair(
    directory: Path, board: str, unit: int, item: str = "线路", idx: int = 0, model: str = "Cons"
) -> str:
    stem = f"{model}_{board}_{unit}_{item}_{idx}"
    _img(directory / f"{stem}.jpg")
    _img(directory / f"{stem}_T.jpg")
    return stem


def test_parse_stem():
    assert parse_stem("Cons_202312250505183453_41_线路_1") == (
        "Cons",
        "202312250505183453",
        "41",
        "线路",
    )
    # 实测存在第二个机种前缀
    assert parse_stem("Pros_20240817123_7_大焊盘1_0")[0] == "Pros"
    # AVI 检测项自身带下划线也要能切对
    assert parse_stem("Cons_A_7_大焊盘_均匀度_3") == ("Cons", "A", "7", "大焊盘_均匀度")
    # 格式不符时不炸,整段当板号,机种留空
    assert parse_stem("weird") == ("", "weird", "", "")


def test_collect_pairs_labels_from_dir(tmp_path: Path):
    _pair(tmp_path / "测试集" / "假点", "B1", 1)
    _pair(tmp_path / "测试集" / "基材异物-中等", "B2", 1)
    _pair(tmp_path / "训练集", "B3", 1)  # 散在根目录 = 未标注

    pairs = collect_pairs(tmp_path)
    assert len(pairs) == 3
    labels = sorted((p.label or "-") for p in pairs)
    assert labels == ["-", "假点", "基材异物"]  # 目录别名已归一


def test_collect_pairs_drops_orphan_template(tmp_path: Path):
    _img(tmp_path / "假点" / "Cons_B9_1_线路_0_T.jpg")  # 只有模板图
    assert collect_pairs(tmp_path) == []


def test_pair_without_template(tmp_path: Path):
    _img(tmp_path / "假点" / "Cons_B8_1_线路_0.jpg")
    (pair,) = collect_pairs(tmp_path)
    assert pair.has_template is False
    assert describe([pair])["missing_template"] == 1


def test_group_split_no_board_leak(tmp_path: Path):
    for board in ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"):
        for unit in range(3):
            _pair(tmp_path / "假点", board, unit)
    labeled, _ = split_labeled(collect_pairs(tmp_path))
    train, val = group_split(labeled, val_ratio=0.25, seed=7)
    assert train and val
    assert len(train) + len(val) == len(labeled)
    # 同一块板不能同时出现在两侧,否则验证指标虚高
    assert not ({p.board for p in train} & {p.board for p in val})


def test_group_split_is_deterministic(tmp_path: Path):
    for board in ("B1", "B2", "B3", "B4"):
        _pair(tmp_path / "假点", board, 0)
    labeled, _ = split_labeled(collect_pairs(tmp_path))
    a = group_split(labeled, 0.25, 42)
    b = group_split(labeled, 0.25, 42)
    assert [p.stem for p in a[1]] == [p.stem for p in b[1]]


def test_model_split_holds_out_whole_product(tmp_path: Path):
    """换型泛化只能靠没训过的机种来验,不能靠按板号切。"""
    for board in ("C1", "C2"):
        _pair(tmp_path / "假点", board, 0, model="Cons")
    _pair(tmp_path / "焊盘氧化", "P1", 0, model="Pros")

    pairs = collect_pairs(tmp_path)
    train, holdout = model_split(pairs, "Pros")
    assert {p.model_code for p in train} == {"Cons"}
    assert {p.model_code for p in holdout} == {"Pros"}

    snap = describe(pairs)
    assert snap["by_model"] == {"Cons": 2, "Pros": 1}

    with pytest.raises(ValueError):
        model_split(pairs, "NotAModel")


def test_labels_helpers():
    assert canonical("基材异物-中等") == "基材异物"
    assert label_id("假点") == 0
    assert is_ok("假点") and not is_ok("焊盘氧化")
    assert part_of("焊盘划痕") == "焊盘"
    assert part_of("假点") == "-"
    with pytest.raises(KeyError):
        label_id("不存在的类")


def test_decide_favours_recall():
    # 高置信度假点才放行
    assert decide("假点", 0.95, 0.90) == "pass"
    # 不确定的假点转人工,而不是直接放行
    assert decide("假点", 0.89, 0.90) == "review"
    # 真缺陷一律 NG
    assert decide("基材漏铜", 0.99, 0.90) == "ng"
