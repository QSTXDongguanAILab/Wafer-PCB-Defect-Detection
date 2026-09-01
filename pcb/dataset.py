"""PCB 成对 ROI 样本的扫描、解析与切分(纯 Python,不依赖 torch)。

数据形态:AVI 每报一个疑点就存一对 100x100 小图
    <stem>.jpg    待检图
    <stem>_T.jpg  同位置的标准模板图

文件名:Cons_<板号>_<单元号>_<AVI检测项>_<序号>[_T].jpg
例:   Cons_202312250505183453_41_线路_1.jpg

标签来自所在目录名(测试集/<类名>/);直接散落在根目录的为未标注样本。
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pcb.labels import LABEL_TO_ID, canonical

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
TEMPLATE_SUFFIX = "_T"


@dataclass(frozen=True)
class Pair:
    """一个 AVI 疑点样本:待检图 + 模板图(模板可能缺失)。"""

    stem: str
    test_path: Path
    template_path: Path | None
    label: str | None  # 规范类名;None = 未标注
    board: str  # 板号,切分时按它分组,防同板泄漏
    unit: str  # 板内单元号
    avi_item: str  # AVI 检测项目(线路 / 大焊盘均匀度 / 漏铜 ...)

    @property
    def has_template(self) -> bool:
        return self.template_path is not None


def parse_stem(stem: str) -> tuple[str, str, str]:
    """从文件名主干取 (板号, 单元号, AVI检测项)。格式不符时退化为整段当板号。"""
    parts = stem.split("_")
    if len(parts) < 5:
        return stem, "", ""
    return parts[1], parts[2], "_".join(parts[3:-1])


def _label_from_dir(directory: Path) -> str | None:
    """目录名若是已知类别则作为标签,否则视为未标注。"""
    name = canonical(directory.name)
    return name if name in LABEL_TO_ID else None


def collect_pairs(root: Path) -> list[Pair]:
    """递归扫描 root 下所有成对样本。

    - 标签 = 直接父目录名(需是已知类别),否则 None
    - 只有模板图没有待检图的,视为脏数据丢弃(实测测试集存在个别单张)
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在: {root}")

    # 先按目录分组收集,模板图与待检图必然同目录
    by_dir: dict[Path, dict[str, Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        by_dir.setdefault(path.parent, {})[path.stem] = path

    pairs: list[Pair] = []
    for directory, files in by_dir.items():
        label = _label_from_dir(directory)
        for stem, path in sorted(files.items()):
            if stem.endswith(TEMPLATE_SUFFIX):
                continue  # 模板图由它的待检图带出
            board, unit, avi_item = parse_stem(stem)
            pairs.append(
                Pair(
                    stem=stem,
                    test_path=path,
                    template_path=files.get(stem + TEMPLATE_SUFFIX),
                    label=label,
                    board=board,
                    unit=unit,
                    avi_item=avi_item,
                )
            )
    pairs.sort(key=lambda p: (str(p.test_path)))
    return pairs


def split_labeled(pairs: list[Pair]) -> tuple[list[Pair], list[Pair]]:
    """拆成 (已标注, 未标注)。"""
    labeled = [p for p in pairs if p.label]
    unlabeled = [p for p in pairs if not p.label]
    return labeled, unlabeled


def group_split(
    pairs: list[Pair], val_ratio: float = 0.25, seed: int = 20260901
) -> tuple[list[Pair], list[Pair]]:
    """按板号分组切 train/val。

    同一块板切出的多个 ROI 高度相似,随机切图会让同板样本同时进 train 和 val,
    指标虚高十几个点然后上线崩掉。所以整板一起进同一侧。
    """
    if not pairs:
        return [], []
    boards: dict[str, list[Pair]] = {}
    for p in pairs:
        boards.setdefault(p.board, []).append(p)

    # 用 seed+板号做哈希排序:加数据后已有板的归属不变,结果可复现
    ordered = sorted(
        boards.items(),
        key=lambda kv: hashlib.sha256(f"{seed}:{kv[0]}".encode()).hexdigest(),
    )
    target_val = int(round(len(pairs) * val_ratio))
    val: list[Pair] = []
    train: list[Pair] = []
    for _board, items in ordered:
        if len(val) < target_val:
            val.extend(items)
        else:
            train.extend(items)
    return train, val


def class_counts(pairs: list[Pair]) -> dict[str, int]:
    counts = Counter(p.label or "(未标注)" for p in pairs)
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def describe(pairs: list[Pair]) -> dict:
    """给 CLI / manifest 用的统计快照。"""
    labeled, unlabeled = split_labeled(pairs)
    return {
        "pairs": len(pairs),
        "labeled": len(labeled),
        "unlabeled": len(unlabeled),
        "missing_template": sum(1 for p in pairs if not p.has_template),
        "boards": len({p.board for p in pairs}),
        "by_class": class_counts(labeled),
        "by_avi_item": dict(
            sorted(Counter(p.avi_item for p in pairs).items(), key=lambda kv: -kv[1])
        ),
    }
