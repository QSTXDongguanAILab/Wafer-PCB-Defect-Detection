"""PCB 缺陷类别定义(终检 AVI 复判 / 假点过滤)。

类名取自甲方给的分类目录,共 10 类。其中「假点」是 AVI 误报,
判为假点即放行,其余 9 类为真缺陷,需人工复检或直接判 NG。

命名规律:部位(基材 / 焊盘) × 缺陷形态。
"""
from __future__ import annotations

# 放行类:AVI 报了但实际没缺陷
OK_LABEL = "假点"

# 规范类名(顺序即 class id,0 固定为假点,方便二分类头对齐)
LABELS: tuple[str, ...] = (
    "假点",
    "基材划痕",
    "基材异物",
    "基材擦花",
    "基材漏铜",
    "焊盘划痕",
    "焊盘异物",
    "焊盘损伤",
    "焊盘氧化",
    "焊盘脏污",
)

# 目录名 → 规范类名。测试集用了带严重度后缀的写法,训练集没有,这里统一。
ALIASES: dict[str, str] = {
    "基材异物-中等": "基材异物",
    "基材异物-轻微": "基材异物",
    "基材异物-严重": "基材异物",
}

LABEL_TO_ID: dict[str, int] = {name: i for i, name in enumerate(LABELS)}
OK_ID = LABEL_TO_ID[OK_LABEL]
NUM_CLASSES = len(LABELS)

# 判级结果
STATUS_PASS = "pass"      # 放行
STATUS_REVIEW = "review"  # 人工复检(模型不确定)
STATUS_NG = "ng"          # 判为真缺陷

STATUS_TEXT = {
    STATUS_PASS: "放行",
    STATUS_REVIEW: "人工复检",
    STATUS_NG: "缺陷(NG)",
}


def canonical(name: str) -> str:
    """把目录名/外部标签归一到规范类名。"""
    n = (name or "").strip()
    return ALIASES.get(n, n)


def label_id(name: str) -> int:
    """规范化后取 class id;未知类名抛错,避免静默错标。"""
    n = canonical(name)
    if n not in LABEL_TO_ID:
        raise KeyError(f"未知 PCB 类别: {name!r}(已知 {list(LABELS)})")
    return LABEL_TO_ID[n]


def is_ok(name: str) -> bool:
    return canonical(name) == OK_LABEL


def part_of(name: str) -> str:
    """部位:基材 / 焊盘 / -(假点无部位)。"""
    n = canonical(name)
    for part in ("基材", "焊盘"):
        if n.startswith(part):
            return part
    return "-"


def decide(pred_label: str, ok_prob: float, release_min_prob: float) -> str:
    """判级:只有「高置信度的假点」才放行。

    漏检(真缺陷判成假点)会让不良品流到客户,是事故;
    误报只是多一次人工复检,是成本。所以不确定时一律转复检。
    """
    if is_ok(pred_label):
        return STATUS_PASS if ok_prob >= release_min_prob else STATUS_REVIEW
    return STATUS_NG
