"""光伏硅片缺陷类别(切片后分选工序)。

标注是 Pascal-VOC XML,`<object><name>` 用的是产线缺陷代码,
甲方给的需求说明里没有代码↔中文名对照表 —— 这张表必须去问数据方,
在拿到之前不要自行猜译,否则整套 SOP 和看板文案都会错。

以下代码集合与统计是从 537 份 XML 实测汇总出来的(共 932 个框):
    X    373 框 / 211 图        HBB  117 / 101
    HBX  115 /  85              HS    76 /  76
    BX    65 /  33              KD    63 /  63
    DQK   51 /  51              HB    49 /  49
    XQK   14 /  13              BYW    5 /   2
    KYW    3 /   2              XHB    1 /   1

注意:测试集按「主缺陷代码」建了 9 个目录,但单张图的 XML 里常有多个不同代码的框,
另有 KYW / XHB / XQK 三个代码只出现在 XML 里、没有对应目录。
所以训练标签一律以 XML 为准,目录名只当粗分类线索。
"""
from __future__ import annotations

# 类别顺序固定(字母序),即 YOLO class id;改动会让旧权重失配
CODES: tuple[str, ...] = (
    "BX",
    "BYW",
    "DQK",
    "HB",
    "HBB",
    "HBX",
    "HS",
    "KD",
    "KYW",
    "X",
    "XHB",
    "XQK",
)

CODE_TO_ID: dict[str, int] = {c: i for i, c in enumerate(CODES)}
NUM_CLASSES = len(CODES)

# 代码 → 中文名。全部待甲方确认,确认后逐条补全并同步 rag_agent/data/sop/。
CODE_NAMES: dict[str, str] = {c: "待确认" for c in CODES}

# 样本极少的类:BYW(2图) / KYW(2图) / XHB(1图) 无法参与训练与评测,
# 先合并进忽略集,等数据方补样本再放开。
RARE_CODES: frozenset[str] = frozenset({"BYW", "KYW", "XHB"})


def code_id(code: str) -> int:
    c = (code or "").strip().upper()
    if c not in CODE_TO_ID:
        raise KeyError(f"未知硅片缺陷代码: {code!r}(已知 {list(CODES)})")
    return CODE_TO_ID[c]


def display(code: str) -> str:
    c = (code or "").strip().upper()
    name = CODE_NAMES.get(c, "待确认")
    return c if name == "待确认" else f"{c} {name}"
