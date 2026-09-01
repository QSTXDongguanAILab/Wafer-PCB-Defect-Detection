"""PCB 假点过滤模型:小 CNN + 双头(二分类 + 十分类)。

为什么自己写而不套 ResNet:输入只有 96x96、通道可能是 9,样本量三位数。
ImageNet 预训练权重的第一层是 3 通道,改通道后预训练价值本就大打折扣;
这个体量的小网 CPU 上几分钟就能跑完一轮实验,迭代速度比精度更值钱。

双头的分工:
    head_binary 假点 vs NG —— 业务主任务,决定放行还是转人工
    head_multi  10 类      —— 给处置 SOP 用,顺带当辅助监督
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from pcb.labels import LABELS, NUM_CLASSES
from pcb.loader import channels_for

ARCH_VERSION = 1


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class PairClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = NUM_CLASSES, width: int = 32) -> None:
        super().__init__()
        w = width
        self.features = nn.Sequential(
            _block(in_channels, w),      # 96 -> 48
            _block(w, w * 2),            # 48 -> 24
            _block(w * 2, w * 4),        # 24 -> 12
            _block(w * 4, w * 8),        # 12 -> 6
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(0.3)
        self.head_binary = nn.Linear(w * 8, 2)
        self.head_multi = nn.Linear(w * 8, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.features(x).flatten(1)
        z = self.dropout(z)
        return self.head_binary(z), self.head_multi(z)


def build_model(input_mode: str, width: int = 32) -> PairClassifier:
    return PairClassifier(channels_for(input_mode), NUM_CLASSES, width=width)


def save_checkpoint(
    model: PairClassifier,
    path: str | Path,
    *,
    input_mode: str,
    img_size: int,
    width: int = 32,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """权重连同输入约定一起存:推理端不能靠 config 猜 input_mode。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arch_version": ARCH_VERSION,
            "state_dict": model.state_dict(),
            "input_mode": input_mode,
            "img_size": int(img_size),
            "width": int(width),
            "labels": list(LABELS),
            "metrics": metrics or {},
        },
        path,
    )
    return path


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[PairClassifier, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PCB 权重不存在: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("labels") and list(ckpt["labels"]) != list(LABELS):
        raise ValueError(
            f"权重类别与当前 pcb/labels.py 不一致\n权重: {ckpt['labels']}\n当前: {list(LABELS)}"
        )
    model = build_model(ckpt["input_mode"], width=int(ckpt.get("width", 32)))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
