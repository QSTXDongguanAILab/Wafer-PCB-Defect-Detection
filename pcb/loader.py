"""PCB 成对样本的 torch Dataset:输入表示 + 几何/光照增广。

输入表示(input_mode)是本任务最关键的超参:
    single      仅待检图                3ch —— 基线的基线
    stack       待检 + 模板 通道拼接    6ch
    diff        待检 - 模板 差分        3ch —— AVI 现场判据本就是「与标准图的差异」
    stack_diff  待检 + 模板 + 差分      9ch
四种都要跑一遍再定,别靠推理拍板。
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from pcb.dataset import Pair
from pcb.labels import OK_ID, label_id

INPUT_MODES = ("single", "stack", "diff", "stack_diff")
_CHANNELS = {"single": 3, "stack": 6, "diff": 3, "stack_diff": 9}


def channels_for(mode: str) -> int:
    if mode not in _CHANNELS:
        raise ValueError(f"未知 input_mode={mode!r},可选 {INPUT_MODES}")
    return _CHANNELS[mode]


def _load(path, size: int) -> np.ndarray:
    """读图 → RGB → resize → HWC float32 [0,1]。"""
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0


def build_input(test_img: np.ndarray, tmpl_img: np.ndarray, mode: str) -> np.ndarray:
    """按 input_mode 拼出 CHW 张量数据(未转 torch)。"""
    t = test_img * 2.0 - 1.0
    m = tmpl_img * 2.0 - 1.0
    if mode == "single":
        out = t
    elif mode == "stack":
        out = np.concatenate([t, m], axis=2)
    elif mode == "diff":
        out = t - m
    elif mode == "stack_diff":
        out = np.concatenate([t, m, t - m], axis=2)
    else:
        raise ValueError(f"未知 input_mode={mode!r}")
    return np.ascontiguousarray(out.transpose(2, 0, 1))


def _augment(
    test_img: np.ndarray, tmpl_img: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """几何变换对待检图和模板图必须完全一致,否则人为造出假差分。

    光照抖动同样两图同步 —— 模拟整体亮度漂移,而不是制造伪缺陷。
    """
    if rng.random() < 0.5:
        test_img, tmpl_img = test_img[:, ::-1], tmpl_img[:, ::-1]
    if rng.random() < 0.5:
        test_img, tmpl_img = test_img[::-1], tmpl_img[::-1]
    k = int(rng.integers(0, 4))
    if k:
        test_img, tmpl_img = np.rot90(test_img, k), np.rot90(tmpl_img, k)
    gain = 1.0 + float(rng.normal(0, 0.06))
    bias = float(rng.normal(0, 0.03))
    test_img = np.clip(test_img * gain + bias, 0.0, 1.0)
    tmpl_img = np.clip(tmpl_img * gain + bias, 0.0, 1.0)
    return np.ascontiguousarray(test_img), np.ascontiguousarray(tmpl_img)


class PairDataset(Dataset):
    """成对 ROI 分类数据集。

    每条样本返回 (x, y_multi, y_binary):
        y_multi  10 类 class id
        y_binary 0=假点(放行) 1=真缺陷(NG) —— 业务真正要的那一刀
    未标注样本(推理/预标注用)的 y 全为 -1。
    """

    def __init__(
        self,
        pairs: list[Pair],
        *,
        input_mode: str = "stack_diff",
        img_size: int = 96,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        self.pairs = list(pairs)
        self.input_mode = input_mode
        self.img_size = int(img_size)
        self.augment = bool(augment)
        self.channels = channels_for(input_mode)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        import torch

        p = self.pairs[index]
        test_img = _load(p.test_path, self.img_size)
        # 模板缺失时用待检图自身兜底:差分恒为 0,等价于退化成单流,不至于整条样本丢掉
        tmpl_img = _load(p.template_path, self.img_size) if p.has_template else test_img.copy()
        if self.augment:
            test_img, tmpl_img = _augment(test_img, tmpl_img, self._rng)
        x = torch.from_numpy(build_input(test_img, tmpl_img, self.input_mode))

        if p.label is None:
            return x, -1, -1
        cid = label_id(p.label)
        return x, cid, int(cid != OK_ID)
