"""PCB 假点过滤推理封装:给 API 和预标注脚本共用。

权重里存了 input_mode / img_size,推理端不从 config 猜,避免训练与推理表示不一致
这类「指标好看、上线全错」的问题。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.config import get_settings
from pcb.dataset import Pair
from pcb.labels import LABELS, OK_LABEL, STATUS_NG, STATUS_PASS, STATUS_TEXT, decide
from pcb.loader import _load, build_input


class ModelNotReady(RuntimeError):
    """权重还没训出来。让调用方给出可执行的提示,而不是返回一个瞎猜的结果。"""


class PairPredictor:
    def __init__(self, weights: Path | None = None, device: str | None = None) -> None:
        s = get_settings()
        self.settings = s
        self.weights = Path(weights) if weights else s.resolve(s.pcb.model_path)
        self.device = device or s.device
        self.model = None
        self.meta: dict[str, Any] = {}

    @property
    def ready(self) -> bool:
        return self.weights.exists()

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.ready:
            raise ModelNotReady(
                f"PCB 权重不存在: {self.weights}。先训练:python -m pcb.train --compare"
            )
        from pcb.model import load_checkpoint

        self.model, self.meta = load_checkpoint(self.weights, self.device)

    def predict_arrays(self, test_img: np.ndarray, tmpl_img: np.ndarray | None) -> dict:
        """test_img/tmpl_img: HWC float32 [0,1],尺寸须为权重记录的 img_size。"""
        import torch

        self.load()
        if tmpl_img is None:
            tmpl_img = test_img.copy()
        x = build_input(test_img, tmpl_img, self.meta["input_mode"])
        with torch.no_grad():
            logit_b, logit_m = self.model(torch.from_numpy(x)[None].to(self.device))
            p_bin = torch.softmax(logit_b, 1)[0].cpu().numpy()
            p_multi = torch.softmax(logit_m, 1)[0].cpu().numpy()

        ok_prob = float(p_bin[0])
        # 放行与否只看二分类头(主任务),类别名交给多分类头(给 SOP 用)
        status = decide(OK_LABEL if ok_prob >= 0.5 else STATUS_NG, ok_prob, self.settings.pcb.release_min_prob)
        if status == STATUS_PASS:
            label, conf = OK_LABEL, ok_prob
        else:
            # 已判定不放行,类名就在 9 个真缺陷里选,不允许再回到「假点」自相矛盾
            ng_idx = int(np.argmax(p_multi[1:])) + 1
            label, conf = LABELS[ng_idx], float(p_multi[ng_idx])
        return {
            "label": label,
            "ok_prob": round(ok_prob, 4),
            "confidence": round(float(conf), 4),
            "status": status,
            "status_text": STATUS_TEXT[status],
            "probs": {LABELS[i]: round(float(v), 4) for i, v in enumerate(p_multi)},
            "input_mode": self.meta["input_mode"],
            "model": self.weights.name,
        }

    def predict_paths(self, test_path: str | Path, template_path: str | Path | None) -> dict:
        self.load()
        size = int(self.meta["img_size"])
        test_img = _load(test_path, size)
        tmpl_img = _load(template_path, size) if template_path else None
        out = self.predict_arrays(test_img, tmpl_img)
        out["has_template"] = template_path is not None
        return out

    def predict_pair(self, pair: Pair) -> dict:
        return self.predict_paths(pair.test_path, pair.template_path)


_predictor: PairPredictor | None = None


def get_predictor() -> PairPredictor:
    global _predictor
    if _predictor is None:
        _predictor = PairPredictor()
    return _predictor
