"""硅片缺陷检测推理封装(YOLO),权重未就绪时明确报错而不是瞎猜。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.config import get_settings
from wafer.labels import display


class ModelNotReady(RuntimeError):
    pass


class WaferDetector:
    def __init__(self, weights: Path | None = None) -> None:
        s = get_settings()
        self.settings = s
        self.weights = Path(weights) if weights else s.resolve(s.wafer.model_path)
        self.model = None

    @property
    def ready(self) -> bool:
        return self.weights.exists()

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.ready:
            raise ModelNotReady(
                f"硅片权重不存在: {self.weights}。先跑 python -m wafer.prepare --write "
                "再 python -m wafer.train"
            )
        from ultralytics import YOLO

        self.model = YOLO(str(self.weights))

    def predict(self, image_bgr: np.ndarray) -> list[dict]:
        self.load()
        s = self.settings.wafer
        res = self.model.predict(
            source=image_bgr,
            conf=s.conf,
            iou=s.iou,
            imgsz=s.img_size,
            device=self.settings.device,
            verbose=False,
        )[0]
        out: list[dict] = []
        if res.boxes is None or not len(res.boxes):
            return out
        names = res.names or {}
        for box, score, cid in zip(
            res.boxes.xyxy.cpu().numpy(),
            res.boxes.conf.cpu().numpy(),
            res.boxes.cls.cpu().numpy().astype(int),
        ):
            code = str(names.get(int(cid), cid))
            out.append(
                {
                    "label": code,
                    "label_text": display(code),
                    "confidence": round(float(score), 4),
                    "bbox_xyxy": [round(float(v), 2) for v in box.tolist()],
                }
            )
        return out


_detector: WaferDetector | None = None


def get_detector() -> WaferDetector:
    global _detector
    if _detector is None:
        _detector = WaferDetector()
    return _detector
