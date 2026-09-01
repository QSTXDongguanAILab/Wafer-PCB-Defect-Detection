"""配置加载:config.yaml → 嵌套 dataclass,路径统一相对项目根解析。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PcbSettings:
    model_path: str = "models/pcb_pair_cls.pt"
    dataset_root: str = "../分类数据/PCB分类数据"
    input_mode: str = "stack_diff"
    img_size: int = 96
    release_min_prob: float = 0.90
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 20260901
    val_ratio: float = 0.25
    multi_loss_weight: float = 0.5


@dataclass
class WaferSettings:
    model_path: str = "models/wafer_yolo.pt"
    dataset_root: str = "../分类数据/硅片分类数据/硅片分类数据"
    yolo_dir: str = "data/wafer/yolo"
    img_size: int = 640
    epochs: int = 100
    batch_size: int = 16
    seed: int = 20260901
    val_ratio: float = 0.2
    conf: float = 0.25
    iou: float = 0.5


@dataclass
class Settings:
    app_name: str = "PCB 与光伏硅片缺陷检测"
    app_name_en: str = "Wafer-PCB-Defect-Detection"
    host: str = "127.0.0.1"
    port: int = 8788
    device: str = "cpu"
    jpeg_quality: int = 90
    db_path: str = "data/inspect.db"
    outputs_dir: str = "data/outputs"
    models_dir: str = "models"
    artifacts_dir: str = "artifacts"
    logs_dir: str = "logs"
    pcb: PcbSettings = field(default_factory=PcbSettings)
    wafer: WaferSettings = field(default_factory=WaferSettings)
    project_root: Path = PROJECT_ROOT

    def resolve(self, rel: str | Path) -> Path:
        p = Path(rel)
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()

    @property
    def db_file(self) -> Path:
        return self.resolve(self.db_path)

    @property
    def outputs_path(self) -> Path:
        return self.resolve(self.outputs_dir)

    @property
    def artifacts_path(self) -> Path:
        return self.resolve(self.artifacts_dir)

    @property
    def logs_path(self) -> Path:
        return self.resolve(self.logs_dir)


def _fill(cls: type, data: dict[str, Any]):
    """只接收 dataclass 已声明的字段,忽略 YAML 里的额外键(注释性配置不会炸)。"""
    from dataclasses import fields as dc_fields

    known = {f.name for f in dc_fields(cls)}
    return cls(**{k: v for k, v in (data or {}).items() if k in known})


_settings: Settings | None = None


def load_settings(config_path: str | Path | None = None) -> Settings:
    global _settings
    path = Path(config_path or os.environ.get("WPDD_CONFIG") or PROJECT_ROOT / "config.yaml")
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    data: dict[str, Any] = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    top = {k: v for k, v in data.items() if k not in {"pcb", "wafer"}}
    s = _fill(Settings, top)
    s.pcb = _fill(PcbSettings, data.get("pcb") or {})
    s.wafer = _fill(WaferSettings, data.get("wafer") or {})
    s.project_root = PROJECT_ROOT

    s.outputs_path.mkdir(parents=True, exist_ok=True)
    s.artifacts_path.mkdir(parents=True, exist_ok=True)
    s.logs_path.mkdir(parents=True, exist_ok=True)
    s.db_file.parent.mkdir(parents=True, exist_ok=True)
    _settings = s
    return s


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        return load_settings()
    return _settings
