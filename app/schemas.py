"""API 出入参模型。"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Task = Literal["pcb", "wafer"]


class HealthResponse(BaseModel):
    status: str
    app: str
    app_en: str
    version: str
    device: str
    pcb_model: str
    pcb_ready: bool
    wafer_model: str
    wafer_ready: bool
    agent_enabled: bool


class ClassInfo(BaseModel):
    id: int
    name: str
    is_ok: bool = False
    part: Optional[str] = None
    note: Optional[str] = None


class TaskInfo(BaseModel):
    task: Task
    title: str
    kind: str = Field(description="classification | detection")
    ready: bool
    model_path: str
    stage: str
    classes: list[ClassInfo]
    dataset: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class DetectionItem(BaseModel):
    label: str
    label_text: Optional[str] = None
    confidence: float
    bbox_xyxy: Optional[list[float]] = None


class PcbInspectResponse(BaseModel):
    id: Optional[int] = None
    created_at: str
    task: Task = "pcb"
    label: str
    label_text: Optional[str] = None
    confidence: float
    ok_prob: float
    status: str
    status_text: str
    probs: dict[str, float] = Field(default_factory=dict)
    input_mode: Optional[str] = None
    model: Optional[str] = None
    elapsed_ms: float = 0.0
    has_template: bool = False
    board_id: Optional[str] = None
    avi_item: Optional[str] = None
    work_order: Optional[str] = None
    batch_id: Optional[str] = None
    note: Optional[str] = None


class WaferInspectResponse(BaseModel):
    id: Optional[int] = None
    created_at: str
    task: Task = "wafer"
    num_detections: int
    detections: list[DetectionItem] = Field(default_factory=list)
    label: Optional[str] = None
    confidence: Optional[float] = None
    status: Optional[str] = None
    model: Optional[str] = None
    elapsed_ms: float = 0.0
    work_order: Optional[str] = None
    batch_id: Optional[str] = None
    note: Optional[str] = None


class RecordSummary(BaseModel):
    id: int
    created_at: str
    task: str
    source: str
    label: Optional[str] = None
    confidence: Optional[float] = None
    ok_prob: Optional[float] = None
    status: Optional[str] = None
    num_detections: int = 0
    image_path: Optional[str] = None
    template_path: Optional[str] = None
    model: Optional[str] = None
    input_mode: Optional[str] = None
    note: Optional[str] = None
    elapsed_ms: Optional[float] = None
    board_id: Optional[str] = None
    avi_item: Optional[str] = None
    work_order: Optional[str] = None
    batch_id: Optional[str] = None


class RecordDetail(RecordSummary):
    detections: list[DetectionItem] = Field(default_factory=list)


class StatsResponse(BaseModel):
    total_records: int
    by_task: dict[str, int] = Field(default_factory=dict)
    by_status: dict[str, int] = Field(default_factory=dict)
    by_label: dict[str, int] = Field(default_factory=dict)
    by_work_order: dict[str, int] = Field(default_factory=dict)
    avg_elapsed_ms: Optional[float] = None
    pcb_release_rate: Optional[float] = None
    pcb_records: int = 0


class BatchDeleteRequest(BaseModel):
    ids: list[int] = Field(default_factory=list)
