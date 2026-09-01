"""FastAPI 入口:PCB 假点过滤 + 硅片缺陷检测 + 记录追溯 + 看板。"""
from __future__ import annotations

import csv
import io
import mimetypes
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.config import load_settings
from app.db import (
    clear_records,
    count_records,
    delete_many,
    delete_record,
    get_record,
    image_paths,
    init_db,
    insert_record,
    list_records,
    stats as db_stats,
)
from app.schemas import (
    BatchDeleteRequest,
    HealthResponse,
    PcbInspectResponse,
    RecordDetail,
    RecordSummary,
    StatsResponse,
    TaskInfo,
    WaferInspectResponse,
)
from app.tasks import all_tasks, pcb_task, wafer_task

settings = load_settings()
init_db(settings)
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD = 16 * 1024 * 1024

AGENT_ENABLED = False

def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db(settings)
    from pcb.infer import get_predictor
    from wafer.infer import get_detector

    for name, obj in (("PCB", get_predictor()), ("硅片", get_detector())):
        state = "ready" if obj.ready else f"未训练({obj.weights})"
        print(f"[{settings.app_name_en}] {name} 模型: {state}")
    yield


app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description="PCB 假点过滤(成对 ROI 分类)+ 光伏硅片缺陷检测(目标检测)+ 记录追溯 + 处置 Agent",
    lifespan=lifespan,
)

# 缺陷处置 RAG/Agent:langchain 栈未装或未配 key 时跳过,不影响主功能
try:
    from rag_agent.api import router as _rag_router

    app.include_router(_rag_router, prefix="/agent")
    AGENT_ENABLED = True
    print(f"[{settings.app_name_en}] rag_agent 已挂载 /agent")
except Exception as _e:  # noqa: BLE001
    print(f"[{settings.app_name_en}] rag_agent 未启用: {_e}")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _rel_to_abs(rel: str | None) -> Path | None:
    if not rel:
        return None
    p = Path(rel)
    return p if p.is_absolute() else (settings.project_root / p).resolve()


def _save_upload(raw: bytes, prefix: str, suffix: str = ".jpg") -> str:
    """落盘并返回项目相对路径,方便记录与 /files 回显。"""
    out_dir = settings.outputs_path
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = out_dir / f"{prefix}_{ts}{suffix}"
    path.write_bytes(raw)
    return str(path.relative_to(settings.project_root)).replace("\\", "/")


async def _read_image(file: UploadFile) -> tuple[bytes, np.ndarray]:
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "上传文件为空")
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(400, f"图片过大(>{MAX_UPLOAD // 1024 // 1024}MB)")
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "无法解码图片")
    return raw, img


@app.get("/", response_class=HTMLResponse)
def dashboard() -> Response:
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return HTMLResponse(f"<h1>{settings.app_name}</h1><p>缺少 app/static/index.html</p>")
    return HTMLResponse(
        index.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store, max-age=0"}
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    from pcb.infer import get_predictor
    from wafer.infer import get_detector

    pcb_p, wafer_d = get_predictor(), get_detector()
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        app_en=settings.app_name_en,
        version=__version__,
        device=settings.device,
        pcb_model=settings.pcb.model_path,
        pcb_ready=pcb_p.ready,
        wafer_model=settings.wafer.model_path,
        wafer_ready=wafer_d.ready,
        agent_enabled=AGENT_ENABLED,
    )


@app.get("/tasks", response_model=list[TaskInfo])
def tasks() -> list[TaskInfo]:
    return [TaskInfo(**t) for t in all_tasks()]


@app.get("/tasks/{task}", response_model=TaskInfo)
def task_detail(task: str) -> TaskInfo:
    if task == "pcb":
        return TaskInfo(**pcb_task())
    if task == "wafer":
        return TaskInfo(**wafer_task())
    raise HTTPException(404, f"未知任务: {task}(可选 pcb / wafer)")


def _to_rgb_float(img_bgr: np.ndarray, size: int) -> np.ndarray:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != size or rgb.shape[1] != size:
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    return rgb.astype(np.float32) / 255.0


@app.post("/pcb/inspect", response_model=PcbInspectResponse)
async def pcb_inspect(
    image: UploadFile = File(..., description="待检 ROI 图"),
    template: Optional[UploadFile] = File(None, description="同位置标准模板图(_T)"),
    note: Optional[str] = Form(None),
    work_order: Optional[str] = Form(None),
    batch_id: Optional[str] = Form(None),
    board_id: Optional[str] = Form(None),
    avi_item: Optional[str] = Form(None),
    save: bool = Form(True),
) -> PcbInspectResponse:
    """AVI 疑点复判:给出类别 + 假点概率 + 放行/复检/NG 判级。"""
    from pcb.infer import ModelNotReady, get_predictor
    from pcb.labels import STATUS_TEXT

    predictor = get_predictor()
    try:
        predictor.load()
    except ModelNotReady as exc:
        raise HTTPException(503, str(exc)) from exc

    size = int(predictor.meta["img_size"])
    raw_test, img_test = await _read_image(image)
    arr_test = _to_rgb_float(img_test, size)
    arr_tmpl = None
    raw_tmpl = None
    if template is not None:
        raw_tmpl, img_tmpl = await _read_image(template)
        arr_tmpl = _to_rgb_float(img_tmpl, size)

    t0 = time.perf_counter()
    try:
        result = predictor.predict_arrays(arr_test, arr_tmpl)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"PCB 推理失败:{type(exc).__name__}: {exc}") from exc
    elapsed = (time.perf_counter() - t0) * 1000

    test_path = _save_upload(raw_test, "pcb") if save else None
    tmpl_path = _save_upload(raw_tmpl, "pcb_T") if (save and raw_tmpl) else None
    record_id = None
    if save:
        record_id = insert_record(
            created_at=_now(),
            task="pcb",
            source="upload",
            label=result["label"],
            confidence=result["confidence"],
            ok_prob=result["ok_prob"],
            status=result["status"],
            image_path=test_path,
            template_path=tmpl_path,
            model=result["model"],
            input_mode=result["input_mode"],
            note=note,
            elapsed_ms=elapsed,
            board_id=board_id,
            avi_item=avi_item,
            work_order=work_order,
            batch_id=batch_id,
        )
    return PcbInspectResponse(
        id=record_id,
        created_at=_now(),
        label=result["label"],
        label_text=STATUS_TEXT[result["status"]],
        confidence=result["confidence"],
        ok_prob=result["ok_prob"],
        status=result["status"],
        status_text=result["status_text"],
        probs=result["probs"],
        input_mode=result["input_mode"],
        model=result["model"],
        elapsed_ms=round(elapsed, 1),
        has_template=arr_tmpl is not None,
        board_id=board_id,
        avi_item=avi_item,
        work_order=work_order,
        batch_id=batch_id,
        note=note,
    )


@app.post("/wafer/inspect", response_model=WaferInspectResponse)
async def wafer_inspect(
    image: UploadFile = File(..., description="硅片图(640×640 灰度)"),
    note: Optional[str] = Form(None),
    work_order: Optional[str] = Form(None),
    batch_id: Optional[str] = Form(None),
    save: bool = Form(True),
) -> WaferInspectResponse:
    from wafer.infer import ModelNotReady, get_detector

    detector = get_detector()
    try:
        detector.load()
    except ModelNotReady as exc:
        raise HTTPException(503, str(exc)) from exc

    raw, img = await _read_image(image)
    t0 = time.perf_counter()
    try:
        dets = detector.predict(img)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"硅片推理失败:{type(exc).__name__}: {exc}") from exc
    elapsed = (time.perf_counter() - t0) * 1000

    top = max(dets, key=lambda d: d["confidence"]) if dets else None
    status = "ng" if dets else "pass"
    image_path = _save_upload(raw, "wafer", Path(image.filename or "x.png").suffix or ".png") if save else None
    record_id = None
    if save:
        record_id = insert_record(
            created_at=_now(),
            task="wafer",
            source="upload",
            label=top["label"] if top else None,
            confidence=top["confidence"] if top else None,
            status=status,
            detections=dets,
            image_path=image_path,
            model=detector.weights.name,
            note=note,
            elapsed_ms=elapsed,
            work_order=work_order,
            batch_id=batch_id,
        )
    return WaferInspectResponse(
        id=record_id,
        created_at=_now(),
        num_detections=len(dets),
        detections=dets,
        label=top["label"] if top else None,
        confidence=top["confidence"] if top else None,
        status=status,
        model=detector.weights.name,
        elapsed_ms=round(elapsed, 1),
        work_order=work_order,
        batch_id=batch_id,
        note=note,
    )


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    return StatsResponse(**db_stats())


@app.get("/records", response_model=list[RecordSummary])
def records(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    task: Optional[str] = Query(None, description="pcb | wafer"),
    label: Optional[str] = Query(None),
    status: Optional[str] = Query(None, description="pass | review | ng"),
    work_order: Optional[str] = Query(None),
    batch_id: Optional[str] = Query(None),
) -> list[RecordSummary]:
    rows = list_records(
        limit=limit, offset=offset, task=task, label=label,
        status=status, work_order=work_order, batch_id=batch_id,
    )
    return [RecordSummary(**{k: v for k, v in r.items() if k != "detections"}) for r in rows]


@app.get("/records/count")
def records_count(task: Optional[str] = Query(None)) -> dict:
    return {"count": count_records(task)}


CSV_COLUMNS = (
    "id", "created_at", "task", "source", "label", "confidence", "ok_prob", "status",
    "num_detections", "elapsed_ms", "board_id", "avi_item", "work_order", "batch_id",
    "model", "input_mode", "note", "image_path", "template_path",
)


@app.get("/records/export.csv")
def records_export_csv(
    limit: int = Query(1000, ge=1, le=20000),
    task: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    work_order: Optional[str] = Query(None),
):
    rows = list_records(limit=limit, task=task, status=status, work_order=work_order)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(CSV_COLUMNS)
    for r in rows:
        writer.writerow([r.get(c) if r.get(c) is not None else "" for c in CSV_COLUMNS])
    data = buf.getvalue().encode("utf-8-sig")  # BOM:Excel 打开中文不乱码
    return StreamingResponse(
        iter([data]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="records.csv"'},
    )


@app.get("/records/{record_id}", response_model=RecordDetail)
def record_detail(record_id: int) -> RecordDetail:
    row = get_record(record_id)
    if row is None:
        raise HTTPException(404, f"记录 {record_id} 不存在")
    return RecordDetail(**row)


def _unlink(paths: list[str]) -> int:
    n = 0
    for rel in paths:
        p = _rel_to_abs(rel)
        if p and p.exists():
            try:
                p.unlink()
                n += 1
            except OSError:
                pass
    return n


@app.delete("/records/{record_id}")
def record_delete(record_id: int) -> dict:
    if get_record(record_id) is None:
        raise HTTPException(404, f"记录 {record_id} 不存在")
    files = _unlink(image_paths([record_id]))
    if not delete_record(record_id):
        raise HTTPException(500, "删除失败")
    return {"deleted": record_id, "files_removed": files}


@app.post("/records/delete-batch")
def records_delete_batch(body: BatchDeleteRequest) -> dict:
    ids = sorted({int(i) for i in body.ids if int(i) > 0})
    if not ids:
        raise HTTPException(400, "ids 不能为空")
    if len(ids) > 500:
        raise HTTPException(400, "一次最多删 500 条")
    files = _unlink(image_paths(ids))
    return {"deleted": delete_many(ids), "files_removed": files, "ids": ids}


@app.delete("/records")
def records_clear(
    confirm: str = Query(..., description="必须传 YES"),
    task: Optional[str] = Query(None, description="只清某条业务线"),
) -> dict:
    if confirm != "YES":
        raise HTTPException(400, "需要 confirm=YES")
    rows = list_records(limit=20000, task=task)
    files = _unlink([p for r in rows for p in (r.get("image_path"), r.get("template_path")) if p])
    return {"deleted": clear_records(task), "files_removed": files}


@app.get("/files/{file_path:path}")
def get_file(file_path: str):
    """回显落盘图片。限制在项目根内,防目录穿越。"""
    p = _rel_to_abs(file_path)
    if p is None or not p.exists() or not p.is_file():
        raise HTTPException(404, "文件不存在")
    try:
        p.resolve().relative_to(settings.project_root.resolve())
    except ValueError as exc:
        raise HTTPException(403, "禁止访问项目目录之外的文件") from exc
    return FileResponse(str(p), media_type=mimetypes.guess_type(str(p))[0] or "application/octet-stream")
