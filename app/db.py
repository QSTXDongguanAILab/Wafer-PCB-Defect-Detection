"""SQLite 检测记录:PCB(分类判级)与硅片(目标检测)共用一张表。

两种任务的差异靠 task 字段和 detections_json 吸收:
    PCB   一条记录 = 一个 AVI 疑点,label/ok_prob/status 是主字段,detections 为空
    硅片  一条记录 = 一张图,detections_json 存框,label 取置信度最高的代码
这样列表、筛选、统计、导出只写一套。
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any

from app.config import Settings, get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    task TEXT NOT NULL,
    source TEXT NOT NULL,
    label TEXT,
    confidence REAL,
    ok_prob REAL,
    status TEXT,
    num_detections INTEGER NOT NULL DEFAULT 0,
    detections_json TEXT NOT NULL DEFAULT '[]',
    image_path TEXT,
    template_path TEXT,
    model TEXT,
    input_mode TEXT,
    note TEXT,
    elapsed_ms REAL,
    board_id TEXT,
    avi_item TEXT,
    work_order TEXT,
    batch_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_records_created_at ON records(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_records_task ON records(task);
CREATE INDEX IF NOT EXISTS idx_records_label ON records(label);
CREATE INDEX IF NOT EXISTS idx_records_status ON records(status);
CREATE INDEX IF NOT EXISTS idx_records_work_order ON records(work_order);
"""

_FIELDS = (
    "created_at", "task", "source", "label", "confidence", "ok_prob", "status",
    "num_detections", "detections_json", "image_path", "template_path", "model",
    "input_mode", "note", "elapsed_ms", "board_id", "avi_item", "work_order", "batch_id",
)


def connect(settings: Settings | None = None) -> sqlite3.Connection:
    settings = settings or get_settings()
    settings.db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.db_file), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(settings: Settings | None = None) -> None:
    conn = connect(settings)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _row(r: sqlite3.Row) -> dict[str, Any]:
    d = dict(r)
    raw = d.pop("detections_json", "[]")
    try:
        d["detections"] = json.loads(raw or "[]")
    except json.JSONDecodeError:
        d["detections"] = []
    return d


def insert_record(
    *,
    created_at: str,
    task: str,
    source: str = "upload",
    label: str | None = None,
    confidence: float | None = None,
    ok_prob: float | None = None,
    status: str | None = None,
    detections: list[dict[str, Any]] | None = None,
    image_path: str | None = None,
    template_path: str | None = None,
    model: str | None = None,
    input_mode: str | None = None,
    note: str | None = None,
    elapsed_ms: float | None = None,
    board_id: str | None = None,
    avi_item: str | None = None,
    work_order: str | None = None,
    batch_id: str | None = None,
    settings: Settings | None = None,
) -> int:
    init_db(settings)
    dets = detections or []
    values = {
        "created_at": created_at,
        "task": task,
        "source": source,
        "label": label,
        "confidence": confidence,
        "ok_prob": ok_prob,
        "status": status,
        "num_detections": len(dets),
        "detections_json": json.dumps(dets, ensure_ascii=False),
        "image_path": image_path,
        "template_path": template_path,
        "model": model,
        "input_mode": input_mode,
        "note": note,
        "elapsed_ms": elapsed_ms,
        "board_id": board_id,
        "avi_item": avi_item,
        "work_order": (work_order or "").strip() or None,
        "batch_id": (batch_id or "").strip() or None,
    }
    conn = connect(settings)
    try:
        cols = ", ".join(_FIELDS)
        marks = ", ".join("?" for _ in _FIELDS)
        cur = conn.execute(
            f"INSERT INTO records ({cols}) VALUES ({marks})",
            [values[f] for f in _FIELDS],
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def list_records(
    *,
    limit: int = 20,
    offset: int = 0,
    task: str | None = None,
    label: str | None = None,
    status: str | None = None,
    work_order: str | None = None,
    batch_id: str | None = None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    init_db(settings)
    conn = connect(settings)
    try:
        sql = "SELECT * FROM records WHERE 1=1"
        params: list[Any] = []
        for column, value in (
            ("task", task),
            ("label", label),
            ("status", status),
            ("work_order", work_order),
            ("batch_id", batch_id),
        ):
            if value:
                sql += f" AND {column} = ?"
                params.append(value)
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        return [_row(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def get_record(record_id: int, settings: Settings | None = None) -> dict[str, Any] | None:
    init_db(settings)
    conn = connect(settings)
    try:
        row = conn.execute("SELECT * FROM records WHERE id = ?", (record_id,)).fetchone()
        return _row(row) if row else None
    finally:
        conn.close()


def delete_record(record_id: int, settings: Settings | None = None) -> bool:
    init_db(settings)
    conn = connect(settings)
    try:
        cur = conn.execute("DELETE FROM records WHERE id = ?", (record_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_many(ids: list[int], settings: Settings | None = None) -> int:
    if not ids:
        return 0
    init_db(settings)
    conn = connect(settings)
    try:
        marks = ",".join("?" for _ in ids)
        cur = conn.execute(f"DELETE FROM records WHERE id IN ({marks})", list(ids))
        conn.commit()
        return int(cur.rowcount)
    finally:
        conn.close()


def clear_records(task: str | None = None, settings: Settings | None = None) -> int:
    init_db(settings)
    conn = connect(settings)
    try:
        if task:
            cur = conn.execute("DELETE FROM records WHERE task = ?", (task,))
        else:
            cur = conn.execute("DELETE FROM records")
        conn.commit()
        return int(cur.rowcount)
    finally:
        conn.close()


def count_records(task: str | None = None, settings: Settings | None = None) -> int:
    init_db(settings)
    conn = connect(settings)
    try:
        if task:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM records WHERE task = ?", (task,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) AS c FROM records").fetchone()
        return int(row["c"] if row else 0)
    finally:
        conn.close()


def image_paths(ids: list[int] | None = None, settings: Settings | None = None) -> list[str]:
    """记录关联的落盘图片(删除记录时一并清理)。"""
    init_db(settings)
    conn = connect(settings)
    try:
        if ids:
            marks = ",".join("?" for _ in ids)
            rows = conn.execute(
                f"SELECT image_path, template_path FROM records WHERE id IN ({marks})", list(ids)
            ).fetchall()
        else:
            rows = conn.execute("SELECT image_path, template_path FROM records").fetchall()
        out: list[str] = []
        for r in rows:
            out.extend(p for p in (r["image_path"], r["template_path"]) if p)
        return out
    finally:
        conn.close()


def stats(settings: Settings | None = None) -> dict[str, Any]:
    """看板 KPI:总量、按任务/状态/类别分布、平均耗时。

    PCB 关键 KPI 是放行率(pass 占比)—— 直接对应省下多少人工复检。
    """
    init_db(settings)
    conn = connect(settings)
    try:

        def group(column: str, where: str = "") -> dict[str, int]:
            sql = f"SELECT {column} AS k, COUNT(*) AS c FROM records {where} GROUP BY {column}"
            return {
                str(r["k"]): int(r["c"]) for r in conn.execute(sql).fetchall() if r["k"] is not None
            }

        total = int(conn.execute("SELECT COUNT(*) AS c FROM records").fetchone()["c"])
        avg_ms = conn.execute(
            "SELECT AVG(elapsed_ms) AS a FROM records WHERE elapsed_ms IS NOT NULL"
        ).fetchone()["a"]
        by_status_pcb = group("status", "WHERE task = 'pcb'")
        pcb_total = sum(by_status_pcb.values())
        return {
            "total_records": total,
            "by_task": group("task"),
            "by_status": group("status"),
            "by_label": dict(
                sorted(group("label").items(), key=lambda kv: -kv[1])[:20]
            ),
            "by_work_order": group("work_order"),
            "avg_elapsed_ms": round(float(avg_ms), 1) if avg_ms is not None else None,
            "pcb_release_rate": (
                round(by_status_pcb.get("pass", 0) / pcb_total, 4) if pcb_total else None
            ),
            "pcb_records": pcb_total,
        }
    finally:
        conn.close()
