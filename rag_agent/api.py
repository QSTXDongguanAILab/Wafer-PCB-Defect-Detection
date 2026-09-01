"""rag_agent FastAPI 路由,挂在 /agent(由 app/main.py include)。"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.db import get_record
from rag_agent.config import rag_config

router = APIRouter(tags=["rag-agent"])


def _preflight() -> None:
    """启用前置条件检查:缺依赖/缺 key/缺索引时给可执行的 503,而不是裸 500。

    裸 500 时 FastAPI 返回 text/plain "Internal Server Error",
    前端 r.json() 会抛 SyntaxError,错误信息完全丢失。
    """
    if not rag_config.api_key or rag_config.api_key.startswith("sk-your-key"):
        raise HTTPException(
            status_code=503,
            detail="SILICONFLOW_API_KEY 未设置:复制 rag_agent/.env.example 为 "
            "rag_agent/.env 并填入真实 key,然后重启服务。",
        )
    if not rag_config.chroma_dir.exists():
        raise HTTPException(
            status_code=503,
            detail="SOP 向量索引未构建:先运行 python -m rag_agent.build_index。",
        )


@router.get("/", response_class=HTMLResponse)
def workspace():
    """处置工作台(独立前端页,零侵入主看板 static/)。"""
    return (Path(__file__).parent / "ui.html").read_text(encoding="utf-8")


@router.get("/dispose")
def dispose_record(record_id: int, use_agent: bool = True):
    """读检测记录的缺陷类别,返回处置方案。

    - use_agent=True(默认):Agent 编排(查SOP + 查历史 + 多步方案 + 高危标注)。
    - use_agent=False:直接 RAG 检索(纯 SOP + 来源)。
    """
    rec = get_record(record_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"record {record_id} not found")
    top_label = rec.get("label") or "unknown"
    _preflight()
    try:
        if use_agent:
            from rag_agent.agent.graph import dispose_with_agent

            result = dispose_with_agent(top_label, record_id)
        else:
            from rag_agent.rag.retriever import dispose

            result = dispose(top_label)
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"处置 Agent 依赖未安装({exc.name}):"
            f"pip install -r rag_agent/requirements.txt",
        ) from exc
    except Exception as exc:  # 上游 LLM/向量库异常,带类型回传前端便于排查
        raise HTTPException(
            status_code=502, detail=f"处置失败:{type(exc).__name__}: {exc}"
        ) from exc
    return {
        "record_id": record_id,
        "top_label": top_label,
        "status": rec.get("status"),
        **result,
    }


class ConfirmRequest(BaseModel):
    record_id: int
    action: str
    approved: bool = True
    operator: str = "unknown"


@router.post("/dispose/confirm")
def confirm_action(req: ConfirmRequest):
    """高危处置动作人工确认(HITL):Agent 标注的高危项由人来批准。"""
    from rag_agent.hitl import record_confirmation

    entry = record_confirmation(req.record_id, req.action, req.approved, req.operator)
    return {"confirmed": True, **entry}


@router.get("/health")
def health():
    return {"ok": True, "module": "rag_agent"}
