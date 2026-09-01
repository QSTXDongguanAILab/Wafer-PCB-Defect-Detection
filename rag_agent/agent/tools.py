"""Agent 工具:LangChain @tool 包装,供 ReAct Agent 调用。

- query_sop:查缺陷维修 SOP(走 RAG 检索)
- query_history:查历史同类检测记录(读 records 表)
"""
from __future__ import annotations

from langchain_core.tools import tool

from app.db import list_records
from rag_agent.rag.retriever import retrieve


@tool
def query_sop(defect_label: str) -> str:
    """查询指定缺陷类别的标准处置流程 SOP。

    参数 defect_label:缺陷类名。
      PCB 十类:假点 / 基材划痕 / 基材异物 / 基材擦花 / 基材漏铜 /
               焊盘划痕 / 焊盘异物 / 焊盘损伤 / 焊盘氧化 / 焊盘脏污
      光伏硅片:产线缺陷代码(BX / DQK / HB / HBB / HBX / HS / KD / X ...)
    返回该缺陷的判定依据、严重度分级、处置步骤、复检标准。
    """
    hits = retrieve(defect_label, k=4)
    if not hits:
        return f"未找到 {defect_label} 的处置 SOP,可能无对应规程。"
    parts = [f"[{h['section']}] {h['content']}" for h in hits]
    return f"{defect_label} 处置 SOP:\n\n" + "\n\n".join(parts)


@tool
def query_history(defect_label: str, limit: int = 5) -> str:
    """查询历史检测记录中该缺陷类别的出现情况(判断频次/趋势/是否首例)。

    参数 defect_label:缺陷类名。limit:返回条数。
    """
    try:
        recs = list_records(label=defect_label, limit=limit)
    except Exception as e:  # noqa: BLE001
        return f"查询历史失败: {e}"
    if not recs:
        return f"历史中未查到 {defect_label} 的检测记录(本次可能是首例)。"
    lines = []
    for r in recs:
        lines.append(
            f"- 记录#{r.get('id')} [{r.get('status')}] {r.get('created_at', '')} "
            f"业务线:{r.get('task')} 工单:{r.get('work_order') or '-'} "
            f"类别:{r.get('label') or '-'} 假点概率:{r.get('ok_prob')}"
        )
    return f"{defect_label} 历史记录(近{len(recs)}条):\n" + "\n".join(lines)
