"""LangGraph ReAct Agent:编排「查 SOP + 查历史」→ 给多步处置方案。

用 langgraph.prebuilt.create_react_agent 构建 Function Calling 循环,
Qwen3-30B-A3B-Instruct 做决策,工具为 query_sop / query_history。
高危动作(换件/停机等)在方案中标注,由 HITL 模块识别 + 端点确认。
"""
from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from rag_agent.agent.tools import query_history, query_sop
from rag_agent.config import rag_config
from rag_agent.hitl import extract_high_risk_actions
from rag_agent.rag.retriever import retrieve

SYSTEM_PROMPT = """你是 PCB 与光伏硅片产线的缺陷处置专家 Agent。给定检出的缺陷类别,按以下流程工作:
1. 调用 query_sop 查询该缺陷的标准处置流程(SOP)。
2. 调用 query_history 查询该缺陷的历史出现情况,判断是否频发。
3. 综合两者,给出多步处置方案:判定确认 → 严重度评估 → 处置步骤 → 复检标准。

规则:
- 涉及"报废/返修/重工/停线/停机/补焊/挖修/整批拦截"等高危动作时,在该步骤后明确标注【需人工确认】。
- PCB 的「假点」是 AVI 误报,处置就是放行,不要给它编返修步骤。
- 用中文回答,条理清晰,分步骤编号。
- 严格基于查询到的 SOP 和历史作答,不要编造规程之外的步骤;SOP 标为待确认的内容要如实说明。"""

_agent: Any = None


def get_agent() -> Any:
    """惰性构建并缓存 ReAct Agent。"""
    global _agent
    if _agent is None:
        kwargs: dict[str, Any] = {}
        # Qwen3/3.5 是混合推理模型,默认开思考会让 ReAct 每轮多吐大段 reasoning,
        # 处置方案只要结论,关掉省时间也省 token(非 Qwen 模型不认这个字段,故加前缀判断)
        if rag_config.chat_model.startswith("Qwen/"):
            kwargs["extra_body"] = {"enable_thinking": False}
        llm = ChatOpenAI(
            model=rag_config.chat_model,
            api_key=rag_config.api_key,
            base_url=rag_config.base_url,
            temperature=0.3,
            **kwargs,
        )
        _agent = create_react_agent(
            llm, tools=[query_sop, query_history], prompt=SYSTEM_PROMPT
        )
    return _agent


def dispose_with_agent(defect_label: str, record_id: int | None = None) -> dict:
    """Agent 编排处置:多步方案 + 高危待确认项。

    Returns: {found, dispose, high_risk_actions, needs_confirmation}
    """
    # SOP 库里没有该类别时直接拒答:否则模型会绕过 query_sop 的"未找到"提示
    # 自行编造处置步骤(实测 4B 会给 person 类编出一套完整方案),违反 SYSTEM_PROMPT
    # 的"不要编造规程之外的步骤",也省掉一次无意义的 LLM 调用。
    if not retrieve(defect_label, k=1):
        return {
            "found": False,
            "dispose": f"未找到 {defect_label} 类型的处置 SOP,请人工处理。",
            "high_risk_actions": [],
            "needs_confirmation": False,
        }
    agent = get_agent()
    rid = f"(检测记录 #{record_id}) " if record_id is not None else ""
    user_msg = (
        f"本次检测{rid}检出缺陷类别:{defect_label}。"
        f"请查询其 SOP 和历史出现情况,给出完整处置方案。"
    )
    result = agent.invoke({"messages": [("user", user_msg)]})
    plan = result["messages"][-1].content
    high_risk = extract_high_risk_actions(plan)
    return {
        "found": True,
        "dispose": plan,
        "high_risk_actions": high_risk,
        "needs_confirmation": bool(high_risk),
    }
