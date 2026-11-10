"""ЖИ-агенттің LangGraph күйлер графы (Сурет 13).

    START → agent ──(tool_calls)──→ tools ──→ agent
              │
              └──(жауап жобасы)──→ verify ──(өтті)──→ END
                                     │
                                     ├──(өтпеді, қайталау < N)──→ agent (шлюз ескертуімен)
                                     └──(өтпеді, лимит)──→ safe_fail → END

ReAct циклі: LLM ойлайды → құрал шақырады → нәтижені бақылайды → ... → жауап.
Тексеру шлюзі (verify) — детерминистік код: LLM-нің жауабындағы әр сан құрал нәтижесімен сәйкес келуі тиіс.
"""
from __future__ import annotations

import json
import operator
import time
import uuid
from typing import Annotated, Any, TypedDict

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.agent import prompts
from app.agent.gate import verify_answer
from app.agent.llm import LLMNotConfigured, get_chat_model, usage_tokens
from app.agent.tools import compact, langchain_tools, run_tool
from app.core.config import get_settings
from app.core.farms import get_farm


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    lang: str
    farm_id: str
    tool_outputs: Annotated[list[dict], operator.add]
    trace: Annotated[list[dict], operator.add]
    retries: int
    rounds: int
    tokens: int
    gate: dict
    final: str
    status: str  # ok | safe_fail | error


def _system_prompt(lang: str, farm_id: str) -> str:
    from app.services.forecast_service import now_utc, to_local

    farm = get_farm(farm_id)
    now = to_local(now_utc(), farm)
    try:
        from app.services.history_service import data_range

        r = data_range("kelmarsh")
        hist = f"{r['start_local']} – {r['end_local']}"
    except Exception:
        hist = "—"
    return prompts.SYSTEM[lang].format(
        now_local=now.strftime("%Y-%m-%d %H:%M (%A)"), tomorrow=(now + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        farm_id=farm.id, farm_name=farm.name, capacity=farm.capacity_mw, farms=prompts.farms_text(), history_range=hist,
    )


def build_graph(llm=None):
    s = get_settings()
    model = llm or get_chat_model()
    bound = model.bind_tools(langchain_tools())

    def agent(state: AgentState) -> dict:
        t0 = time.time()
        msg = bound.invoke(state["messages"])
        calls = [{"name": c["name"], "args": c["args"]} for c in (getattr(msg, "tool_calls", None) or [])]
        ev = {"step": "llm", "t": round(time.time() - t0, 2), "tool_calls": calls,
              "draft": None if calls else (msg.content or "")[:4000]}
        return {"messages": [msg], "trace": [ev], "tokens": state.get("tokens", 0) + usage_tokens(msg),
                "rounds": state.get("rounds", 0) + (1 if calls else 0)}

    def tools(state: AgentState) -> dict:
        last: AIMessage = state["messages"][-1]
        msgs, outs, trace = [], [], []
        for call in last.tool_calls:
            args = dict(call["args"])
            args.setdefault("farm_id", state["farm_id"])
            out = run_tool(call["name"], args, lang=state["lang"])
            outs.append(out)
            trace.append({"step": "tool", "name": call["name"], "args": args, "t": out.get("_elapsed_s"),
                          "error": out.get("error"), "gate": out.get("gate"),
                          "summary": out.get("summary") or out.get("metrics")})
            msgs.append(ToolMessage(content=json.dumps(compact(call["name"], out), ensure_ascii=False, default=str),
                                    tool_call_id=call["id"], name=call["name"]))
        return {"messages": msgs, "tool_outputs": outs, "trace": trace}

    def route_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None) and state.get("rounds", 0) <= s.agent_max_tool_rounds:
            return "tools"
        return "verify"

    def verify(state: AgentState) -> dict:
        draft = state["messages"][-1].content or ""
        g = verify_answer(draft, state.get("tool_outputs", []), state["question"]).to_dict()
        ev = {"step": "gate", "passed": g["passed"], "checks": g["checks"], "retry": state.get("retries", 0)}
        upd: dict[str, Any] = {"gate": g, "trace": [ev]}
        if g["passed"]:
            upd.update(final=draft, status="ok")
        else:
            nums = next((c.get("unsupported") for c in g["checks"] if c["check"] == "faithfulness"), []) or []
            upd["retries"] = state.get("retries", 0) + 1
            if upd["retries"] <= s.agent_max_retries:
                failed = [c["check"] for c in g["checks"] if not c["passed"]]
                txt = prompts.GATE_FEEDBACK[state["lang"]].format(nums=", ".join(map(str, nums)) or ", ".join(failed))
                upd["messages"] = [HumanMessage(content=txt)]
        return upd

    def route_verify(state: AgentState) -> str:
        if state.get("status") == "ok":
            return END
        return "agent" if state.get("retries", 0) <= s.agent_max_retries else "safe_fail"

    def safe_fail(state: AgentState) -> dict:
        g = state.get("gate", {})
        nums = next((c.get("unsupported") for c in g.get("checks", []) if c["check"] == "faithfulness"), []) or []
        # расталмаған сандардың өзі пайдаланушыға көрсетілмейді (тек саны) — олар трассада қалады
        txt = prompts.SAFE_FAIL[state["lang"]].format(n=len(nums) or 1)
        return {"final": txt, "status": "safe_fail", "trace": [{"step": "safe_fail"}]}

    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_node("verify", verify)
    g.add_node("safe_fail", safe_fail)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_agent, {"tools": "tools", "verify": "verify"})
    g.add_edge("tools", "agent")
    g.add_conditional_edges("verify", route_verify, {END: END, "agent": "agent", "safe_fail": "safe_fail"})
    g.add_edge("safe_fail", END)
    return g.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def initial_state(question: str, lang: str, farm_id: str, history: list[dict] | None = None) -> AgentState:
    lang = lang if lang in prompts.SYSTEM else "kk"
    msgs: list[BaseMessage] = [SystemMessage(content=_system_prompt(lang, farm_id))]
    for h in (history or [])[-6:]:  # қысқа диалог контексі (тек мәтін)
        if h.get("role") == "user":
            msgs.append(HumanMessage(content=h.get("content", "")))
        elif h.get("role") == "assistant":
            msgs.append(AIMessage(content=h.get("content", "")))
    msgs.append(HumanMessage(content=question))
    return {"messages": msgs, "question": question, "lang": lang, "farm_id": farm_id, "tool_outputs": [],
            "trace": [], "retries": 0, "rounds": 0, "tokens": 0}


def _result(state: dict, t0: float, session_id: str) -> dict:
    return {
        "session_id": session_id,
        "answer": state.get("final", ""),
        "status": state.get("status", "error"),
        "gate": state.get("gate"),
        "trace": state.get("trace", []),
        "artifacts": [_artifact(o) for o in state.get("tool_outputs", []) if "error" not in o],
        "tools_called": [o.get("tool") for o in state.get("tool_outputs", [])],
        "latency_s": round(time.time() - t0, 2),
        "tokens": state.get("tokens", 0),
    }


def _artifact(out: dict) -> dict:
    """Интерфейс үшін: құрал нәтижесінен график/кесте/файл."""
    keep = ("tool", "farm", "summary", "series", "schedule", "files", "table", "metrics", "periods",
            "low_wind_windows", "factors", "groups", "baseline", "scenario", "unit", "capacity_mw", "model")
    return {k: out[k] for k in keep if k in out}


def ask(question: str, lang: str = "kk", farm_id: str = "kelmarsh", session_id: str | None = None,
        history: list[dict] | None = None, log: bool = True) -> dict:
    session_id = session_id or uuid.uuid4().hex[:12]
    t0 = time.time()
    try:
        state = get_graph().invoke(initial_state(question, lang, farm_id, history), {"recursion_limit": 40})
    except LLMNotConfigured as e:
        state = {"final": prompts.LLM_ERROR[lang if lang in prompts.LLM_ERROR else "kk"].format(err=e), "status": "error", "trace": []}
    except Exception as e:  # API қатесі және т.б.
        state = {"final": prompts.LLM_ERROR[lang if lang in prompts.LLM_ERROR else "kk"].format(err=f"{type(e).__name__}: {e}"),
                 "status": "error", "trace": []}
    res = _result(state, t0, session_id)
    if log:
        log_interaction(res, question, lang, "agent")
    return res


def stream(question: str, lang: str = "kk", farm_id: str = "kelmarsh", session_id: str | None = None,
           history: list[dict] | None = None):
    """SSE үшін: әр қадамнан кейін оқиға шығарады, соңында толық нәтиже."""
    session_id = session_id or uuid.uuid4().hex[:12]
    t0 = time.time()
    try:
        graph = get_graph()
    except LLMNotConfigured as e:
        res = _result({"final": prompts.LLM_ERROR[lang].format(err=e), "status": "error"}, t0, session_id)
        yield {"event": "final", "data": res}
        return
    acc: dict[str, Any] = dict(initial_state(question, lang, farm_id, history))
    try:
        for upd in graph.stream(acc, {"recursion_limit": 40}, stream_mode="updates"):
            for node, delta in upd.items():
                if not delta:
                    continue
                for ev in delta.get("trace", []):
                    yield {"event": "trace", "data": {"node": node, **_jsonable(ev)}}
                for k, v in delta.items():
                    if k in ("trace", "tool_outputs"):
                        acc[k] = acc.get(k, []) + v
                    elif k != "messages":
                        acc[k] = v
    except Exception as e:
        acc.update(final=prompts.LLM_ERROR[lang].format(err=f"{type(e).__name__}: {e}"), status="error")
    res = _result(acc, t0, session_id)
    log_interaction(res, question, lang, "agent")
    yield {"event": "final", "data": res}


def _jsonable(x):
    return json.loads(json.dumps(x, ensure_ascii=False, default=str))


def log_interaction(res: dict, question: str, lang: str, system: str) -> None:
    try:
        from app.core.db import session
        from app.db.models import AgentLog

        faith = None
        for c in (res.get("gate") or {}).get("checks", []):
            if c.get("check") == "faithfulness":
                faith = c.get("score")
        with session() as s:
            s.add(AgentLog(session_id=res["session_id"], lang=lang, question=question, tools_called=res.get("tools_called"),
                           answer=res.get("answer", ""), gate_result=_jsonable(res.get("gate")), faithfulness=faith,
                           latency_s=res.get("latency_s"), tokens=res.get("tokens"), system=system))
    except Exception:
        pass


def export_graph_png(path) -> bool:
    """Сурет 13 үшін графтың Mermaid сипаттамасы (.mmd). PNG-ні figures.py өзі салады
    (сыртқы рендер-қызметтерге сұраныс жіберілмейді)."""
    from pathlib import Path

    class _Dummy:
        def bind_tools(self, tools):
            return self

    g = build_graph(llm=_Dummy())
    Path(path).with_suffix(".mmd").write_text(g.get_graph().draw_mermaid(), encoding="utf-8")
    return True
