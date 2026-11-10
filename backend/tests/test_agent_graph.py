"""LangGraph агентінің логикасы — сценарийлік «жалған» LLM-мен (API кілтінсіз)."""
import pytest
from langchain_core.messages import AIMessage

from app.agent import graph
from app.models.registry import bundle_path

pytestmark = pytest.mark.skipif(not bundle_path("kelmarsh").exists(), reason="модельдер оқытылмаған")


class ScriptedLLM:
    """Алдын ала берілген жауаптарды кезекпен қайтарады."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        r = self.replies.pop(0)
        return r(messages) if callable(r) else r


def _tool_call(name, args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}"}])


def _answer_from_tool(messages):
    import json

    tool_msg = [m for m in messages if m.type == "tool"][-1]
    m = json.loads(tool_msg.content)["metrics"]
    return AIMessage(content=f"Ансамбльдің 24 сағаттағы nMAE мәні {m['nmae_pct']}%, MAE {m['mae_mw']} МВт.")


def _run(llm, question="Ансамбль дәлдігі қандай?"):
    g = graph.build_graph(llm=llm)
    return g.invoke(graph.initial_state(question, "kk", "kelmarsh"), {"recursion_limit": 30})


def test_grounded_answer_passes_gate():
    llm = ScriptedLLM([_tool_call("backtest", {"farm_id": "kelmarsh", "model": "ensemble", "horizon": 24}), _answer_from_tool])
    st = _run(llm)
    assert st["status"] == "ok"
    assert st["gate"]["passed"]
    assert [t["step"] for t in st["trace"]] == ["llm", "tool", "llm", "gate"]


def test_fabricated_number_triggers_retry_then_safe_fail():
    bad = AIMessage(content="Ансамбльдің дәлдігі 17.3% және ертең 1234 МВт·сағ өндіріледі.")
    llm = ScriptedLLM([_tool_call("backtest", {"farm_id": "kelmarsh", "model": "ensemble", "horizon": 24}), bad, bad, bad])
    st = _run(llm)
    assert st["status"] == "safe_fail"
    assert "17.3" not in st["final"] and "1234" not in st["final"]
    assert sum(1 for t in st["trace"] if t["step"] == "gate") == 3  # 1 + 2 қайталау


def test_retry_can_recover():
    bad = AIMessage(content="nMAE шамамен 17.3%.")
    llm = ScriptedLLM([_tool_call("backtest", {"farm_id": "kelmarsh", "model": "ensemble", "horizon": 24}), bad, _answer_from_tool])
    st = _run(llm)
    assert st["status"] == "ok" and st["retries"] == 1


def test_tool_error_is_reported_honestly():
    reply = AIMessage(content="Кешіріңіз, Ерейментаудың SCADA деректері ашық емес, сондықтан тарихи өндірісті көрсете алмаймын.")
    llm = ScriptedLLM([_tool_call("get_history", {"farm_id": "ereymentau"}), reply])
    st = _run(llm, "Ерейментау өткен айда қанша өндірді?")
    assert st["status"] == "ok"
    tool_ev = [t for t in st["trace"] if t["step"] == "tool"][0]
    assert tool_ev["error"]
