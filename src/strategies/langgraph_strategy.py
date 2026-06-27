"""
Strategy A - LangGraph ReAct agent (framework-managed control flow).

LangChain 1.x `create_agent` compiles a LangGraph state machine that runs the
canonical ReAct loop (reason -> act -> observe) over Anthropic's tool-use API.
This is the production-grade, batteries-included baseline.
"""
from __future__ import annotations

import time

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.strategies.base import SYSTEM_PROMPT, StrategyResult
from src.tools import ANALYST_TOOLS


class LangGraphStrategy:
    name = "LangGraph ReAct"
    short = "langgraph"

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        self.llm = ChatAnthropic(
            model=model, anthropic_api_key=api_key, max_tokens=4096, temperature=0
        )
        self.agent = create_agent(self.llm, ANALYST_TOOLS, system_prompt=SYSTEM_PROMPT)

    def run(self, question: str) -> StrategyResult:
        t0 = time.time()
        try:
            result = self.agent.invoke(
                {"messages": [HumanMessage(content=question)]},
                config={"recursion_limit": 25},
            )
        except Exception as e:  # noqa: BLE001
            return StrategyResult(answer="", latency_s=time.time() - t0, error=str(e))

        messages = result.get("messages", [])
        calls: list[dict] = []
        in_tok = out_tok = n_calls = 0
        pending: dict[str, dict] = {}

        for msg in messages:
            if isinstance(msg, AIMessage):
                n_calls += 1
                um = msg.usage_metadata or {}
                in_tok += um.get("input_tokens", 0)
                out_tok += um.get("output_tokens", 0)
                for tc in msg.tool_calls or []:
                    rec = {"tool": tc["name"], "args": _fmt_args(tc["args"]),
                           "output_snippet": None}
                    calls.append(rec)
                    pending[tc["id"]] = rec
            elif isinstance(msg, ToolMessage):
                rec = pending.get(msg.tool_call_id)
                if rec is not None:
                    rec["output_snippet"] = str(msg.content)[:400]

        answer = _extract_answer(messages)
        return StrategyResult(
            answer=answer, tool_calls=calls, latency_s=time.time() - t0,
            input_tokens=in_tok, output_tokens=out_tok, n_llm_calls=n_calls,
        )


def _fmt_args(args: dict) -> str:
    import json
    try:
        return json.dumps(args)
    except TypeError:
        return str(args)


def _extract_answer(messages) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                parts = [p.get("text", "") for p in content
                         if isinstance(p, dict) and p.get("type") == "text"]
                joined = "\n".join(parts).strip()
                if joined:
                    return joined
    return "No answer generated."
