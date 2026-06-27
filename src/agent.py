"""
LangChain 1.x + LangGraph tool-calling agent backed by Claude.

LangChain 1.x wraps LangGraph: create_agent() returns a CompiledStateGraph
that implements the full ReAct loop (Thought → Act → Observe) via Anthropic's
native tool-use API.  A ToolCallTracer callback records every invocation so
the UI can render a transparent reasoning trace.
"""
from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from src.tools import ANALYST_TOOLS

SYSTEM_PROMPT = """\
You are a precise, evidence-driven data analytics assistant.
You have access to a fixed set of analyst tools that operate over a loaded business dataset.

Rules:
1. Always call at least one tool before writing your final answer.
2. Call multiple tools when the question requires composing information.
3. Your final answer must cite which tools you called and what key numbers they returned.
4. Be specific: include numbers, percentages, and time periods from the tool outputs.
5. Never fabricate data. If a tool returns insufficient information, say so clearly.

Available dataset columns (Superstore synthetic dataset):
  Order_ID, Order_Date, Ship_Date, Ship_Mode, Customer_ID, Segment,
  Region, State, Category, Sub_Category, Sales, Quantity, Discount, Profit
"""


class ToolCallTracer(BaseCallbackHandler):
    """Records each tool invocation for the UI reasoning trace panel."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        self.calls.append(
            {
                "tool": serialized.get("name", "unknown"),
                "args": input_str,
                "output_snippet": None,
            }
        )

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        if self.calls and self.calls[-1]["output_snippet"] is None:
            self.calls[-1]["output_snippet"] = str(output)[:400]


def _extract_answer(messages: list[BaseMessage]) -> str:
    """Pull the final text answer from the agent's message list."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                text_parts = [
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                joined = "\n".join(text_parts).strip()
                if joined:
                    return joined
    return "No answer generated."


def build_agent(api_key: str):
    """Return a compiled LangGraph agent ready to invoke."""
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        anthropic_api_key=api_key,
        max_tokens=4096,
        temperature=0,
    )
    return create_agent(llm, ANALYST_TOOLS, system_prompt=SYSTEM_PROMPT)


def run_query(
    agent,
    question: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Run a question through the agent and return (answer, tool_call_trace)."""
    tracer = ToolCallTracer()
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"callbacks": [tracer]},
    )
    messages: list[BaseMessage] = result.get("messages", [])
    answer = _extract_answer(messages)
    return answer, tracer.calls
