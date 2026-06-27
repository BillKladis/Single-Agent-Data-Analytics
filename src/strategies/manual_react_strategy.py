"""
Strategy B - Manual ReAct loop (raw Anthropic SDK, no framework).

A hand-rolled reason/act/observe loop directly on the Anthropic Messages API.
Implementing the loop by hand makes the mechanics explicit and gives exact,
unmediated token accounting from `response.usage`, which is the cleanest
baseline to compare framework overhead against.
"""
from __future__ import annotations

import json
import time

from anthropic import Anthropic

from src.strategies.base import SYSTEM_PROMPT, StrategyResult
from src.tools import ANALYST_TOOLS


def _to_anthropic_schema(tool) -> dict:
    """Convert a LangChain tool into an Anthropic tool spec."""
    schema = tool.get_input_schema().model_json_schema()
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return {
        "name": tool.name,
        "description": (tool.description or "").strip(),
        "input_schema": {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        },
    }


class ManualReActStrategy:
    name = "Manual ReAct (raw SDK)"
    short = "manual_react"

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6",
                 max_steps: int = 10):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_steps = max_steps
        self.tool_specs = [_to_anthropic_schema(t) for t in ANALYST_TOOLS]
        self.registry = {t.name: t for t in ANALYST_TOOLS}

    def run(self, question: str) -> StrategyResult:
        messages = [{"role": "user", "content": question}]
        calls: list[dict] = []
        in_tok = out_tok = n_calls = 0
        t0 = time.time()
        try:
            for _ in range(self.max_steps):
                resp = self.client.messages.create(
                    model=self.model, max_tokens=4096, system=SYSTEM_PROMPT,
                    tools=self.tool_specs, messages=messages,
                )
                n_calls += 1
                in_tok += resp.usage.input_tokens
                out_tok += resp.usage.output_tokens
                messages.append(
                    {"role": "assistant", "content": [b.model_dump() for b in resp.content]}
                )
                tool_uses = [b for b in resp.content if b.type == "tool_use"]

                if resp.stop_reason != "tool_use" or not tool_uses:
                    text = "".join(b.text for b in resp.content if b.type == "text")
                    return StrategyResult(
                        answer=text.strip() or "No answer generated.",
                        tool_calls=calls, latency_s=time.time() - t0,
                        input_tokens=in_tok, output_tokens=out_tok, n_llm_calls=n_calls,
                    )

                results = []
                for tu in tool_uses:
                    try:
                        out = self.registry[tu.name].invoke(tu.input)
                    except Exception as e:  # noqa: BLE001
                        out = f"Tool error: {e}"
                    calls.append({"tool": tu.name, "args": json.dumps(tu.input),
                                  "output_snippet": str(out)[:400]})
                    results.append({"type": "tool_result", "tool_use_id": tu.id,
                                    "content": str(out)})
                messages.append({"role": "user", "content": results})

            return StrategyResult(
                answer="Reached max reasoning steps without a final answer.",
                tool_calls=calls, latency_s=time.time() - t0,
                input_tokens=in_tok, output_tokens=out_tok, n_llm_calls=n_calls,
            )
        except Exception as e:  # noqa: BLE001
            return StrategyResult(answer="", tool_calls=calls,
                                  latency_s=time.time() - t0, error=str(e))
