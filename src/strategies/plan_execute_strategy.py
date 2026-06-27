"""
Strategy C - Plan-then-Execute (decoupled planning and execution).

Unlike ReAct, which interleaves a model call before every tool, this
architecture commits to a full tool plan up front in a single call, executes
the whole plan deterministically, then makes one synthesis call over all
observations. It trades adaptivity for fewer LLM round-trips - a meaningful
contrast in the cost/accuracy study.
"""
from __future__ import annotations

import json
import re
import time

from anthropic import Anthropic

from src.strategies.base import SYSTEM_PROMPT, StrategyResult
from src.tools import ANALYST_TOOLS


def _tool_catalog() -> str:
    lines = []
    for t in ANALYST_TOOLS:
        schema = t.get_input_schema().model_json_schema()
        props = ", ".join(schema.get("properties", {}).keys()) or "(none)"
        desc = (t.description or "").strip().split("\n")[0]
        lines.append(f"- {t.name}({props}): {desc}")
    return "\n".join(lines)


PLAN_INSTRUCTION = """\
You are planning how to answer a business question using analyst tools.
Given the question, output a JSON array of the tool calls needed - in order.
Each element must be: {{"tool": "<tool_name>", "args": {{...}}}}.
Use only these tools:

{catalog}

Output ONLY the JSON array, no prose. Keep the plan minimal but sufficient.
Question: {question}
"""


class PlanExecuteStrategy:
    name = "Plan-then-Execute"
    short = "plan_execute"

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6",
                 max_plan_steps: int = 6):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_plan_steps = max_plan_steps
        self.registry = {t.name: t for t in ANALYST_TOOLS}
        self.catalog = _tool_catalog()

    def run(self, question: str) -> StrategyResult:
        calls: list[dict] = []
        in_tok = out_tok = n_calls = 0
        t0 = time.time()
        try:
            # 1) Plan
            plan_msg = PLAN_INSTRUCTION.format(catalog=self.catalog, question=question)
            plan_resp = self.client.messages.create(
                model=self.model, max_tokens=1024,
                messages=[{"role": "user", "content": plan_msg}],
            )
            n_calls += 1
            in_tok += plan_resp.usage.input_tokens
            out_tok += plan_resp.usage.output_tokens
            plan_text = "".join(b.text for b in plan_resp.content if b.type == "text")
            plan = _parse_plan(plan_text)[: self.max_plan_steps]

            # 2) Execute deterministically
            observations = []
            for step in plan:
                name = step.get("tool")
                args = step.get("args", {}) or {}
                if name not in self.registry:
                    continue
                try:
                    out = self.registry[name].invoke(args)
                except Exception as e:  # noqa: BLE001
                    out = f"Tool error: {e}"
                calls.append({"tool": name, "args": json.dumps(args),
                              "output_snippet": str(out)[:400]})
                observations.append(f"[{name}({json.dumps(args)})]\n{out}")

            if not observations:
                observations.append("(planner produced no executable tool calls)")

            # 3) Synthesize
            synth = (
                f"Question: {question}\n\n"
                f"Tool observations:\n" + "\n\n".join(observations) +
                "\n\nWrite the final evidence-based answer. Cite the tools used "
                "and the key numbers. Do not invent data."
            )
            synth_resp = self.client.messages.create(
                model=self.model, max_tokens=2048, system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": synth}],
            )
            n_calls += 1
            in_tok += synth_resp.usage.input_tokens
            out_tok += synth_resp.usage.output_tokens
            answer = "".join(b.text for b in synth_resp.content if b.type == "text")

            return StrategyResult(
                answer=answer.strip() or "No answer generated.", tool_calls=calls,
                latency_s=time.time() - t0, input_tokens=in_tok, output_tokens=out_tok,
                n_llm_calls=n_calls,
            )
        except Exception as e:  # noqa: BLE001
            return StrategyResult(answer="", tool_calls=calls,
                                  latency_s=time.time() - t0, error=str(e))


def _parse_plan(text: str) -> list[dict]:
    """Extract a JSON array of tool calls from the planner output."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
        return [d for d in data if isinstance(d, dict) and "tool" in d]
    except json.JSONDecodeError:
        return []
