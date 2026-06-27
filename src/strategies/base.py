"""Shared interface and result type for all agent strategies."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# A single system prompt shared by every architecture so the comparison
# isolates the *control flow*, not prompt wording.
SYSTEM_PROMPT = """\
You are a precise, evidence-driven data analytics assistant.
You answer business questions about a loaded dataset strictly by calling the
provided analyst tools - never by guessing numbers.

Rules:
1. Call at least one tool before answering. Compose several when needed.
2. Ground every claim in tool output: cite the tools you used and the key numbers.
3. Be specific - include figures, percentages, p-values, and periods.
4. Never fabricate data. If the tools cannot answer, say so plainly.

Dataset columns (synthetic Superstore):
  Order_ID, Order_Date, Ship_Date, Ship_Mode, Customer_ID, Segment,
  Region, State, Category, Sub_Category, Sales, Quantity, Discount, Profit
"""


@dataclass
class StrategyResult:
    """Uniform record returned by every strategy run."""

    answer: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    n_llm_calls: int = 0
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def tools_used(self) -> list[str]:
        return [c["tool"] for c in self.tool_calls]
