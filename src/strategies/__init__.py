"""Agent strategy registry for the comparative study."""
from src.strategies.base import StrategyResult, SYSTEM_PROMPT
from src.strategies.langgraph_strategy import LangGraphStrategy
from src.strategies.manual_react_strategy import ManualReActStrategy
from src.strategies.plan_execute_strategy import PlanExecuteStrategy

STRATEGIES = {
    LangGraphStrategy.short: LangGraphStrategy,
    ManualReActStrategy.short: ManualReActStrategy,
    PlanExecuteStrategy.short: PlanExecuteStrategy,
}

STRATEGY_LABELS = {
    LangGraphStrategy.short: LangGraphStrategy.name,
    ManualReActStrategy.short: ManualReActStrategy.name,
    PlanExecuteStrategy.short: PlanExecuteStrategy.name,
}

__all__ = [
    "STRATEGIES", "STRATEGY_LABELS", "StrategyResult", "SYSTEM_PROMPT",
    "LangGraphStrategy", "ManualReActStrategy", "PlanExecuteStrategy",
]
