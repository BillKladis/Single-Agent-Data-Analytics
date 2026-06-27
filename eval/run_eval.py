"""
Run the comparative evaluation: every strategy against every benchmark task.

Outputs:
  eval/results/per_task.csv      - one row per (strategy, task)
  eval/results/summary.csv       - aggregated metrics per strategy
  eval/results/summary.json      - same, machine-readable
  eval/results/comparison.png    - 4-panel metric comparison
  eval/results/accuracy_by_task.png - per-task correctness heatmap

Usage:  python -m eval.run_eval        (needs ANTHROPIC_API_KEY in env/.env)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from eval.benchmark import build_benchmark
from eval.metrics import aggregate, tool_scores
from src import tools as tool_store
from src import viz
from src.data_gen import load_or_generate
from src.strategies import STRATEGIES, STRATEGY_LABELS

RESULTS_DIR = Path("eval/results")


def run(model: str = "claude-sonnet-4-6") -> None:
    load_dotenv()
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key or key == "your-key-here":
        print("ERROR: set ANTHROPIC_API_KEY in .env"); sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_or_generate("data/superstore.csv")
    tool_store.init_tools(df)
    tasks = build_benchmark(df)
    print(f"Benchmark: {len(tasks)} tasks x {len(STRATEGIES)} strategies "
          f"= {len(tasks) * len(STRATEGIES)} runs\n")

    per_task: list[dict] = []
    agents = {short: cls(key, model=model) for short, cls in STRATEGIES.items()}

    for short, agent in agents.items():
        label = STRATEGY_LABELS[short]
        print(f"--- {label} ---")
        for task in tasks:
            tool_store.reset_run()
            res = agent.run(task.question)
            correct = bool(res.answer) and task.checker(res.answer)
            ts = tool_scores(res.tools_used, task.expected_tools)
            per_task.append({
                "strategy": short,
                "strategy_label": label,
                "task": task.id,
                "correct": int(correct),
                "tool_f1": ts["f1"],
                "tool_recall": ts["recall"],
                "tool_precision": ts["precision"],
                "total_tokens": res.total_tokens,
                "latency_s": round(res.latency_s, 2),
                "n_llm_calls": res.n_llm_calls,
                "n_tool_calls": len(res.tool_calls),
                "tools_used": "|".join(res.tools_used),
                "error": res.error or "",
            })
            flag = "OK " if correct else "XX "
            print(f"  {flag} {task.id:22s} f1={ts['f1']:.2f} "
                  f"tok={res.total_tokens:5d} {res.latency_s:5.1f}s")
        print()

    pt = pd.DataFrame(per_task)
    pt.to_csv(RESULTS_DIR / "per_task.csv", index=False)

    summary = {}
    for short in STRATEGIES:
        rows = [r for r in per_task if r["strategy"] == short]
        summary[short] = {"label": STRATEGY_LABELS[short], **aggregate(rows)}
    sdf = pd.DataFrame(summary).T
    sdf.to_csv(RESULTS_DIR / "summary.csv")
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("=== SUMMARY ===")
    cols = ["accuracy", "tool_f1", "total_tokens", "latency_s", "n_llm_calls"]
    print(sdf[cols].round(3).to_string())

    _plot_comparison(summary)
    _plot_task_heatmap(pt)
    print(f"\nWrote results + charts to {RESULTS_DIR}/")


def _plot_comparison(summary: dict) -> None:
    viz.apply_theme()
    labels = [summary[s]["label"] for s in summary]
    colors = viz.SEQUENCE[: len(labels)]
    panels = [
        ("accuracy", "Answer accuracy", "{:.0%}", 1),
        ("tool_f1", "Tool-selection F1", "{:.2f}", 1),
        ("total_tokens", "Mean tokens / question", "{:,.0f}", None),
        ("latency_s", "Mean latency (s)", "{:.1f}", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, title, fmt, ymax) in zip(axes.flat, panels):
        vals = [summary[s][key] for s in summary]
        bars = ax.bar(labels, vals, color=colors, alpha=0.9, width=0.6)
        ax.set_title(title)
        if ymax:
            ax.set_ylim(0, ymax * 1.08)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=12, ha="right", fontsize=9)
    fig.suptitle("Agent architecture comparison (curated analytics toolbox)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(RESULTS_DIR / "comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_task_heatmap(pt: pd.DataFrame) -> None:
    viz.apply_theme()
    import seaborn as sns
    grid = pt.pivot(index="task", columns="strategy_label", values="correct")
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(grid) + 2))
    sns.heatmap(grid, annot=True, fmt=".0f", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.6, linecolor="white", cbar=False, ax=ax)
    ax.set_title("Per-task correctness (1 = correct)")
    ax.set_xlabel(""); ax.set_ylabel("benchmark task")
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "accuracy_by_task.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
