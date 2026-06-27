"""
Gold benchmark for the agent comparison.

Each task pairs a business question with (a) the set of analyst tools a correct
approach should use and (b) a programmatic answer checker whose ground truth is
computed directly from the dataframe - so the benchmark is self-validating and
cannot drift from the data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass
class BenchmarkTask:
    id: str
    question: str
    expected_tools: set[str]
    checker: Callable[[str], bool]
    rationale: str


def _norm(text: str) -> str:
    return text.replace(",", "").lower()


def _has_number(answer: str, value: float, rel_tol: float = 0.02) -> bool:
    """True if a number within rel_tol of `value` appears in the answer."""
    a = _norm(answer)
    for m in re.findall(r"-?\d+(?:\.\d+)?", a):
        try:
            num = float(m)
        except ValueError:
            continue
        if value == 0:
            if abs(num) < 1e-9:
                return True
        elif abs(num - value) <= abs(value) * rel_tol:
            return True
    return False


def _contains_all(answer: str, *terms: str) -> bool:
    low = answer.lower()
    return all(t.lower() in low for t in terms)


def build_benchmark(df: pd.DataFrame) -> list[BenchmarkTask]:
    # ----- ground truth computed from the data -----
    region_profit = df.groupby("Region")["Profit"].sum()
    best_region = region_profit.idxmax()

    n_high_disc = int((df["Discount"] > 0.3).sum())

    sub_sales = df.groupby("Sub_Category")["Sales"].sum()
    top_sub = sub_sales.idxmax()

    seg_matrix = df.pivot_table(index="Category", columns="Region",
                                values="Profit", aggfunc="sum").stack()
    worst_cat, worst_region = seg_matrix.idxmin()

    seg_sales = df.groupby("Segment")["Sales"].mean()
    best_segment = seg_sales.idxmax()

    q1, q3 = df["Profit"].quantile(0.25), df["Profit"].quantile(0.75)
    iqr = q3 - q1
    n_outliers = int(((df["Profit"] < q1 - 1.5 * iqr) |
                      (df["Profit"] > q3 + 1.5 * iqr)).sum())

    n_rows = len(df)
    n_cols = df.shape[1]

    tasks = [
        BenchmarkTask(
            "region_profit",
            "Which region has the highest total profit?",
            {"group_compare"},
            lambda a: best_region.lower() in a.lower(),
            f"Ground truth: {best_region}",
        ),
        BenchmarkTask(
            "discount_corr",
            "Is discount hurting profit? Give the correlation coefficient.",
            {"correlate"},
            lambda a: ("negativ" in a.lower()) and _has_number(a, -0.60, rel_tol=0.15),
            "Ground truth: r ~ -0.60, negative",
        ),
        BenchmarkTask(
            "high_discount_count",
            "How many orders have a discount greater than 0.3?",
            {"filter_count"},
            lambda a: _has_number(a, n_high_disc, rel_tol=0.0),
            f"Ground truth: {n_high_disc}",
        ),
        BenchmarkTask(
            "top_subcategory",
            "Which product sub-category generates the most total sales?",
            {"top_n", "pareto_analysis"},
            lambda a: top_sub.lower() in a.lower(),
            f"Ground truth: {top_sub}",
        ),
        BenchmarkTask(
            "category_anova",
            "Does average profit differ significantly across categories? Give a p-value.",
            {"statistical_test"},
            lambda a: ("significant" in a.lower()
                       and "not statistically significant" not in a.lower()
                       and "not significant" not in a.lower()),
            "Ground truth: ANOVA p < 0.001, significant",
        ),
        BenchmarkTask(
            "segment_loss",
            "Across category-by-region segments, which one loses the most money?",
            {"segment_profitability"},
            lambda a: _contains_all(a, worst_cat, worst_region),
            f"Ground truth: {worst_cat} / {worst_region}",
        ),
        BenchmarkTask(
            "pareto_sales",
            "Do a few sub-categories drive most of the sales (80/20 rule)?",
            {"pareto_analysis"},
            lambda a: _has_number(a, 80, rel_tol=0.10) or _has_number(a, 82, rel_tol=0.08),
            "Ground truth: ~80% concentrated in top sub-categories",
        ),
        BenchmarkTask(
            "seasonality",
            "What is the monthly sales trend - is there seasonality?",
            {"trend_over_time"},
            lambda a: any(k in a.lower() for k in
                          ("season", "q4", "december", "fourth quarter", "year-end", "holiday")),
            "Ground truth: Q4 / December seasonal peak",
        ),
        BenchmarkTask(
            "regression_r2",
            "Predict profit from sales, discount and quantity. What is the R-squared?",
            {"linear_regression"},
            lambda a: _has_number(a, 0.39, rel_tol=0.20),
            "Ground truth: R^2 ~ 0.39",
        ),
        BenchmarkTask(
            "profit_outliers",
            "How many outliers are in the Profit column using the IQR method?",
            {"detect_outliers"},
            lambda a: _has_number(a, n_outliers, rel_tol=0.0),
            f"Ground truth: {n_outliers}",
        ),
        BenchmarkTask(
            "segment_avg_sales",
            "Which customer segment has the highest average sales?",
            {"group_compare"},
            lambda a: best_segment.lower() in a.lower(),
            f"Ground truth: {best_segment}",
        ),
        BenchmarkTask(
            "describe_shape",
            "Describe the dataset: how many rows and columns does it have?",
            {"describe_dataset"},
            lambda a: _has_number(a, n_rows, rel_tol=0.0) and _has_number(a, n_cols, rel_tol=0.0),
            f"Ground truth: {n_rows} rows x {n_cols} cols",
        ),
    ]
    return tasks
