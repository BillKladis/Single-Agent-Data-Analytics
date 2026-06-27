"""
Analyst toolbox — 7 curated pandas operations exposed as LangChain tools.

Each tool returns a plain-text summary for the agent to reason over and
registers any generated figure in a module-level store the UI reads after
the agent finishes its run.
"""
import io
import json
import operator as op
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool

sns.set_theme(style="whitegrid", palette="muted")

# ---------------------------------------------------------------------------
# Module-level state shared with the Streamlit UI
# ---------------------------------------------------------------------------
_df: Optional[pd.DataFrame] = None
_figures: list[bytes] = []
_figure_labels: list[str] = []


def init_tools(df: pd.DataFrame) -> None:
    global _df
    _df = df


def reset_run() -> None:
    global _figures, _figure_labels
    _figures = []
    _figure_labels = []


def get_figures() -> list[tuple[str, bytes]]:
    return list(zip(_figure_labels, _figures))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _save_fig(fig: plt.Figure, label: str) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    _figures.append(buf.read())
    _figure_labels.append(label)
    plt.close(fig)


def _require_df() -> pd.DataFrame:
    if _df is None:
        raise RuntimeError("Dataset not loaded. Call init_tools(df) first.")
    return _df


OPERATORS = {
    ">": op.gt, "<": op.lt, ">=": op.ge, "<=": op.le,
    "==": op.eq, "!=": op.ne,
}


# ---------------------------------------------------------------------------
# Tool 1: describe_dataset
# ---------------------------------------------------------------------------
@tool
def describe_dataset() -> str:
    """Return shape, column names, data types, and missingness summary of the dataset."""
    df = _require_df()
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    info_rows = []
    for col in df.columns:
        info_rows.append(
            f"  {col}: dtype={df[col].dtype}, nulls={missing[col]} ({missing_pct[col]}%)"
        )
    result = (
        f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n"
        f"Columns:\n" + "\n".join(info_rows) + "\n"
        f"\nNumeric summary:\n{df.describe(include='number').round(2).to_string()}"
    )
    return result


# ---------------------------------------------------------------------------
# Tool 2: investigate_distribution
# ---------------------------------------------------------------------------
@tool
def investigate_distribution(column: str) -> str:
    """Return summary statistics and a histogram or value-count breakdown for one column.

    Args:
        column: The column name to investigate.
    """
    df = _require_df()
    if column not in df.columns:
        return f"Column '{column}' not found. Available: {list(df.columns)}"

    series = df[column].dropna()
    fig, ax = plt.subplots(figsize=(7, 4))

    if pd.api.types.is_numeric_dtype(series):
        stats = series.describe().round(4)
        skew = round(float(series.skew()), 4)
        ax.hist(series, bins=30, edgecolor="white", color="#4C72B0", alpha=0.85)
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        _save_fig(fig, f"Distribution of {column}")
        result = f"Numeric distribution — {column}:\n{stats.to_string()}\nskewness={skew}"
    else:
        counts = series.value_counts().head(15)
        ax.barh(counts.index[::-1], counts.values[::-1], color="#4C72B0", alpha=0.85)
        ax.set_title(f"Value counts — {column}")
        ax.set_xlabel("Count")
        _save_fig(fig, f"Value counts — {column}")
        result = f"Categorical distribution — {column} (top values):\n{counts.to_string()}"

    return result


# ---------------------------------------------------------------------------
# Tool 3: group_compare
# ---------------------------------------------------------------------------
@tool
def group_compare(metric_column: str, group_by_column: str, agg: str) -> str:
    """Compute an aggregated metric grouped by a categorical column and produce a bar chart.

    Args:
        metric_column: Numeric column to aggregate (e.g. 'Sales', 'Profit').
        group_by_column: Categorical column to group by (e.g. 'Region', 'Category').
        agg: Aggregation — one of 'mean', 'sum', 'median', 'count'.
    """
    df = _require_df()
    for col in [metric_column, group_by_column]:
        if col not in df.columns:
            return f"Column '{col}' not found. Available: {list(df.columns)}"
    if agg not in ("mean", "sum", "median", "count"):
        return f"agg must be one of mean/sum/median/count, got '{agg}'."

    grouped = df.groupby(group_by_column)[metric_column].agg(agg).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(grouped) + 1)))
    colors = ["#d65f5f" if v < 0 else "#4C72B0" for v in grouped.values]
    ax.barh(grouped.index.astype(str), grouped.values, color=colors, alpha=0.88)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_title(f"{agg.capitalize()} of {metric_column} by {group_by_column}")
    ax.set_xlabel(f"{agg}({metric_column})")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    _save_fig(fig, f"{agg}({metric_column}) by {group_by_column}")

    return (
        f"{agg}({metric_column}) grouped by {group_by_column}:\n"
        + grouped.sort_values(ascending=False).round(2).to_string()
    )


# ---------------------------------------------------------------------------
# Tool 4: correlate
# ---------------------------------------------------------------------------
@tool
def correlate(column_a: str, column_b: str) -> str:
    """Compute the Pearson correlation between two numeric columns and generate a scatter plot.

    Args:
        column_a: First numeric column (e.g. 'Discount').
        column_b: Second numeric column (e.g. 'Profit').
    """
    df = _require_df()
    for col in [column_a, column_b]:
        if col not in df.columns:
            return f"Column '{col}' not found. Available: {list(df.columns)}"

    pair = df[[column_a, column_b]].dropna()
    r = round(float(pair[column_a].corr(pair[column_b])), 4)
    n = len(pair)

    sample = pair.sample(min(1500, n), random_state=42)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(sample[column_a], sample[column_b], alpha=0.25, s=18, color="#4C72B0")
    m, b = np.polyfit(pair[column_a], pair[column_b], 1)
    x_line = np.linspace(pair[column_a].min(), pair[column_a].max(), 100)
    ax.plot(x_line, m * x_line + b, color="#d65f5f", linewidth=1.8, label=f"r = {r}")
    ax.set_xlabel(column_a)
    ax.set_ylabel(column_b)
    ax.set_title(f"Correlation: {column_a} vs {column_b}")
    ax.legend()
    _save_fig(fig, f"Correlation: {column_a} vs {column_b}")

    strength = (
        "strong" if abs(r) > 0.6 else "moderate" if abs(r) > 0.3 else "weak"
    )
    direction = "positive" if r > 0 else "negative"
    return (
        f"Pearson r = {r} (n={n:,})\n"
        f"Interpretation: {strength} {direction} linear relationship between "
        f"{column_a} and {column_b}.\n"
        f"OLS slope: {round(m, 4)} — a 1-unit increase in {column_a} is "
        f"associated with a {round(m, 2):+.2f} change in {column_b}."
    )


# ---------------------------------------------------------------------------
# Tool 5: top_n
# ---------------------------------------------------------------------------
@tool
def top_n(column: str, n: int, by: str) -> str:
    """Return the top N rows ranked by a specific column.

    Args:
        column: Column to display in output (e.g. 'Sub_Category').
        n: Number of rows to return (1–20).
        by: Column to rank by (e.g. 'Profit', 'Sales').
    """
    df = _require_df()
    for col in [column, by]:
        if col not in df.columns:
            return f"Column '{col}' not found. Available: {list(df.columns)}"
    n = max(1, min(n, 20))

    if column == by:
        result_df = df[[column]].nlargest(n, by).reset_index(drop=True)
    else:
        result_df = df[[column, by]].nlargest(n, by).reset_index(drop=True)

    if pd.api.types.is_numeric_dtype(result_df[by]):
        grouped = df.groupby(column)[by].sum().nlargest(n)
        fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * n + 1)))
        ax.barh(grouped.index[::-1].astype(str), grouped.values[::-1], color="#4C72B0", alpha=0.88)
        ax.set_title(f"Top {n} {column} by {by}")
        ax.set_xlabel(by)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        _save_fig(fig, f"Top {n} {column} by {by}")

    return f"Top {n} {column} ranked by {by}:\n{result_df.to_string(index=False)}"


# ---------------------------------------------------------------------------
# Tool 6: filter_count
# ---------------------------------------------------------------------------
@tool
def filter_count(column: str, operator: str, value: str) -> str:
    """Count how many rows satisfy a filter condition.

    Args:
        column: Column to filter on (e.g. 'Discount', 'Region').
        operator: One of '>', '<', '>=', '<=', '==', '!='.
        value: Threshold value as a string (cast automatically to match dtype).
    """
    df = _require_df()
    if column not in df.columns:
        return f"Column '{column}' not found. Available: {list(df.columns)}"
    if operator not in OPERATORS:
        return f"operator must be one of {list(OPERATORS.keys())}, got '{operator}'."

    series = df[column]
    try:
        if pd.api.types.is_numeric_dtype(series):
            typed_value = float(value)
        else:
            typed_value = value
        mask = OPERATORS[operator](series, typed_value)
        count = int(mask.sum())
        pct = round(count / len(df) * 100, 2)
        return (
            f"Rows where {column} {operator} {value}: {count:,} "
            f"({pct}% of {len(df):,} total rows)"
        )
    except Exception as e:
        return f"Error evaluating filter: {e}"


# ---------------------------------------------------------------------------
# Tool 7: trend_over_time
# ---------------------------------------------------------------------------
@tool
def trend_over_time(date_column: str, metric_column: str, freq: str) -> str:
    """Aggregate a numeric metric over time and plot the trend.

    Args:
        date_column: Date column name (e.g. 'Order_Date').
        metric_column: Numeric column to aggregate (e.g. 'Sales', 'Profit').
        freq: Time frequency — 'M' for monthly, 'Q' for quarterly, 'Y' for yearly.
    """
    df = _require_df()
    for col in [date_column, metric_column]:
        if col not in df.columns:
            return f"Column '{col}' not found. Available: {list(df.columns)}"
    if freq not in ("M", "Q", "Y"):
        return "freq must be 'M', 'Q', or 'Y'."

    ts = (
        df.set_index(date_column)[metric_column]
        .resample(freq)
        .sum()
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ts.index, ts.values, marker="o", markersize=4, color="#4C72B0", linewidth=1.8)
    ax.fill_between(ts.index, ts.values, alpha=0.12, color="#4C72B0")
    ax.set_title(f"{metric_column} trend ({freq} frequency)")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Sum of {metric_column}")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.autofmt_xdate()
    _save_fig(fig, f"{metric_column} trend over time ({freq})")

    freq_label = {"M": "monthly", "Q": "quarterly", "Y": "yearly"}[freq]
    peak_period = ts.idxmax()
    trough_period = ts.idxmin()
    return (
        f"{freq_label.capitalize()} {metric_column} trend ({len(ts)} periods):\n"
        + ts.round(2).to_string()
        + f"\n\nPeak period : {peak_period} ({ts.max():,.2f})"
        + f"\nTrough period: {trough_period} ({ts.min():,.2f})"
    )


# ---------------------------------------------------------------------------
# Exported tool list
# ---------------------------------------------------------------------------
ANALYST_TOOLS = [
    describe_dataset,
    investigate_distribution,
    group_compare,
    correlate,
    top_n,
    filter_count,
    trend_over_time,
]
