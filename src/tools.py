"""
Analyst toolbox — 13 curated analytic primitives exposed as LangChain tools.

Each tool returns a plain-text summary for the agent to reason over and, where
relevant, registers a Matplotlib figure in a module-level store that the UI
reads after the agent finishes its run.  The same callables back every agent
strategy (LangGraph, manual ReAct, plan-execute), so the toolbox is the single
shared substrate of the comparative study.
"""
from __future__ import annotations

import io
import operator as op
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from scipy import stats
from sklearn.linear_model import LinearRegression

from src import viz

# ---------------------------------------------------------------------------
# Module-level state shared with the UI
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


def _save_fig(fig: plt.Figure, label: str) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=115, bbox_inches="tight")
    buf.seek(0)
    _figures.append(buf.read())
    _figure_labels.append(label)
    plt.close(fig)


def _require_df() -> pd.DataFrame:
    if _df is None:
        raise RuntimeError("Dataset not loaded. Call init_tools(df) first.")
    return _df


OPERATORS = {">": op.gt, "<": op.lt, ">=": op.ge, "<=": op.le, "==": op.eq, "!=": op.ne}


# ===========================================================================
# Descriptive primitives (original toolbox)
# ===========================================================================
@tool
def describe_dataset() -> str:
    """Return shape, column names, data types, and missingness summary of the dataset."""
    df = _require_df()
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    rows = [
        f"  {c}: dtype={df[c].dtype}, nulls={missing[c]} ({missing_pct[c]}%)"
        for c in df.columns
    ]
    return (
        f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n"
        f"Columns:\n" + "\n".join(rows) + "\n"
        f"\nNumeric summary:\n{df.describe(include='number').round(2).to_string()}"
    )


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
        stats_txt = series.describe().round(4)
        skew = round(float(series.skew()), 4)
        ax.hist(series, bins=30, edgecolor="white", color=viz.PALETTE["primary"], alpha=0.9)
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        viz.thousands(ax, "x")
        _save_fig(fig, f"Distribution of {column}")
        return f"Numeric distribution - {column}:\n{stats_txt.to_string()}\nskewness={skew}"
    counts = series.value_counts().head(15)
    ax.barh(counts.index[::-1].astype(str), counts.values[::-1],
            color=viz.PALETTE["primary"], alpha=0.9)
    ax.set_title(f"Value counts - {column}")
    ax.set_xlabel("Count")
    _save_fig(fig, f"Value counts - {column}")
    return f"Categorical distribution - {column} (top values):\n{counts.to_string()}"


@tool
def group_compare(metric_column: str, group_by_column: str, agg: str) -> str:
    """Compute an aggregated metric grouped by a categorical column and produce a bar chart.

    Args:
        metric_column: Numeric column to aggregate (e.g. 'Sales', 'Profit').
        group_by_column: Categorical column to group by (e.g. 'Region', 'Category').
        agg: Aggregation - one of 'mean', 'sum', 'median', 'count'.
    """
    df = _require_df()
    for col in (metric_column, group_by_column):
        if col not in df.columns:
            return f"Column '{col}' not found. Available: {list(df.columns)}"
    if agg not in ("mean", "sum", "median", "count"):
        return f"agg must be one of mean/sum/median/count, got '{agg}'."

    grouped = df.groupby(group_by_column)[metric_column].agg(agg).sort_values()
    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(grouped) + 1)))
    ax.barh(grouped.index.astype(str), grouped.values,
            color=viz.bar_colors(grouped.values), alpha=0.9)
    ax.axvline(0, color="#666666", linewidth=0.8)
    ax.set_title(f"{agg.capitalize()} of {metric_column} by {group_by_column}")
    ax.set_xlabel(f"{agg}({metric_column})")
    viz.annotate_barh(ax, grouped.values)
    viz.thousands(ax, "x")
    _save_fig(fig, f"{agg}({metric_column}) by {group_by_column}")
    return (
        f"{agg}({metric_column}) grouped by {group_by_column}:\n"
        + grouped.sort_values(ascending=False).round(2).to_string()
    )


@tool
def correlate(column_a: str, column_b: str) -> str:
    """Compute the Pearson correlation between two numeric columns and generate a scatter plot.

    Args:
        column_a: First numeric column (e.g. 'Discount').
        column_b: Second numeric column (e.g. 'Profit').
    """
    df = _require_df()
    for col in (column_a, column_b):
        if col not in df.columns:
            return f"Column '{col}' not found. Available: {list(df.columns)}"

    pair = df[[column_a, column_b]].dropna()
    r = round(float(pair[column_a].corr(pair[column_b])), 4)
    n = len(pair)
    sample = pair.sample(min(1500, n), random_state=42)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(sample[column_a], sample[column_b], alpha=0.25, s=18,
               color=viz.PALETTE["primary"])
    m, b = np.polyfit(pair[column_a], pair[column_b], 1)
    x_line = np.linspace(pair[column_a].min(), pair[column_a].max(), 100)
    ax.plot(x_line, m * x_line + b, color=viz.PALETTE["negative"], linewidth=1.8,
            label=f"r = {r}")
    ax.set_xlabel(column_a)
    ax.set_ylabel(column_b)
    ax.set_title(f"Correlation: {column_a} vs {column_b}")
    ax.legend()
    _save_fig(fig, f"Correlation: {column_a} vs {column_b}")
    strength = "strong" if abs(r) > 0.6 else "moderate" if abs(r) > 0.3 else "weak"
    direction = "positive" if r > 0 else "negative"
    return (
        f"Pearson r = {r} (n={n:,})\n"
        f"Interpretation: {strength} {direction} linear relationship between "
        f"{column_a} and {column_b}.\n"
        f"OLS slope: {round(m, 4)} - a 1-unit increase in {column_a} is associated "
        f"with a {round(m, 2):+.2f} change in {column_b}."
    )


@tool
def top_n(column: str, n: int, by: str) -> str:
    """Return the top N categories ranked by an aggregated column.

    Args:
        column: Column to rank (e.g. 'Sub_Category').
        n: Number of rows to return (1-20).
        by: Numeric column to rank by (e.g. 'Profit', 'Sales').
    """
    df = _require_df()
    for col in (column, by):
        if col not in df.columns:
            return f"Column '{col}' not found. Available: {list(df.columns)}"
    n = max(1, min(n, 20))

    if pd.api.types.is_numeric_dtype(df[by]) and not pd.api.types.is_numeric_dtype(df[column]):
        grouped = df.groupby(column)[by].sum().nlargest(n)
        fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * n + 1)))
        ax.barh(grouped.index[::-1].astype(str), grouped.values[::-1],
                color=viz.PALETTE["primary"], alpha=0.9)
        ax.set_title(f"Top {n} {column} by {by}")
        ax.set_xlabel(by)
        viz.thousands(ax, "x")
        _save_fig(fig, f"Top {n} {column} by {by}")
        return f"Top {n} {column} ranked by sum of {by}:\n{grouped.round(2).to_string()}"

    result_df = df[[column, by]].nlargest(n, by).reset_index(drop=True)
    return f"Top {n} {column} ranked by {by}:\n{result_df.to_string(index=False)}"


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
        typed = float(value) if pd.api.types.is_numeric_dtype(series) else value
        mask = OPERATORS[operator](series, typed)
        count = int(mask.sum())
        pct = round(count / len(df) * 100, 2)
        return (
            f"Rows where {column} {operator} {value}: {count:,} "
            f"({pct}% of {len(df):,} total rows)"
        )
    except Exception as e:
        return f"Error evaluating filter: {e}"


@tool
def trend_over_time(date_column: str, metric_column: str, freq: str) -> str:
    """Aggregate a numeric metric over time and plot the trend.

    Args:
        date_column: Date column name (e.g. 'Order_Date').
        metric_column: Numeric column to aggregate (e.g. 'Sales', 'Profit').
        freq: Time frequency - 'M' monthly, 'Q' quarterly, 'Y' yearly.
    """
    df = _require_df()
    for col in (date_column, metric_column):
        if col not in df.columns:
            return f"Column '{col}' not found. Available: {list(df.columns)}"
    if freq not in ("M", "Q", "Y"):
        return "freq must be 'M', 'Q', or 'Y'."

    freq_alias = {"M": "ME", "Q": "QE", "Y": "YE"}[freq]
    ts = df.set_index(date_column)[metric_column].resample(freq_alias).sum().dropna()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ts.index, ts.values, marker="o", markersize=4,
            color=viz.PALETTE["primary"], linewidth=1.8)
    ax.fill_between(ts.index, ts.values, alpha=0.12, color=viz.PALETTE["primary"])
    ax.set_title(f"{metric_column} trend ({freq} frequency)")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Sum of {metric_column}")
    viz.thousands(ax, "y")
    fig.autofmt_xdate()
    _save_fig(fig, f"{metric_column} trend over time ({freq})")
    label = {"M": "monthly", "Q": "quarterly", "Y": "yearly"}[freq]
    return (
        f"{label.capitalize()} {metric_column} trend ({len(ts)} periods):\n"
        + ts.round(2).to_string()
        + f"\n\nPeak period : {ts.idxmax().date()} ({ts.max():,.2f})"
        + f"\nTrough period: {ts.idxmin().date()} ({ts.min():,.2f})"
    )


# ===========================================================================
# Inferential & modeling primitives (new toolbox)
# ===========================================================================
@tool
def statistical_test(group_column: str, value_column: str) -> str:
    """Test whether a numeric metric differs significantly across groups.

    Uses Welch's t-test for two groups or one-way ANOVA for three or more.
    Reports the test statistic, p-value, and an effect size (Cohen's d or
    eta-squared), plus a box plot of the metric by group.

    Args:
        group_column: Categorical grouping column (e.g. 'Region', 'Segment').
        value_column: Numeric metric to compare (e.g. 'Profit', 'Sales').
    """
    df = _require_df()
    for col in (group_column, value_column):
        if col not in df.columns:
            return f"Column '{col}' not found. Available: {list(df.columns)}"
    if not pd.api.types.is_numeric_dtype(df[value_column]):
        return f"'{value_column}' must be numeric for a statistical test."

    groups = [g[value_column].dropna().values for _, g in df.groupby(group_column)]
    names = [str(k) for k, _ in df.groupby(group_column)]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return "Need at least two groups with >1 observation each."

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot([g for g in groups], tick_labels=names, showfliers=False,
               patch_artist=True,
               boxprops=dict(facecolor=viz.PALETTE["muted"], color="#444"),
               medianprops=dict(color=viz.PALETTE["negative"], linewidth=1.6))
    ax.set_title(f"{value_column} by {group_column}")
    ax.set_ylabel(value_column)
    viz.thousands(ax, "y")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save_fig(fig, f"{value_column} by {group_column} (box plot)")

    if len(groups) == 2:
        t, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
        n1, n2 = len(groups[0]), len(groups[1])
        s_pooled = np.sqrt(((n1 - 1) * groups[0].var(ddof=1) +
                            (n2 - 1) * groups[1].var(ddof=1)) / (n1 + n2 - 2))
        d = (groups[0].mean() - groups[1].mean()) / s_pooled if s_pooled else 0.0
        mag = "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small" if abs(d) > 0.2 else "negligible"
        sig = "statistically significant" if p < 0.05 else "not statistically significant"
        return (
            f"Welch's t-test of {value_column} between {names[0]} and {names[1]}:\n"
            f"  t = {t:.4f}, p = {p:.4g}\n"
            f"  means: {names[0]}={groups[0].mean():.2f}, {names[1]}={groups[1].mean():.2f}\n"
            f"  Cohen's d = {d:.3f} ({mag} effect)\n"
            f"  Result: difference is {sig} at alpha=0.05."
        )

    f_stat, p = stats.f_oneway(*groups)
    grand = np.concatenate(groups)
    ss_between = sum(len(g) * (g.mean() - grand.mean()) ** 2 for g in groups)
    ss_total = ((grand - grand.mean()) ** 2).sum()
    eta_sq = ss_between / ss_total if ss_total else 0.0
    sig = "statistically significant" if p < 0.05 else "not statistically significant"
    means = "\n".join(f"  {nm}: {g.mean():.2f}" for nm, g in zip(names, groups))
    return (
        f"One-way ANOVA of {value_column} across {len(groups)} {group_column} groups:\n"
        f"  F = {f_stat:.4f}, p = {p:.4g}\n"
        f"  eta-squared = {eta_sq:.3f} (share of variance explained by {group_column})\n"
        f"  Group means:\n{means}\n"
        f"  Result: between-group differences are {sig} at alpha=0.05."
    )


@tool
def linear_regression(feature_columns: list[str], target_column: str) -> str:
    """Fit an ordinary-least-squares regression predicting a numeric target.

    Supports one or more numeric features. Reports R-squared, adjusted
    R-squared, per-feature coefficients, and an actual-vs-predicted plot.

    Args:
        feature_columns: List of numeric predictor columns (e.g. ['Sales','Discount']).
        target_column: Numeric column to predict (e.g. 'Profit').
    """
    df = _require_df()
    cols = list(feature_columns) + [target_column]
    for col in cols:
        if col not in df.columns:
            return f"Column '{col}' not found. Available: {list(df.columns)}"
        if not pd.api.types.is_numeric_dtype(df[col]):
            return f"'{col}' must be numeric for regression."

    data = df[cols].dropna()
    X = data[feature_columns].values
    y = data[target_column].values
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    n, k = len(y), X.shape[1]
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n - k - 1 > 0 else r2

    fig, ax = plt.subplots(figsize=(6, 5))
    samp = np.random.default_rng(42).choice(n, size=min(1500, n), replace=False)
    ax.scatter(y[samp], pred[samp], alpha=0.25, s=16, color=viz.PALETTE["primary"])
    lims = [min(y.min(), pred.min()), max(y.max(), pred.max())]
    ax.plot(lims, lims, color=viz.PALETTE["negative"], linewidth=1.6, label="perfect fit")
    ax.set_xlabel(f"Actual {target_column}")
    ax.set_ylabel(f"Predicted {target_column}")
    ax.set_title(f"OLS fit: {target_column} ~ {' + '.join(feature_columns)}")
    ax.legend()
    _save_fig(fig, f"Regression actual vs predicted ({target_column})")

    coefs = "\n".join(f"  {f}: {c:+.4f}" for f, c in zip(feature_columns, model.coef_))
    return (
        f"OLS regression: {target_column} ~ {' + '.join(feature_columns)} (n={n:,})\n"
        f"  R-squared = {r2:.4f}, adjusted R-squared = {adj_r2:.4f}\n"
        f"  Intercept = {model.intercept_:+.4f}\n"
        f"  Coefficients:\n{coefs}\n"
        f"  Interpretation: the model explains {r2*100:.1f}% of the variance in "
        f"{target_column}."
    )


@tool
def detect_outliers(column: str, method: str = "iqr") -> str:
    """Detect outliers in a numeric column using the IQR or z-score rule.

    Args:
        column: Numeric column to scan (e.g. 'Profit', 'Sales').
        method: 'iqr' (1.5*IQR fences) or 'zscore' (|z|>3).
    """
    df = _require_df()
    if column not in df.columns:
        return f"Column '{column}' not found. Available: {list(df.columns)}"
    if not pd.api.types.is_numeric_dtype(df[column]):
        return f"'{column}' must be numeric."
    if method not in ("iqr", "zscore"):
        return "method must be 'iqr' or 'zscore'."

    s = df[column].dropna()
    if method == "iqr":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (s < low) | (s > high)
        bounds = f"fences = [{low:,.2f}, {high:,.2f}]"
    else:
        z = (s - s.mean()) / s.std(ddof=0)
        mask = z.abs() > 3
        low, high = s.mean() - 3 * s.std(ddof=0), s.mean() + 3 * s.std(ddof=0)
        bounds = f"|z|>3 fences = [{low:,.2f}, {high:,.2f}]"

    n_out = int(mask.sum())
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.boxplot(s.values, vert=False, showfliers=True,
               patch_artist=True,
               boxprops=dict(facecolor=viz.PALETTE["muted"], color="#444"),
               flierprops=dict(marker="o", markerfacecolor=viz.PALETTE["negative"],
                               markersize=3, alpha=0.4),
               medianprops=dict(color=viz.PALETTE["negative"], linewidth=1.6))
    ax.set_title(f"Outliers in {column} ({method.upper()})")
    ax.set_xlabel(column)
    viz.thousands(ax, "x")
    _save_fig(fig, f"Outliers in {column} ({method})")

    extremes = s[mask].sort_values()
    sample = ""
    if n_out:
        lo = extremes.head(3).round(2).tolist()
        hi = extremes.tail(3).round(2).tolist()
        sample = f"\n  Most extreme low: {lo}\n  Most extreme high: {hi}"
    return (
        f"Outlier scan of {column} via {method.upper()} ({bounds}):\n"
        f"  {n_out:,} outliers out of {len(s):,} rows ({n_out/len(s)*100:.2f}%)." + sample
    )


@tool
def segment_profitability(row_dimension: str, col_dimension: str,
                          metric_column: str = "Profit", agg: str = "sum") -> str:
    """Build a two-dimensional segment matrix of an aggregated metric and a heatmap.

    Useful for questions like 'which category/region combinations lose money?'.

    Args:
        row_dimension: Categorical column for rows (e.g. 'Category').
        col_dimension: Categorical column for columns (e.g. 'Region').
        metric_column: Numeric metric to aggregate (default 'Profit').
        agg: 'sum' or 'mean' (default 'sum').
    """
    df = _require_df()
    for col in (row_dimension, col_dimension, metric_column):
        if col not in df.columns:
            return f"Column '{col}' not found. Available: {list(df.columns)}"
    if agg not in ("sum", "mean"):
        return "agg must be 'sum' or 'mean'."

    pivot = pd.pivot_table(df, index=row_dimension, columns=col_dimension,
                           values=metric_column, aggfunc=agg)
    fig, ax = plt.subplots(figsize=(1.4 * len(pivot.columns) + 3, 0.6 * len(pivot) + 2.5))
    import seaborn as sns
    sns.heatmap(pivot, annot=True, fmt=",.0f", cmap="RdYlGn", center=0,
                linewidths=0.5, linecolor="white", ax=ax,
                cbar_kws={"label": f"{agg}({metric_column})"})
    ax.set_title(f"{agg.capitalize()} {metric_column}: {row_dimension} x {col_dimension}")
    _save_fig(fig, f"{metric_column} segment matrix ({row_dimension} x {col_dimension})")

    worst = pivot.stack().idxmin()
    worst_val = pivot.stack().min()
    best = pivot.stack().idxmax()
    best_val = pivot.stack().max()
    return (
        f"{agg}({metric_column}) by {row_dimension} (rows) x {col_dimension} (cols):\n"
        + pivot.round(2).to_string()
        + f"\n\nWeakest segment: {worst} = {worst_val:,.2f}"
        + f"\nStrongest segment: {best} = {best_val:,.2f}"
    )


@tool
def pareto_analysis(category_column: str, value_column: str = "Sales") -> str:
    """Run an 80/20 (Pareto) analysis of a metric across categories.

    Reports how many categories account for ~80% of the metric and renders a
    Pareto chart (sorted bars + cumulative line).

    Args:
        category_column: Categorical column (e.g. 'Sub_Category', 'State').
        value_column: Numeric metric to accumulate (default 'Sales').
    """
    df = _require_df()
    for col in (category_column, value_column):
        if col not in df.columns:
            return f"Column '{col}' not found. Available: {list(df.columns)}"

    totals = df.groupby(category_column)[value_column].sum().sort_values(ascending=False)
    totals = totals[totals > 0]
    if totals.empty:
        return f"No positive {value_column} to accumulate for {category_column}."
    cum_pct = totals.cumsum() / totals.sum() * 100
    n_for_80 = int((cum_pct < 80).sum()) + 1

    fig, ax1 = plt.subplots(figsize=(max(7, 0.5 * len(totals)), 4.5))
    ax1.bar(totals.index.astype(str), totals.values,
            color=viz.PALETTE["primary"], alpha=0.85)
    ax1.set_ylabel(f"{value_column}", color=viz.PALETTE["primary"])
    viz.thousands(ax1, "y")
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(totals.index.astype(str), cum_pct.values, color=viz.PALETTE["accent"],
             marker="o", markersize=4, linewidth=1.8)
    ax2.axhline(80, color=viz.PALETTE["negative"], linestyle="--", linewidth=1.0)
    ax2.set_ylabel("Cumulative %", color=viz.PALETTE["accent"])
    ax2.set_ylim(0, 105)
    ax1.set_title(f"Pareto: {value_column} by {category_column}")
    _save_fig(fig, f"Pareto of {value_column} by {category_column}")

    share = totals.head(n_for_80).sum() / totals.sum() * 100
    return (
        f"Pareto analysis of {value_column} by {category_column}:\n"
        f"  {n_for_80} of {len(totals)} categories ({n_for_80/len(totals)*100:.0f}%) "
        f"account for {share:.1f}% of total {value_column}.\n"
        f"  Top contributors:\n"
        + totals.head(n_for_80).round(2).to_string()
    )


@tool
def correlation_matrix() -> str:
    """Compute the full Pearson correlation matrix of all numeric columns and a heatmap."""
    df = _require_df()
    num = df.select_dtypes("number")
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(1.1 * len(corr) + 2, 1.0 * len(corr) + 1.5))
    import seaborn as sns
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, linewidths=0.5, linecolor="white", ax=ax,
                cbar_kws={"label": "Pearson r"})
    ax.set_title("Correlation matrix (numeric columns)")
    _save_fig(fig, "Correlation matrix")

    pairs = (
        corr.where(mask).stack().reset_index()
        .rename(columns={0: "r", "level_0": "a", "level_1": "b"})
    )
    pairs["abs"] = pairs["r"].abs()
    strong = pairs.sort_values("abs", ascending=False).head(5)
    lines = "\n".join(
        f"  {row.a} ~ {row.b}: r = {row.r:+.3f}" for row in strong.itertuples()
    )
    return f"Pearson correlation matrix ({len(corr)} numeric columns).\nStrongest pairs:\n{lines}"


# ===========================================================================
# Exported registry
# ===========================================================================
ANALYST_TOOLS = [
    describe_dataset,
    investigate_distribution,
    group_compare,
    correlate,
    top_n,
    filter_count,
    trend_over_time,
    statistical_test,
    linear_regression,
    detect_outliers,
    segment_profitability,
    pareto_analysis,
    correlation_matrix,
]

TOOL_NAMES = [t.name for t in ANALYST_TOOLS]
