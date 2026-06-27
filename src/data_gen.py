"""Synthetic Superstore dataset generator with realistic business patterns."""
import numpy as np
import pandas as pd
from pathlib import Path

REGIONS = {
    "West":    {"weight": 0.28, "profit_mult": 1.15},
    "East":    {"weight": 0.28, "profit_mult": 1.05},
    "Central": {"weight": 0.22, "profit_mult": 0.90},
    "South":   {"weight": 0.22, "profit_mult": 0.80},
}

STATES_BY_REGION = {
    "West":    ["California", "Washington", "Oregon", "Nevada", "Arizona", "Colorado"],
    "East":    ["New York", "Pennsylvania", "New Jersey", "Virginia", "Massachusetts", "North Carolina"],
    "Central": ["Texas", "Illinois", "Ohio", "Michigan", "Indiana", "Missouri"],
    "South":   ["Florida", "Georgia", "Tennessee", "Alabama", "Louisiana", "South Carolina"],
}

CATEGORIES = {
    "Technology": {
        "base_margin": 0.26,
        "sub_categories": ["Phones", "Laptops", "Accessories", "Monitors", "Printers", "Copiers"],
        "avg_sales": 350,
        "weight": 0.32,
    },
    "Furniture": {
        "base_margin": 0.10,
        "sub_categories": ["Chairs", "Tables", "Bookcases", "Furnishings", "Storage"],
        "avg_sales": 280,
        "weight": 0.34,
    },
    "Office Supplies": {
        "base_margin": 0.18,
        "sub_categories": ["Paper", "Binders", "Art", "Envelopes", "Labels", "Fasteners", "Supplies"],
        "avg_sales": 80,
        "weight": 0.34,
    },
}

SEGMENTS = {"Consumer": 0.52, "Corporate": 0.30, "Home Office": 0.18}
SHIP_MODES = {"Standard Class": 0.60, "Second Class": 0.20, "First Class": 0.15, "Same Day": 0.05}
DISCOUNTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.2, 0.3, 0.4]


def _seasonal_dates(n: int, rng: np.random.Generator) -> pd.DatetimeIndex:
    """Sample dates 2021-2024 with Q4 sales spike."""
    months = rng.choice(
        range(1, 13),
        size=n,
        p=[0.06, 0.06, 0.08, 0.07, 0.07, 0.08, 0.08, 0.08, 0.09, 0.10, 0.11, 0.12],
    )
    years = rng.choice([2021, 2022, 2023, 2024], size=n, p=[0.20, 0.25, 0.28, 0.27])
    days = rng.integers(1, 29, size=n)
    return pd.to_datetime(
        {
            "year": years,
            "month": months,
            "day": np.minimum(days, pd.to_datetime({"year": years, "month": months, "day": 1}).dt.days_in_month),
        }
    )


def generate_superstore(n_rows: int = 5200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    region_names = list(REGIONS.keys())
    region_weights = [REGIONS[r]["weight"] for r in region_names]
    regions = rng.choice(region_names, size=n_rows, p=region_weights)

    states = np.array([
        rng.choice(STATES_BY_REGION[r]) for r in regions
    ])

    cat_names = list(CATEGORIES.keys())
    cat_weights = [CATEGORIES[c]["weight"] for c in cat_names]
    categories = rng.choice(cat_names, size=n_rows, p=cat_weights)

    sub_categories = np.array([
        rng.choice(CATEGORIES[c]["sub_categories"]) for c in categories
    ])

    seg_names = list(SEGMENTS.keys())
    seg_weights = list(SEGMENTS.values())
    segments = rng.choice(seg_names, size=n_rows, p=seg_weights)

    ship_names = list(SHIP_MODES.keys())
    ship_weights = list(SHIP_MODES.values())
    ship_modes = rng.choice(ship_names, size=n_rows, p=ship_weights)

    order_dates = _seasonal_dates(n_rows, rng)
    ship_offsets = rng.integers(2, 8, size=n_rows)
    ship_dates = order_dates + pd.to_timedelta(ship_offsets, unit="D")

    # Sales: log-normal scaled by category average
    base_sales = np.array([CATEGORIES[c]["avg_sales"] for c in categories])
    sales = rng.lognormal(mean=0.0, sigma=0.65, size=n_rows) * base_sales
    sales = np.round(np.clip(sales, 5.0, 8000.0), 2)

    quantities = rng.integers(1, 10, size=n_rows)
    discounts = np.array([rng.choice(DISCOUNTS) for _ in range(n_rows)])

    # Profit: margin depends on category, region multiplier, and discount
    base_margins = np.array([CATEGORIES[c]["base_margin"] for c in categories])
    region_mults = np.array([REGIONS[r]["profit_mult"] for r in regions])
    # Discount erodes margin; furniture categories hit harder
    cat_discount_sensitivity = np.array(
        [1.8 if c == "Furniture" else 1.2 for c in categories]
    )
    discount_drag = discounts * cat_discount_sensitivity
    effective_margin = base_margins * region_mults - discount_drag
    noise = rng.normal(0, 0.035, size=n_rows)
    profits = np.round(sales * (effective_margin + noise), 2)

    order_ids = [f"ORD-{2021 + i // (n_rows // 4):04d}-{i:05d}" for i in range(n_rows)]
    customer_ids = [f"CUS-{rng.integers(1000, 9999):04d}" for _ in range(n_rows)]

    df = pd.DataFrame(
        {
            "Order_ID": order_ids,
            "Order_Date": order_dates,
            "Ship_Date": ship_dates,
            "Ship_Mode": ship_modes,
            "Customer_ID": customer_ids,
            "Segment": segments,
            "Region": regions,
            "State": states,
            "Category": categories,
            "Sub_Category": sub_categories,
            "Sales": sales,
            "Quantity": quantities,
            "Discount": discounts,
            "Profit": profits,
        }
    )
    df = df.sort_values("Order_Date").reset_index(drop=True)
    return df


def load_or_generate(path: str = "data/superstore.csv") -> pd.DataFrame:
    p = Path(path)
    if p.exists():
        df = pd.read_csv(p, parse_dates=["Order_Date", "Ship_Date"])
        return df
    df = generate_superstore()
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return df


if __name__ == "__main__":
    df = generate_superstore()
    out = Path("data/superstore.csv")
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Generated {len(df):,} rows → {out}")
    print(df.dtypes)
    print(df.describe())
