#!/usr/bin/env python3
"""Load, clean, inspect, and summarize trade data with pandas.

The default column names match Binance's headerless aggregate-trade CSV files.
For a CSV that already has column names, pass ``--has-header``.  A headed file
must contain ``timestamp``, ``price``, and ``quantity`` columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIRECTORY = PROJECT_ROOT / "data" / "processed"

# Binance aggregate-trade files do not include a header row.  Naming every
# field here is clearer than working with integer column positions later.
RAW_COLUMN_NAMES = [
    "trade_id",
    "price",
    "quantity",
    "quote_quantity",
    "timestamp",
    "is_buyer_maker",
    "is_best_match",
]
REQUIRED_COLUMNS = {"timestamp", "price", "quantity"}


def load_orders(
    input_file: Path,
    *,
    has_header: bool = False,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Read an order/trade CSV and return it as a dataframe."""
    if not input_file.is_file():
        raise FileNotFoundError(f"input file does not exist: {input_file}")

    if has_header:
        dataframe = pd.read_csv(input_file, nrows=max_rows)
        dataframe.columns = [
            str(column).strip().lower() for column in dataframe.columns
        ]
    else:
        dataframe = pd.read_csv(input_file, header=None, nrows=max_rows)
        if len(dataframe.columns) != len(RAW_COLUMN_NAMES):
            raise ValueError(
                "headerless CSV must have exactly "
                f"{len(RAW_COLUMN_NAMES)} columns; found {len(dataframe.columns)}"
            )
        dataframe.columns = RAW_COLUMN_NAMES

    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV is missing required columns: {missing}")

    return dataframe

def parse_timestamp(values: pd.Series, timestamp_unit: str) -> pd.Series:
    """Convert numeric epoch values or timestamp strings to UTC datetimes."""
    numeric_values = pd.to_numeric(values, errors="coerce")
    non_missing_values = values.notna().sum()

    if numeric_values.notna().sum() == non_missing_values:
        return cast(
            pd.Series,
            pd.to_datetime(
                numeric_values,
                unit=timestamp_unit,
                errors="coerce",
                utc=True,
            ),
        )

    return cast(pd.Series, pd.to_datetime(values, errors="coerce", utc=True))


def clean_orders(dataframe: pd.DataFrame, timestamp_unit: str = "us") -> pd.DataFrame:
    """Convert important fields and add notional, datetime, and side columns."""
    cleaned = dataframe.copy()

    cleaned["timestamp"] = parse_timestamp(cleaned["timestamp"], timestamp_unit)
    cleaned["price"] = pd.to_numeric(cleaned["price"], errors="coerce")
    cleaned["quantity"] = pd.to_numeric(cleaned["quantity"], errors="coerce")

    # Rows without these values cannot participate in time, price, or volume
    # analysis.  Coercion above turns malformed values into missing values.
    cleaned = cleaned.dropna(subset=["timestamp", "price", "quantity"]).copy()

    cleaned["notional"] = cleaned["price"] * cleaned["quantity"]
    cleaned["datetime"] = cleaned["timestamp"]

    if "is_buyer_maker" in cleaned.columns:
        maker_values = (
            cleaned["is_buyer_maker"]
            .astype("string")
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
            .astype("boolean")
        )
        cleaned["is_buyer_maker"] = maker_values
        # If the buyer supplied resting liquidity, the taker initiated a sell.
        cleaned["side"] = maker_values.map({True: "sell", False: "buy"}).astype(
            "string"
        )

    return cleaned


def filter_orders(
    dataframe: pd.DataFrame,
    *,
    start: str | None = None,
    end: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    min_quantity: float | None = None,
) -> pd.DataFrame:
    """Select rows using optional time, price, and quantity thresholds."""
    selected = pd.Series(True, index=dataframe.index)

    if start is not None:
        selected &= dataframe["timestamp"] >= pd.to_datetime(start, utc=True)
    if end is not None:
        selected &= dataframe["timestamp"] <= pd.to_datetime(end, utc=True)
    if min_price is not None:
        selected &= dataframe["price"] >= min_price
    if max_price is not None:
        selected &= dataframe["price"] <= max_price
    if min_quantity is not None:
        selected &= dataframe["quantity"] >= min_quantity

    return dataframe.loc[selected].copy()


def calculate_vwap(dataframe: pd.DataFrame) -> float:
    """Return volume-weighted average price, or NaN for zero total volume."""
    total_quantity = dataframe["quantity"].sum()
    if total_quantity == 0:
        return float("nan")
    return float(dataframe["notional"].sum() / total_quantity)


def aggregate_orders(dataframe: pd.DataFrame) -> pd.Series:
    """Calculate basic summary measures for the selected records."""
    return pd.Series(
        {
            "total_records": len(dataframe),
            "total_traded_quantity": dataframe["quantity"].sum(),
            "total_notional": dataframe["notional"].sum(),
            "average_price": dataframe["price"].mean(),
            "vwap": calculate_vwap(dataframe),
        }
    )


def group_orders_by_time(
    dataframe: pd.DataFrame, interval: str = "1min"
) -> pd.DataFrame:
    """Aggregate records into fixed time intervals."""
    grouped = (
        dataframe.groupby(pd.Grouper(key="timestamp", freq=interval))
        .agg(
            number_of_trades=("price", "size"),
            total_quantity=("quantity", "sum"),
            total_notional=("notional", "sum"),
        )
        .reset_index()
    )

    grouped["vwap"] = grouped["total_notional"].div(grouped["total_quantity"])
    grouped.loc[grouped["total_quantity"] == 0, "vwap"] = float("nan")
    return grouped


def print_dataframe_inspection(dataframe: pd.DataFrame) -> None:
    """Print common first checks for a newly loaded dataframe."""
    print("\n=== Dataframe shape ===")
    print(dataframe.shape)

    print("\n=== First 5 rows ===")
    print(dataframe.head())

    print("\n=== Column names ===")
    print(dataframe.columns.tolist())

    print("\n=== Data types ===")
    print(dataframe.dtypes)

    print("\n=== Missing values by column ===")
    print(dataframe.isna().sum())

    print("\n=== Descriptive statistics ===")
    print(dataframe.describe(include="all"))


def save_cleaned_data(dataframe: pd.DataFrame, output_file: Path) -> None:
    """Save a cleaned dataframe, creating the output directory if needed."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_file, index=False)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load, clean, inspect, filter, and aggregate trade CSV data."
    )
    parser.add_argument("input_file", type=Path, help="CSV file to inspect")
    parser.add_argument(
        "--has-header",
        action="store_true",
        help="read column names from the first row instead of using the default schema",
    )
    parser.add_argument(
        "--timestamp-unit",
        choices=("s", "ms", "us", "ns"),
        default="us",
        help="unit for numeric epoch timestamps (default: us)",
    )
    parser.add_argument("--start", help="inclusive UTC start time")
    parser.add_argument("--end", help="inclusive UTC end time")
    parser.add_argument("--min-price", type=float, help="minimum price to include")
    parser.add_argument("--max-price", type=float, help="maximum price to include")
    parser.add_argument(
        "--min-quantity", type=float, help="minimum quantity to include"
    )
    parser.add_argument(
        "--interval",
        default="1min",
        help="pandas time interval used for grouping (default: 1min)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        help="read only the first N data rows (useful while exploring a large file)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output CSV (default: data/processed/<input_name>_cleaned.csv)",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    input_file = args.input_file.expanduser()
    output_file = args.output or (
        PROCESSED_DATA_DIRECTORY / f"{input_file.stem}_cleaned.csv"
    )

    raw_orders = load_orders(
        input_file,
        has_header=args.has_header,
        max_rows=args.rows,
    )
    cleaned_orders = clean_orders(raw_orders, timestamp_unit=args.timestamp_unit)
    dropped_rows = len(raw_orders) - len(cleaned_orders)

    print_dataframe_inspection(cleaned_orders)
    print(f"\nRows removed during cleaning: {dropped_rows}")

    filtered_orders = filter_orders(
        cleaned_orders,
        start=args.start,
        end=args.end,
        min_price=args.min_price,
        max_price=args.max_price,
        min_quantity=args.min_quantity,
    )

    print("\n=== Aggregation for filtered records ===")
    print(aggregate_orders(filtered_orders))

    print(f"\n=== Time aggregation ({args.interval}) ===")
    print(group_orders_by_time(filtered_orders, interval=args.interval))

    save_cleaned_data(cleaned_orders, output_file)
    print(f"\nCleaned data saved to: {output_file}")


if __name__ == "__main__":
    main()
