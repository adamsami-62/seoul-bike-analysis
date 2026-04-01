from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


TARGET_COLUMN = "rented_bike_count"
TIMESTAMP_COLUMN = "timestamp"


EXPECTED_COLUMNS = [
    "date",
    "rented_bike_count",
    "hour",
    "temperature_c",
    "humidity_pct",
    "wind_speed_mps",
    "visibility_10m",
    "dew_point_temperature_c",
    "solar_radiation_mj_m2",
    "rainfall_mm",
    "snowfall_cm",
    "seasons",
    "holiday",
    "functioning_day",
]


@dataclass
class DatasetSummary:
    row_count: int
    start_timestamp: str
    end_timestamp: str
    target_mean: float
    target_std: float


def _canonical_column_name(raw_name: str) -> str:
    normalized = raw_name.strip().lower()
    squashed = (
        normalized.replace(" ", "")
        .replace("_", "")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "pct")
        .replace("/", "per")
    )

    if squashed.startswith("temperature"):
        return "temperature_c"
    if squashed.startswith("dewpointtemperature"):
        return "dew_point_temperature_c"
    if squashed == "date":
        return "date"
    if squashed == "rentedbikecount":
        return "rented_bike_count"
    if squashed == "hour":
        return "hour"
    if squashed.startswith("humidity"):
        return "humidity_pct"
    if squashed.startswith("windspeed"):
        return "wind_speed_mps"
    if squashed.startswith("visibility"):
        return "visibility_10m"
    if squashed.startswith("solarradiation"):
        return "solar_radiation_mj_m2"
    if squashed.startswith("rainfall"):
        return "rainfall_mm"
    if squashed.startswith("snowfall"):
        return "snowfall_cm"
    if squashed == "seasons":
        return "seasons"
    if squashed == "holiday":
        return "holiday"
    if squashed == "functioningday":
        return "functioning_day"

    raise ValueError(f"Unrecognized column name: {raw_name}")


def normalize_columns(columns: List[str]) -> Dict[str, str]:
    return {column: _canonical_column_name(column) for column in columns}


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    mapping = normalize_columns(df.columns.tolist())
    df = df.rename(columns=mapping)

    missing = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    df = df[EXPECTED_COLUMNS].copy()

    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df["hour"] = df["hour"].astype(int)
    df[TIMESTAMP_COLUMN] = df["date"] + pd.to_timedelta(df["hour"], unit="h")
    df = df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)

    numeric_columns = [
        "rented_bike_count",
        "temperature_c",
        "humidity_pct",
        "wind_speed_mps",
        "visibility_10m",
        "dew_point_temperature_c",
        "solar_radiation_mj_m2",
        "rainfall_mm",
        "snowfall_cm",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    category_columns = ["seasons", "holiday", "functioning_day"]
    for column in category_columns:
        df[column] = df[column].astype(str).str.strip()

    if df.isna().any().any():
        na_counts = df.isna().sum()
        problematic = na_counts[na_counts > 0].to_dict()
        raise ValueError(f"Missing values detected after parsing: {problematic}")

    return df


def summarize_dataset(df: pd.DataFrame) -> DatasetSummary:
    return DatasetSummary(
        row_count=int(len(df)),
        start_timestamp=str(df[TIMESTAMP_COLUMN].min()),
        end_timestamp=str(df[TIMESTAMP_COLUMN].max()),
        target_mean=float(df[TARGET_COLUMN].mean()),
        target_std=float(df[TARGET_COLUMN].std()),
    )
