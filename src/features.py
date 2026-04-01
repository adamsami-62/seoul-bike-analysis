from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from src.data_pipeline import TARGET_COLUMN, TIMESTAMP_COLUMN


LAG_WINDOWS = [1, 2, 3, 24, 168]
ROLL_WINDOWS = [3, 24, 168]


@dataclass
class FeatureSet:
    frame: pd.DataFrame
    feature_columns: List[str]


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df[TIMESTAMP_COLUMN]

    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def _add_target_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for lag in LAG_WINDOWS:
        df[f"lag_{lag}"] = df[TARGET_COLUMN].shift(lag)

    shifted_target = df[TARGET_COLUMN].shift(1)
    for window in ROLL_WINDOWS:
        df[f"roll_mean_{window}"] = shifted_target.rolling(window=window).mean()

    return df


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["temp_x_humidity"] = df["temperature_c"] * df["humidity_pct"]
    df["temp_x_wind"] = df["temperature_c"] * df["wind_speed_mps"]
    df["rain_x_humidity"] = df["rainfall_mm"] * df["humidity_pct"]
    return df


def build_feature_frame(raw_df: pd.DataFrame) -> FeatureSet:
    df = _add_time_features(raw_df)
    df = _add_target_history_features(df)
    df = _add_interaction_features(df)

    category_columns = ["seasons", "holiday", "functioning_day"]
    df = pd.get_dummies(df, columns=category_columns, drop_first=False)

    df = df.dropna().reset_index(drop=True)

    excluded = {"date", TARGET_COLUMN, TIMESTAMP_COLUMN}
    feature_columns = [column for column in df.columns if column not in excluded]

    return FeatureSet(frame=df, feature_columns=feature_columns)
