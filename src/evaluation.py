from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred)
    safe = np.where(denominator == 0, 1.0, denominator)
    value = 2.0 * np.abs(y_pred - y_true) / safe
    return float(100.0 * np.mean(value))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }


def _build_time_splits(n_rows: int, n_splits: int, test_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    min_required = n_splits * test_size + test_size
    if n_rows <= min_required:
        raise ValueError(
            "Not enough rows for requested walk-forward setup. "
            f"Rows={n_rows}, required>{min_required}."
        )

    train_end = n_rows - (n_splits * test_size)
    splits = []
    for split in range(n_splits):
        test_start = train_end + split * test_size
        test_end = test_start + test_size

        train_idx = np.arange(0, test_start)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))

    return splits


def evaluate_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, object],
    n_splits: int = 4,
    test_size: int = 24 * 7,
) -> pd.DataFrame:
    splits = _build_time_splits(len(X), n_splits=n_splits, test_size=test_size)
    rows = []

    y_values = y.to_numpy()

    for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y_values[test_idx]

        baseline_prev_hour = y_values[test_idx - 1]
        baseline_prev_day = y_values[test_idx - 24]

        for baseline_name, baseline_pred in {
            "baseline_prev_hour": baseline_prev_hour,
            "baseline_prev_day": baseline_prev_day,
        }.items():
            metrics = compute_metrics(y_test, baseline_pred)
            rows.append(
                {
                    "fold": fold_id,
                    "model": baseline_name,
                    **metrics,
                }
            )

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            metrics = compute_metrics(y_test, pred)
            rows.append(
                {
                    "fold": fold_id,
                    "model": model_name,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def evaluate_holdout(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: Dict[str, object],
) -> pd.DataFrame:
    rows = []

    y_test_values = y_test.to_numpy()
    y_full = pd.concat([y_train, y_test]).to_numpy()
    offset = len(y_train)

    baseline_prev_hour = y_full[offset - 1 : -1]
    baseline_prev_day = y_full[offset - 24 : -24]

    for baseline_name, baseline_pred in {
        "baseline_prev_hour": baseline_prev_hour,
        "baseline_prev_day": baseline_prev_day,
    }.items():
        metrics = compute_metrics(y_test_values, baseline_pred)
        rows.append({"model": baseline_name, **metrics})

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = compute_metrics(y_test_values, pred)
        rows.append({"model": model_name, **metrics})

    return pd.DataFrame(rows)
