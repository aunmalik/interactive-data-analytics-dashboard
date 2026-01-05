import pandas as pd
import numpy as np
import pytest

from app.data_standard import (
    normalize_column,
    standardize_column,
    log_transform,
    bin_column,
    percentage_of_total,
    rank_column,
    calculate_growth,
)


def test_normalize_column():
    df = pd.DataFrame({"A": [0, 50, 100]})
    result = normalize_column(df, "A")
    assert "A_normalized" in result.columns
    assert result["A_normalized"].min() == 0
    assert result["A_normalized"].max() == 1


def test_normalize_invalid_column():
    df = pd.DataFrame({"A": [1, 2, 3]})
    result = normalize_column(df, "B")
    assert "B_normalized" not in result.columns


def test_standardize_column():
    df = pd.DataFrame({"A": [10, 20, 30]})
    result = standardize_column(df, "A")
    assert "A_standardized" in result.columns
    assert abs(result["A_standardized"].mean()) < 1e-9


def test_log_transform():
    df = pd.DataFrame({"A": [1, 10, 100]})
    result = log_transform(df, "A")
    assert "A_log" in result.columns
    assert result["A_log"].iloc[0] < result["A_log"].iloc[2]


def test_log_transform_negative():
    df = pd.DataFrame({"A": [-1, 0, 1]})
    result = log_transform(df, "A")
    assert "A_log" not in result.columns


def test_bin_column():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    result = bin_column(df, "A", bins=3)
    assert "A_binned" in result.columns
    assert result["A_binned"].nunique(dropna=True) == 3


def test_percentage_of_total():
    df = pd.DataFrame({"A": [25, 25, 50]})
    result = percentage_of_total(df, "A")
    assert "A_pct" in result.columns
    assert result["A_pct"].sum() == pytest.approx(100.0, abs=0.01)


def test_rank_column_ordering():
    df = pd.DataFrame({"A": [30, 10, 20]})
    result = rank_column(df, "A", ascending=True)
    assert "A_rank" in result.columns

    ranks = result["A_rank"].tolist()
    assert ranks[1] < ranks[2] < ranks[0]


def test_calculate_growth():
    df = pd.DataFrame({"A": [100, 110, 121]})
    result = calculate_growth(df, "A")
    assert "A_growth" in result.columns
    assert result["A_growth"].iloc[1] == pytest.approx(10.0, abs=0.001)
