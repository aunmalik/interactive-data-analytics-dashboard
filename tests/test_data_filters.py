import pandas as pd

from app.data_filters import filter_numeric_range, filter_by_values


def test_filter_numeric_range():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
    result = filter_numeric_range(df, "A", min_val=2, max_val=4)
    assert len(result) == 3
    assert result["A"].min() >= 2
    assert result["A"].max() <= 4


def test_filter_numeric_range_invalid_column():
    df = pd.DataFrame({"A": [1, 2, 3]})
    result = filter_numeric_range(df, "B", min_val=1, max_val=2)
    assert len(result) == 3


def test_filter_by_values_include():
    df = pd.DataFrame({"A": ["x", "y", "z", "x", "y"]})
    result = filter_by_values(df, "A", ["x", "y"], include=True)
    assert len(result) == 4
    assert "z" not in result["A"].values


def test_filter_by_values_exclude():
    df = pd.DataFrame({"A": ["x", "y", "z", "x", "y"]})
    result = filter_by_values(df, "A", ["z"], include=False)
    assert len(result) == 4
    assert "z" not in result["A"].values


def test_filter_by_values_empty_selection():
    df = pd.DataFrame({"A": ["x", "y", "z"]})
    result = filter_by_values(df, "A", [], include=True)
    assert result.equals(df)