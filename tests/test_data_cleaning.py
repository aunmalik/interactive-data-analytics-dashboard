import pandas as pd

from app.data_cleaning import (
    remove_duplicates,
    drop_missing_rows,
    fill_numeric_with_mean,
    fill_numeric_with_median,
    remove_outliers_iqr,
    cap_outliers_iqr,
)


def test_remove_duplicates():
    df = pd.DataFrame({"A": [1, 1, 2], "B": [10, 10, 20]})
    cleaned = remove_duplicates(df)
    assert len(cleaned) == 2


def test_remove_duplicates_no_duplicates():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [10, 20, 30]})
    cleaned = remove_duplicates(df)
    assert len(cleaned) == 3


def test_drop_missing_rows():
    df = pd.DataFrame({"A": [1, None, 3], "B": [10, 20, None]})
    cleaned = drop_missing_rows(df)
    assert len(cleaned) == 1
    assert cleaned.isnull().sum().sum() == 0


def test_fill_numeric_with_mean():
    df = pd.DataFrame({"A": [1, None, 3], "B": [10, 20, 30]})
    filled = fill_numeric_with_mean(df)

    assert filled["A"].isnull().sum() == 0
    assert filled["A"].iloc[1] == 2.0


def test_fill_numeric_with_mean_ignores_non_numeric():
    df = pd.DataFrame({"A": [1, None, 3], "City": ["Riga", None, "Liepaja"]})
    filled = fill_numeric_with_mean(df)

    assert filled["A"].isnull().sum() == 0
    assert filled["City"].isnull().sum() == 1


def test_fill_numeric_with_median():
    df = pd.DataFrame({"A": [1, None, 3, 100]})
    filled = fill_numeric_with_median(df)

    expected = df["A"].dropna().median()
    assert filled["A"].isnull().sum() == 0
    assert filled["A"].iloc[1] == expected


def test_remove_outliers_iqr():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 100]})
    result = remove_outliers_iqr(df, "A", multiplier=1.5)

    assert len(result) < len(df)
    assert 100 not in result["A"].values


def test_remove_outliers_iqr_invalid_column():
    df = pd.DataFrame({"A": [1, 2, 3]})
    result = remove_outliers_iqr(df, "B", multiplier=1.5)
    assert len(result) == 3


def test_remove_outliers_iqr_non_numeric_column_no_change():
    df = pd.DataFrame({"A": ["x", "y", "z"]})
    result = remove_outliers_iqr(df, "A", multiplier=1.5)
    assert result.equals(df)


def test_cap_outliers_iqr():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 100]})
    result = cap_outliers_iqr(df, "A", multiplier=1.5)

    assert len(result) == len(df)
    assert result["A"].max() <= df["A"].max()
    assert result["A"].max() < 100
