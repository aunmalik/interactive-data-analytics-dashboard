import pandas as pd

from app.stats_sum import (
    dataset_overview,
    missing_values_analysis,
    numeric_summary,
    categorical_summary,
    outlier_summary,
    data_quality_score,
)


def test_dataset_overview():
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    overview = dataset_overview(df)

    assert "Rows" in overview["Metric"].values
    assert "Columns" in overview["Metric"].values
    assert overview.shape[1] == 2


def test_missing_values_analysis():
    df = pd.DataFrame({"A": [1, None, 3], "B": [4, 5, 6]})
    missing = missing_values_analysis(df)

    assert len(missing) == 1
    assert "A" in missing["Column"].values
    assert missing["Missing"].iloc[0] == 1


def test_missing_values_analysis_no_missing():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    missing = missing_values_analysis(df)
    assert len(missing) == 0


def test_missing_values_analysis_empty_df():
    df = pd.DataFrame()
    missing = missing_values_analysis(df)
    assert list(missing.columns) == ["Column", "Missing", "Missing %", "Row Numbers"]


def test_numeric_summary():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    summary = numeric_summary(df)

    assert summary is not None
    assert "Mean" in summary.columns
    assert len(summary) == 2


def test_numeric_summary_no_numeric():
    df = pd.DataFrame({"A": ["x", "y", "z"]})
    summary = numeric_summary(df)
    assert summary is None


def test_categorical_summary():
    df = pd.DataFrame({"Category": ["A", "A", "B"]})
    counts = categorical_summary(df, "Category")

    assert len(counts) == 2
    assert counts["Frequency"].sum() == 3


def test_outlier_summary():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 100]})
    summary = outlier_summary(df)

    assert summary is not None
    assert summary["Outliers"].values[0] > 0


def test_data_quality_score():
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50]})
    score = data_quality_score(df)

    assert 0 <= score <= 100
    assert score > 90
