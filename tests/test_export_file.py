import pandas as pd
import pytest

from app.data_export import dataframe_to_csv_bytes, dataframe_to_excel_bytes


def test_dataframe_to_csv_bytes():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    csv_bytes = dataframe_to_csv_bytes(df)

    assert isinstance(csv_bytes, bytes)
    text = csv_bytes.decode("utf-8")
    assert "A,B" in text


def test_dataframe_to_csv_empty():
    df = pd.DataFrame({"A": [], "B": []})
    csv_bytes = dataframe_to_csv_bytes(df)
    assert isinstance(csv_bytes, bytes)


def test_dataframe_to_excel_bytes():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    try:
        excel_bytes = dataframe_to_excel_bytes(df)
    except ImportError:
        pytest.skip("Excel export requires openpyxl")

    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 0