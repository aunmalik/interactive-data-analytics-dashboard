import pandas as pd


def filter_numeric_range(df, column, min_val=None, max_val=None):
    # Filter numeric column by min/max range
    if df is None or column not in df.columns:
        return df

    if not pd.api.types.is_numeric_dtype(df[column]):
        return df

    result = df.copy()

    # Keep empty rows when filtering
    if min_val is not None:
        result = result[result[column].isna() | (result[column] >= min_val)]

    if max_val is not None:
        result = result[result[column].isna() | (result[column] <= max_val)]

    return result.reset_index(drop=True)


def filter_by_values(df, column, values, include=True):
    # Filter rows by specific values in a column
    if df is None or column not in df.columns:
        return df

    if values is None or len(values) == 0:
        return df

    if include:
        result = df[df[column].isin(values)]
    else:
        result = df[~df[column].isin(values)]

    return result.reset_index(drop=True)