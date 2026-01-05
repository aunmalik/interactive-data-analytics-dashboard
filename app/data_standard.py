import pandas as pd
import numpy as np


def _is_valid_numeric(df, column):
    # Helper to check if column exists 
    return column in df.columns and pd.api.types.is_numeric_dtype(df[column])


def normalize_column(df, column):
    # Normalize a numeric column to 0-1 range
    if not _is_valid_numeric(df, column):
        return df

    result = df.copy()
    series = result[column]

    if series.dropna().empty:
        return df

    col_min = series.min()
    col_max = series.max()
    denom = col_max - col_min

    if pd.isna(denom) or denom == 0:
        result[f"{column}_normalized"] = 0.0
    else:
        result[f"{column}_normalized"] = (series - col_min) / denom

    return result


def standardize_column(df, column):
    # Standardize a numeric column to mean=0 and std=1 
    if not _is_valid_numeric(df, column):
        return df

    result = df.copy()
    series = result[column]

    if series.dropna().empty:
        return df

    mean_val = series.mean()
    std_val = series.std()

    if pd.isna(std_val) or std_val == 0:
        result[f"{column}_standardized"] = 0.0
    else:
        result[f"{column}_standardized"] = (series - mean_val) / std_val

    return result


def log_transform(df, column):
    # Apply log transform 
    if not _is_valid_numeric(df, column):
        return df

    result = df.copy()
    series = result[column]

    if (series.dropna() < 0).any():
        return df

    result[f"{column}_log"] = np.log1p(series)
    return result


def bin_column(df, column, bins=5):
    # Convert numeric column into categorical bins
    if not _is_valid_numeric(df, column):
        return df

    if bins < 2:
        return df

    result = df.copy()
    series = result[column]

    if series.dropna().empty:
        return df

    labels = [f"Bin_{i+1}" for i in range(bins)]

    try:
        result[f"{column}_binned"] = pd.cut(
            series,
            bins=bins,
            labels=labels,
            include_lowest=True,
        )
    except ValueError:
        return df

    return result


def percentage_of_total(df, column):
    # Add column showing each value as % of total
    if not _is_valid_numeric(df, column):
        return df

    result = df.copy()
    series = result[column]
    total = series.sum(skipna=True)

    if pd.isna(total) or total == 0:
        result[f"{column}_pct"] = 0.0
    else:
        result[f"{column}_pct"] = (series / total * 100).round(2)

    return result


def rank_column(df, column, ascending=True):
    # Add rank column based on values
    if not _is_valid_numeric(df, column):
        return df

    result = df.copy()
    series = result[column]

    ranks = series.rank(ascending=ascending, method="dense")
    result[f"{column}_rank"] = ranks

    return result


def calculate_growth(df, column):
    # Calculate row-over-row percentage change
    if not _is_valid_numeric(df, column):
        return df

    result = df.copy()
    growth = result[column].pct_change() * 100

    growth = growth.replace([np.inf, -np.inf], np.nan)

    result[f"{column}_growth"] = growth.round(2)
    return result