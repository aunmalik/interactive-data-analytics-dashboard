import pandas as pd


def remove_duplicates(df):
    # Remove duplicate rows
    return df.drop_duplicates().reset_index(drop=True)


def drop_missing_rows(df):
    # Remove rows with any missing values
    return df.dropna().reset_index(drop=True)


def fill_numeric_with_mean(df):
    # Fill missing values in numeric columns with mean
    result = df.copy()
    num_cols = result.select_dtypes(include="number").columns

    for col in num_cols:
        if result[col].isna().any():
            result[col] = result[col].fillna(result[col].mean())

    return result


def fill_numeric_with_median(df):
    # Fill missing values in numeric columns with median
    result = df.copy()
    num_cols = result.select_dtypes(include="number").columns

    for col in num_cols:
        if result[col].isna().any():
            result[col] = result[col].fillna(result[col].median())

    return result


def remove_outliers_iqr(df, column, multiplier=1.5):
    # Remove rows where column value is outside IQR bounds
    if column not in df.columns:
        return df

    if not pd.api.types.is_numeric_dtype(df[column]):
        return df

    series = df[column].dropna()
    if series.empty:
        return df

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    # If IQR is 0, all values are identical meaning no outliers
    if pd.isna(iqr) or iqr == 0:
        return df.reset_index(drop=True)

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    result = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return result.reset_index(drop=True)


def cap_outliers_iqr(df, column, multiplier=1.5):
    # Cap outliers to IQR bounds instead of removing them
    if column not in df.columns:
        return df

    if not pd.api.types.is_numeric_dtype(df[column]):
        return df

    result = df.copy()
    series = result[column].dropna()
    
    if series.empty:
        return result

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        return result

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    result[column] = result[column].clip(lower=lower_bound, upper=upper_bound)
    return result