import pandas as pd
import numpy as np


def dataset_overview(df):
    # Basic dataset statistics for quick overview
    overview = {
        "Rows": int(df.shape[0]),
        "Columns": int(df.shape[1]),
        "Numeric columns": int(len(df.select_dtypes(include="number").columns)),
        "Non-numeric columns": int(len(df.select_dtypes(exclude="number").columns)),
        "Missing values": int(df.isnull().sum().sum()),
        "Duplicate rows": int(df.duplicated().sum()),
        "Memory (MB)": round(df.memory_usage(deep=True).sum() / (1024**2), 2),
    }
    return pd.DataFrame(list(overview.items()), columns=["Metric", "Value"])


def missing_values_analysis(df):
    # Shows which columns have missing data with row numbers
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["Column", "Missing", "Missing %", "Row Numbers"])

    missing_counts = df.isnull().sum()

    row_numbers = []
    for col in df.columns:
        missing_rows = df.index[df[col].isnull()] + 1  
        if len(missing_rows) > 0:
            if len(missing_rows) > 5:
                row_str = ", ".join(map(str, missing_rows[:5])) + f"... (+{len(missing_rows)-5})"
            else:
                row_str = ", ".join(map(str, missing_rows.tolist()))
        else:
            row_str = ""
        row_numbers.append(row_str)

    miss = pd.DataFrame({
        "Column": df.columns,
        "Missing": missing_counts.values,
        "Missing %": (missing_counts.values / len(df) * 100).round(2),
        "Row Numbers": row_numbers,
    })

    miss = miss[miss["Missing"] > 0]
    return miss.sort_values("Missing", ascending=False).reset_index(drop=True)


def numeric_summary(df):
    # Statistical summary for numeric columns
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) == 0:
        return None

    summary = pd.DataFrame({
        "Column": num_cols,
        "Count": df[num_cols].count().values,
        "Missing": df[num_cols].isnull().sum().values,
        "Mean": df[num_cols].mean().values.round(3),
        "Std Dev": df[num_cols].std().values.round(3),
        "Min": df[num_cols].min().values.round(3),
        "25%": df[num_cols].quantile(0.25).values.round(3),
        "Median": df[num_cols].median().values.round(3),
        "75%": df[num_cols].quantile(0.75).values.round(3),
        "Max": df[num_cols].max().values.round(3),
    })

    return summary


def categorical_summary(df, column):
    # Frequency distribution for a categorical column
    if column not in df.columns:
        return None

    series = df[column].copy()
    series = series.astype("object").where(series.notna(), "Missing")

    vc = series.value_counts(dropna=False).reset_index()
    vc.columns = ["Value", "Frequency"]
    vc["Percentage"] = (vc["Frequency"] / vc["Frequency"].sum() * 100).round(2)

    return vc


def outlier_summary(df):
    # Detect outliers in numeric columns using IQR method
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) == 0:
        return None

    results = []
    for col in num_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((series < lower) | (series > upper)).sum())

        results.append({
            "Column": col,
            "Outliers": outlier_count,
            "Outlier %": round(outlier_count / len(series) * 100, 2),
            "Lower Bound": round(lower, 2),
            "Upper Bound": round(upper, 2),
        })

    if not results:
        return None

    return pd.DataFrame(results)


def data_quality_score(df):
    # Overall data quality score (0-100)
    # Weights: Missing=40, Duplicates=30, Outliers=30
    if df is None or len(df) == 0 or df.shape[1] == 0:
        return 0.0

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isnull().sum().sum())

    missing_penalty = (missing_cells / total_cells) * 40 if total_cells else 0
    dup_penalty = (df.duplicated().sum() / len(df)) * 30 if len(df) else 0

    outlier_penalty = 0.0
    num_cols = df.select_dtypes(include="number").columns

    if len(num_cols) > 0:
        per_col = []
        for col in num_cols:
            series = df[col].dropna()
            if len(series) < 4:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            if pd.isna(iqr) or iqr == 0:
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_count = int(((series < lower) | (series > upper)).sum())

            per_col.append((outlier_count / len(series)) * 10)

        if per_col:
            avg_outliers = sum(per_col) / len(per_col)
            outlier_penalty = (avg_outliers / 10) * 30

    score = 100 - missing_penalty - dup_penalty - outlier_penalty
    return float(max(0, min(100, round(score, 1))))