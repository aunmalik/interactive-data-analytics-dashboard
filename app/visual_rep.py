import plotly.express as px
import pandas as pd

from config import MAX_CHART_CATEGORIES


def histogram(df, column):
    # Histogram for numeric column distribution
    if column not in df.columns:
        return None

    if not pd.api.types.is_numeric_dtype(df[column]):
        return None

    fig = px.histogram(
        df,
        x=column,
        title=f"Distribution of {column}",
        nbins=30,
        color_discrete_sequence=["#4F81BD"],
    )
    
    fig.update_layout(
        bargap=0.05,
        xaxis_title=column,
        yaxis_title="Count",
    )
    
    return fig


def bar_chart(df, column):
    # Bar chart for categorical frequency
    if df is None or column not in df.columns:
        return None

    counts = df[column].astype(str).value_counts(dropna=False).reset_index()
    counts.columns = [column, "Count"]

    if len(counts) > MAX_CHART_CATEGORIES:
        counts = counts.head(MAX_CHART_CATEGORIES)
        title = f"Top {MAX_CHART_CATEGORIES}: {column}"
    else:
        title = f"Frequency of {column}"

    return px.bar(counts, x=column, y="Count", title=title)


def scatter_plot(df, x_col, y_col):
    # Scatter plot between two numeric variables
    if x_col not in df.columns or y_col not in df.columns:
        return None
        
    if not pd.api.types.is_numeric_dtype(df[x_col]):
        return None
    if not pd.api.types.is_numeric_dtype(df[y_col]):
        return None

    return px.scatter(
        df,
        x=x_col,
        y=y_col,
        title=f"{y_col} vs {x_col}",
        opacity=0.65,
    )


def line_chart(df, x_col, y_col):
    # Line chart
    if x_col not in df.columns or y_col not in df.columns:
        return None
        
    if not pd.api.types.is_numeric_dtype(df[y_col]):
        return None

    # Sorting 
    df_sorted = df.sort_values(x_col)

    return px.line(
        df_sorted,
        x=x_col,
        y=y_col,
        title=f"{y_col} over {x_col}",
    )


def box_plot(df, column):
    # Box plot showing distribution and outliers
    if column not in df.columns:
        return None
        
    if not pd.api.types.is_numeric_dtype(df[column]):
        return None

    return px.box(df, y=column, title=f"Box Plot: {column}")


def pie_chart(df, column):
    # Pie chart for proportional data
    if df is None or column not in df.columns:
        return None

    counts = df[column].astype(str).value_counts(dropna=False).reset_index()
    counts.columns = [column, "Count"]

    if len(counts) > MAX_CHART_CATEGORIES:
        counts = counts.head(MAX_CHART_CATEGORIES)
        title = f"{column} (Top {MAX_CHART_CATEGORIES})"
    else:
        title = f"Distribution: {column}"

    return px.pie(counts, names=column, values="Count", title=title)


def correlation_heatmap(df):
    # Heatmap showing correlations between numeric variables
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        title="Correlation Heatmap",
        color_continuous_scale="RdYlBu",
        aspect="auto",
        zmin=-1,
        zmax=1,
    )

    fig.update_xaxes(side="bottom")
    fig.update_layout(height=600)

    return fig