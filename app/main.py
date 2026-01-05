import streamlit as st
import pandas as pd
import copy

from config import APP_TITLE
from data_loader import load_dataset, validate_dataframe
from data_cleaning import (
    remove_duplicates,
    drop_missing_rows,
    fill_numeric_with_mean,
    fill_numeric_with_median,
    remove_outliers_iqr,
    cap_outliers_iqr,
)
from data_standard import (
    normalize_column,
    standardize_column,
    log_transform,
    bin_column,
    percentage_of_total,
    rank_column,
    calculate_growth,
)
from stats_sum import (
    dataset_overview,
    missing_values_analysis,
    numeric_summary,
    categorical_summary,
    outlier_summary,
    data_quality_score,
)
from visual_rep import (
    histogram,
    bar_chart,
    scatter_plot,
    line_chart,
    box_plot,
    pie_chart,
    correlation_heatmap,
)
from data_filters import filter_numeric_range, filter_by_values
from data_export import dataframe_to_csv_bytes
from ui_components import apply_base_layout, render_headline


# Page setup
st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_base_layout()


# Session state
if "data" not in st.session_state:
    st.session_state.data = None

if "original_data" not in st.session_state:
    st.session_state.original_data = None

if "last_file_name" not in st.session_state:
    st.session_state.last_file_name = None

if "history" not in st.session_state:
    st.session_state.history = []

if "history_index" not in st.session_state:
    st.session_state.history_index = -1

HISTORY_LIMIT = 25


# Helper functions
def dataset_loaded():
    return st.session_state.get("data") is not None


def get_df():
    return st.session_state.get("data")


def safe_base_filename():
    # Get clean filename for exports
    name = st.session_state.get("last_file_name") or "dataset"
    base = name.rsplit(".", 1)[0]
    base = base.strip().replace(" ", "_")
    return base or "dataset"


def _push_history(df, reset=False):
    # Save dataframe state for undo/redo
    if reset:
        st.session_state.history = [df.copy()]
        st.session_state.history_index = 0
        return

    idx = st.session_state.history_index
    if 0 <= idx < len(st.session_state.history) - 1:
        st.session_state.history = st.session_state.history[:idx + 1]

    st.session_state.history.append(df.copy())
    st.session_state.history_index += 1

    # Keep history under limit
    if len(st.session_state.history) > HISTORY_LIMIT:
        overflow = len(st.session_state.history) - HISTORY_LIMIT
        st.session_state.history = st.session_state.history[overflow:]
        st.session_state.history_index = max(0, st.session_state.history_index - overflow)


def set_df(df, filename=None, set_original=False):
    # Update dataset in session state
    st.session_state.data = df.copy()

    if set_original or st.session_state.get("original_data") is None:
        st.session_state.original_data = df.copy()

    if filename:
        st.session_state.last_file_name = filename

    _push_history(df, reset=set_original)


def undo_history():
    # one step back
    if st.session_state.history_index > 0:
        st.session_state.history_index -= 1
        st.session_state.data = st.session_state.history[st.session_state.history_index].copy()
        return True
    return False


def redo_history():
    # one step forward
    if st.session_state.history_index < len(st.session_state.history) - 1:
        st.session_state.history_index += 1
        st.session_state.data = st.session_state.history[st.session_state.history_index].copy()
        return True
    return False


def reset_to_original():
    # Reset to first loaded version
    if st.session_state.get("original_data") is None:
        return False

    st.session_state.data = st.session_state.original_data.copy()
    _push_history(st.session_state.data, reset=True)
    return True


def clear_dataset():
    # Remove dataset from memory
    st.session_state.data = None
    st.session_state.original_data = None
    st.session_state.last_file_name = None
    st.session_state.history = []
    st.session_state.history_index = -1


def plotly_png_bytes(fig):
    # Export chart as PNG with colors
    if fig is None:
        return None
    try:
        import plotly.io as pio

        fig_copy = copy.deepcopy(fig)
        colors = ["#4F81BD", "#C0504D", "#9BBB59", "#8064A2", "#4BACC6", "#F79646"]

        # Force colors for each trace
        for i, trace in enumerate(fig_copy.data):
            color = colors[i % len(colors)]
            try:
                if hasattr(trace, 'marker') and trace.marker is not None:
                    if not isinstance(trace.marker.color, (list, tuple)) or len(trace.marker.color) == 0:
                        trace.marker.color = color
                if hasattr(trace, 'line') and trace.line is not None:
                    trace.line.color = color
            except Exception:
                pass

        fig_copy.update_layout(
            template="simple_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="black"),
        )

        return pio.to_image(fig_copy, format="png", scale=2, engine="kaleido")
    except Exception:
        return None


# Sidebar
def sidebar_block():
    with st.sidebar:
        st.markdown("### Dataset Info")

        df = get_df()
        if df is None:
            st.info("No dataset loaded yet.")
            return

        file_name = st.session_state.get("last_file_name")
        if file_name:
            st.write(f"**File:** {file_name}")

        st.write(f"**Rows:** {df.shape[0]:,}")
        st.write(f"**Columns:** {df.shape[1]}")
        st.write(f"**Missing values:** {int(df.isnull().sum().sum()):,}")
        st.write(f"**Duplicates:** {int(df.duplicated().sum()):,}")

        size_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.write(f"**Size:** {size_mb:.2f} MB")

        history = st.session_state.get("history", [])
        idx = st.session_state.get("history_index", -1)
        if history and idx >= 0:
            st.caption(f"History: step {idx + 1} of {len(history)}")

        st.markdown("---")

        can_undo = idx > 0
        can_redo = history and (0 <= idx < len(history) - 1)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Undo", use_container_width=True, disabled=not can_undo):
                if undo_history():
                    st.rerun()

        with c2:
            if st.button("Redo", use_container_width=True, disabled=not can_redo):
                if redo_history():
                    st.rerun()

        st.markdown("---")

        c3, c4 = st.columns(2)
        with c3:
            if st.button("Reset", use_container_width=True):
                if reset_to_original():
                    st.rerun()

        with c4:
            if st.button("Clear", use_container_width=True):
                clear_dataset()
                st.rerun()


# UI blocks
def upload_block(key_prefix="upload"):
    st.markdown("### Upload")

    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
        key=f"{key_prefix}_uploader",
        help="CSV / XLSX / XLS (max 200MB)",
    )

    if uploaded_file is None:
        st.info("Choose a file to begin.")
        return

    if st.button("Load file", use_container_width=True, key=f"{key_prefix}_load_btn"):
        try:
            df = load_dataset(uploaded_file)
            ok, msg = validate_dataframe(df)

            if not ok:
                st.error(msg)
                return

            set_df(df, filename=uploaded_file.name, set_original=True)
            st.success(f"Loaded: {uploaded_file.name} ({df.shape[0]:,} rows × {df.shape[1]} columns)")
            st.rerun()

        except Exception as e:
            st.error(f"Upload error: {e}")


def preview_block(df, key_prefix="preview"):
    st.markdown("### Dataset preview")

    row_options = [10, 25, 50, 100, "All"]
    rows = st.selectbox(
        "Rows to display",
        row_options,
        index=4,
        key=f"{key_prefix}_rows",
    )

    if rows == "All":
        display_df = df.copy()
    else:
        display_df = df.head(rows).copy()

    display_df.index = range(1, len(display_df) + 1)
    st.dataframe(display_df, use_container_width=True)


def overview_block(df):
    st.markdown("### Overview")

    quality = data_quality_score(df)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Missing", f"{int(df.isnull().sum().sum()):,}")
    c4.metric("Duplicates", f"{int(df.duplicated().sum()):,}")
    c5.metric("Quality", f"{quality}/100")


def cleaning_block(df, key_prefix="clean"):
    st.markdown("### Data cleaning")

    dup_count = int(df.duplicated().sum())
    miss_total = int(df.isnull().sum().sum())

    left, right = st.columns(2)

    with left:
        st.markdown("**Duplicates**")
        if dup_count == 0:
            st.success("No duplicate rows found.")
        else:
            dup_rows = df[df.duplicated()].index + 1
            dup_rows_list = dup_rows.tolist()
            if len(dup_rows_list) > 10:
                dup_display = ", ".join(map(str, dup_rows_list[:10])) + f"... (+{len(dup_rows_list)-10} more)"
            else:
                dup_display = ", ".join(map(str, dup_rows_list))

            st.info(f"Found {dup_count} duplicate rows: Row {dup_display}")
            if st.button("Remove duplicates", use_container_width=True, key=f"{key_prefix}_rm_dups"):
                before = len(df)
                new_df = remove_duplicates(df)
                set_df(new_df)
                st.success(f"Removed {before - len(new_df)} rows.")
                st.rerun()

    with right:
        st.markdown("**Missing values**")
        if miss_total == 0:
            st.success("No missing values found.")
        else:
            st.info(f"Total missing values: {miss_total:,}")

            miss_df = missing_values_analysis(df)
            if miss_df is not None and not miss_df.empty:
                st.dataframe(miss_df.head(8), use_container_width=True, hide_index=True)

            method = st.selectbox(
                "Method",
                ["Drop rows", "Fill numeric with mean", "Fill numeric with median"],
                key=f"{key_prefix}_miss_method",
            )

            if st.button("Apply", use_container_width=True, key=f"{key_prefix}_miss_apply"):
                if method == "Drop rows":
                    new_df = drop_missing_rows(df)
                elif method == "Fill numeric with mean":
                    new_df = fill_numeric_with_mean(df)
                else:
                    new_df = fill_numeric_with_median(df)

                set_df(new_df)
                st.success("Done.")
                st.rerun()

    st.markdown("---")
    st.markdown("**Outlier handling (IQR)**")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        st.info("No numeric columns available.")
        return

    c1, c2, c3 = st.columns(3)

    with c1:
        col = st.selectbox("Column", num_cols, key=f"{key_prefix}_out_col")

    with c2:
        action = st.selectbox("Action", ["Remove outliers", "Cap outliers"], key=f"{key_prefix}_out_action")

    with c3:
        multiplier = st.number_input("IQR multiplier", value=1.5, min_value=0.5, max_value=10.0, step=0.5, key=f"{key_prefix}_out_mult")

    if st.button("Apply outlier treatment", use_container_width=True, key=f"{key_prefix}_out_apply"):
        if action == "Remove outliers":
            new_df = remove_outliers_iqr(df, col, multiplier=multiplier)
        else:
            new_df = cap_outliers_iqr(df, col, multiplier=multiplier)

        set_df(new_df)
        st.success("Applied.")
        st.rerun()


def transform_block(df, key_prefix="transform"):
    st.markdown("### Data Transformations")
    st.markdown("Add new calculated columns based on existing data.")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        st.info("No numeric columns available.")
        return

    col1, col2 = st.columns(2)

    with col1:
        col = st.selectbox("Column to transform", num_cols, key=f"{key_prefix}_col")

    with col2:
        transform_type = st.selectbox(
            "Transformation type",
            ["Normalize (0-1 range)", "Standardize (z-score)", "Log transform",
             "Bin into categories", "Percentage of total", "Rank values", "Calculate growth"],
            key=f"{key_prefix}_type",
        )

    bins = 5
    if transform_type == "Bin into categories":
        bins = st.slider("Number of bins", min_value=2, max_value=10, value=5, key=f"{key_prefix}_bins")

    ascending = True
    if transform_type == "Rank values":
        ascending = st.checkbox("Ascending order (1 = lowest)", value=True, key=f"{key_prefix}_asc")

    st.markdown("**Column preview (before)**")
    before_df = df[[col]].head(10).copy()
    before_df.index = range(1, len(before_df) + 1)
    st.dataframe(before_df, use_container_width=True)

    if st.button("Apply transformation", use_container_width=True, key=f"{key_prefix}_apply"):
        if transform_type == "Normalize (0-1 range)":
            new_df = normalize_column(df, col)
            new_col_name = f"{col}_normalized"
        elif transform_type == "Standardize (z-score)":
            new_df = standardize_column(df, col)
            new_col_name = f"{col}_standardized"
        elif transform_type == "Log transform":
            new_df = log_transform(df, col)
            new_col_name = f"{col}_log"
        elif transform_type == "Bin into categories":
            new_df = bin_column(df, col, bins=bins)
            new_col_name = f"{col}_binned"
        elif transform_type == "Percentage of total":
            new_df = percentage_of_total(df, col)
            new_col_name = f"{col}_pct"
        elif transform_type == "Rank values":
            new_df = rank_column(df, col, ascending=ascending)
            new_col_name = f"{col}_rank"
        else:
            new_df = calculate_growth(df, col)
            new_col_name = f"{col}_growth"

        if new_col_name in new_df.columns:
            set_df(new_df)
            st.success(f"Added column: {new_col_name}")
            st.rerun()
        else:
            st.error("Transformation failed.")

    current_df = get_df()
    if current_df is None:
        return

    transformed_cols = [c for c in current_df.columns
                        if any(c.endswith(s) for s in ["_normalized", "_standardized", "_log", "_binned", "_pct", "_rank", "_growth"])]

    if transformed_cols:
        st.markdown("---")
        st.markdown("**Transformed columns**")

        view_col = st.selectbox("View transformation for", options=list(set([tc.rsplit("_", 1)[0] for tc in transformed_cols])), key=f"{key_prefix}_view_col")

        related_cols = [view_col] + [tc for tc in transformed_cols if tc.startswith(view_col + "_")]
        related_cols = [c for c in related_cols if c in current_df.columns]

        show_df = current_df[related_cols].reset_index(drop=True).copy()
        show_df.index = range(1, len(show_df) + 1)
        st.dataframe(show_df, use_container_width=True)


def filtering_block(df, key_prefix="filter"):
    st.markdown("### Filters")

    mode = st.selectbox("Filter type", ["Numeric range", "Categorical values"], key=f"{key_prefix}_mode")

    if mode == "Numeric range":
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            st.info("No numeric columns available.")
            return

        col = st.selectbox("Column", num_cols, key=f"{key_prefix}_num_col")
        col_min = float(df[col].min()) if len(df) else 0.0
        col_max = float(df[col].max()) if len(df) else 0.0

        a, b = st.columns(2)
        with a:
            min_val = st.number_input("Min", value=col_min, key=f"{key_prefix}_min_val")
        with b:
            max_val = st.number_input("Max", value=col_max, key=f"{key_prefix}_max_val")

        if st.button("Apply filter", use_container_width=True, key=f"{key_prefix}_apply_num"):
            filtered = filter_numeric_range(df, col, min_val=min_val, max_val=max_val)
            set_df(filtered)
            st.success(f"Filtered: {len(df):,} → {len(filtered):,} rows")
            st.rerun()

    else:
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()
        if not cat_cols:
            st.info("No categorical columns available.")
            return

        col = st.selectbox("Column", cat_cols, key=f"{key_prefix}_cat_col")
        values = df[col].dropna().unique().tolist()

        selected = st.multiselect("Values to keep", options=values, default=values[:5] if len(values) > 5 else values, key=f"{key_prefix}_cat_vals")

        if st.button("Apply filter", use_container_width=True, key=f"{key_prefix}_apply_cat"):
            filtered = filter_by_values(df, col, selected, include=True)
            set_df(filtered)
            st.success(f"Filtered: {len(df):,} → {len(filtered):,} rows")
            st.rerun()

    st.markdown("---")
    st.markdown(" use Reset in the sidebar to go back to the original dataset.")


def stats_block(df, key_prefix="stats"):
    st.markdown("### Statistics")

    view = st.selectbox("View", ["Overview", "Numeric", "Categorical", "Outliers"], key=f"{key_prefix}_view")

    if view == "Overview":
        ov = dataset_overview(df)
        st.dataframe(ov, hide_index=True, use_container_width=True)

    elif view == "Numeric":
        num = numeric_summary(df)
        if num is None:
            st.info("No numeric columns found.")
        else:
            st.dataframe(num, hide_index=True, use_container_width=True)

    elif view == "Categorical":
        cols = df.select_dtypes(exclude="number").columns.tolist()
        if not cols:
            st.info("No categorical columns found.")
        else:
            col = st.selectbox("Column", cols, key=f"{key_prefix}_cat_col")
            out = categorical_summary(df, col)
            if out is not None:
                st.dataframe(out.head(50), hide_index=True, use_container_width=True)

    else:
        outs = outlier_summary(df)
        if outs is None:
            st.info("No numeric columns found.")
        else:
            st.dataframe(outs, hide_index=True, use_container_width=True)


def viz_block(df, key_prefix="viz"):
    st.markdown("### Visualizations")

    chart = st.selectbox("Chart type", ["Histogram", "Bar", "Scatter", "Line", "Box", "Pie", "Correlation"], key=f"{key_prefix}_chart_type")

    fig = None

    if chart == "Histogram":
        cols = df.select_dtypes(include="number").columns.tolist()
        if cols:
            col = st.selectbox("Column", cols, key=f"{key_prefix}_hist_col")
            fig = histogram(df, col)

    elif chart == "Bar":
        col = st.selectbox("Column", df.columns.tolist(), key=f"{key_prefix}_bar_col")
        fig = bar_chart(df, col)

    elif chart == "Scatter":
        cols = df.select_dtypes(include="number").columns.tolist()
        if len(cols) >= 2:
            x = st.selectbox("X-axis", cols, key=f"{key_prefix}_sc_x")
            y = st.selectbox("Y-axis", cols, key=f"{key_prefix}_sc_y")
            fig = scatter_plot(df, x, y)

    elif chart == "Line":
        y_cols = df.select_dtypes(include="number").columns.tolist()
        if y_cols:
            x = st.selectbox("X-axis", df.columns.tolist(), key=f"{key_prefix}_ln_x")
            y = st.selectbox("Y-axis", y_cols, key=f"{key_prefix}_ln_y")
            fig = line_chart(df, x, y)

    elif chart == "Box":
        cols = df.select_dtypes(include="number").columns.tolist()
        if cols:
            col = st.selectbox("Column", cols, key=f"{key_prefix}_box_col")
            fig = box_plot(df, col)

    elif chart == "Pie":
        col = st.selectbox("Column", df.columns.tolist(), key=f"{key_prefix}_pie_col")
        fig = pie_chart(df, col)

    else:
        fig = correlation_heatmap(df)

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chart_render")

        png = plotly_png_bytes(fig)
        if png:
            st.download_button("Download chart (PNG)", data=png, file_name=f"chart_{safe_base_filename()}.png", mime="image/png", use_container_width=True, key=f"{key_prefix}_dl_png")


def export_block(df, key_prefix="export"):
    st.markdown("### Export")

    c1, c2 = st.columns(2)

    with c1:
        st.download_button("Download dataset (CSV)", data=dataframe_to_csv_bytes(df), file_name=f"{safe_base_filename()}_export.csv", mime="text/csv", use_container_width=True, key=f"{key_prefix}_dl_data")

    with c2:
        ov = dataset_overview(df)
        st.download_button("Download overview (CSV)", data=dataframe_to_csv_bytes(ov), file_name=f"overview_{safe_base_filename()}.csv", mime="text/csv", use_container_width=True, key=f"{key_prefix}_dl_overview")

    st.markdown("---")
    st.markdown("**Preview (first 25 rows)**")
    st.dataframe(df.head(25), use_container_width=True)


# Render app
sidebar_block()
render_headline()

tabs = st.tabs(["Home", "Clean", "Transform", "Filter", "Stats", "Charts", "Export"])

with tabs[0]:
    upload_block(key_prefix="home_upload")
    if not dataset_loaded():
        st.info("Upload a dataset to start.")
    else:
        df = get_df()
        preview_block(df, key_prefix="home_preview")
        st.markdown("---")
        overview_block(df)
        st.markdown("---")
        cleaning_block(get_df(), key_prefix="home_clean")
        st.markdown("---")
        transform_block(get_df(), key_prefix="home_transform")
        st.markdown("---")
        stats_block(get_df(), key_prefix="home_stats")
        st.markdown("---")
        viz_block(get_df(), key_prefix="home_viz")
        st.markdown("---")
        export_block(get_df(), key_prefix="home_export")

with tabs[1]:
    if not dataset_loaded():
        st.info("Upload data in the Home tab first.")
    else:
        preview_block(get_df(), key_prefix="clean_preview")
        st.markdown("---")
        cleaning_block(get_df(), key_prefix="clean_tab")

with tabs[2]:
    if not dataset_loaded():
        st.info("Upload data in the Home tab first.")
    else:
        preview_block(get_df(), key_prefix="transform_preview")
        st.markdown("---")
        transform_block(get_df(), key_prefix="transform_tab")

with tabs[3]:
    if not dataset_loaded():
        st.info("Upload data in the Home tab first.")
    else:
        preview_block(get_df(), key_prefix="filter_preview")
        st.markdown("---")
        filtering_block(get_df(), key_prefix="filter_tab")

with tabs[4]:
    if not dataset_loaded():
        st.info("Upload data in the Home tab first.")
    else:
        preview_block(get_df(), key_prefix="stats_preview")
        st.markdown("---")
        stats_block(get_df(), key_prefix="stats_tab")

with tabs[5]:
    if not dataset_loaded():
        st.info("Upload data in the Home tab first.")
    else:
        preview_block(get_df(), key_prefix="viz_preview")
        st.markdown("---")
        viz_block(get_df(), key_prefix="viz_tab")

with tabs[6]:
    if not dataset_loaded():
        st.info("Upload data in the Home tab first.")
    else:
        preview_block(get_df(), key_prefix="export_preview")
        st.markdown("---")
        export_block(get_df(), key_prefix="export_tab")