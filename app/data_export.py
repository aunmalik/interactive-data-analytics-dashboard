from io import BytesIO
import pandas as pd


def dataframe_to_csv_bytes(df):
    # Convert dataframe to CSV bytes for download
    return df.to_csv(index=False).encode("utf-8")


def dataframe_to_excel_bytes(df, sheet_name="Sheet1"):
    # Convert dataframe to Excel bytes for download
    # Excel sheet names have 31 char limit
    safe_name = (sheet_name or "Sheet1")[:31]

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=safe_name)

    return output.getvalue()