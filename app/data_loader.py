import pandas as pd
from config import MAX_FILE_SIZE_MB, ALLOWED_FILE_TYPES


def load_dataset(uploaded_file):
    # Load a CSV or Excel file from Streamlit uploader
    if uploaded_file is None:
        raise ValueError("No file provided.")

    # Check file size
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File too large ({size_mb:.2f} MB). Max: {MAX_FILE_SIZE_MB} MB")

    filename = (uploaded_file.name or "").lower()

    # Check file type
    if not any(filename.endswith(ext) for ext in ALLOWED_FILE_TYPES):
        raise ValueError(f"Unsupported format. Allowed: {', '.join(ALLOWED_FILE_TYPES)}")

    if filename.endswith(".csv"):
        return _load_csv(uploaded_file)
    return _load_excel(uploaded_file)


def _load_csv(uploaded_file):
    # Try different encodings, had issues with files from different sources
    encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]

    for enc in encodings:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    raise ValueError("Couldn't read CSV file. Try saving as UTF-8.")


def _load_excel(uploaded_file):
    # Load Excel file
    uploaded_file.seek(0)
    try:
        return pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e:
        raise ValueError(f"Couldn't read Excel file: {e}")


def validate_dataframe(df):
    # Check if dataframe is usable
    if df is None or len(df) == 0:
        return False, "Dataset is empty"

    if df.columns is None or len(df.columns) == 0:
        return False, "No columns found"

    # Check for unnamed columns 
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        return False, f"Found {len(unnamed)} unnamed columns"

    # Check for duplicate column names
    if df.columns.duplicated().any():
        return False, "Duplicate column names detected"

    return True, "OK"