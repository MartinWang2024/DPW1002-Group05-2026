import ast
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = ROOT_DIR / "archive"
CLEANED_DIR = ROOT_DIR / "cleaned_archive"


def load_csv(filename: str, **kwargs) -> pd.DataFrame:
    path = ARCHIVE_DIR / filename
    df = pd.read_csv(path, **kwargs)
    print(f"\n{'=' * 60}")
    print(f"[Load] {filename}  shape={df.shape}")
    return df


def save_csv(df: pd.DataFrame, filename: str) -> None:
    path = CLEANED_DIR / filename
    df.to_csv(path, index=False)
    print(f"[Save] {filename}  shape={df.shape}  -> {path}")


def iqr_bounds(series: pd.Series, k: float = 1.5) -> tuple[float, float]:
    pos = series[series > 0].dropna()
    q1, q3 = pos.quantile(0.25), pos.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def safe_parse_list(val) -> list:
    if pd.isna(val) or val == "":
        return []
    try:
        result = ast.literal_eval(val)
        return result if isinstance(result, list) else []
    except Exception:
        return []


def extract_names(val, key: str = "name") -> list:
    items = safe_parse_list(val)
    return [item.get(key, "") for item in items if isinstance(item, dict) and item.get(key)]