import numpy as np
import pandas as pd

try:
    from .clean_utils import load_csv, safe_parse_list, save_csv
except ImportError:
    from clean_utils import load_csv, safe_parse_list, save_csv


def clean_keywords() -> pd.DataFrame:
    df = load_csv("keywords.csv")
    original_len = len(df)

    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype(int)
    df = df.drop_duplicates(subset=["id"])
    print(f"  ID cleaning/deduplication: removed {before - len(df)}")

    df["keywords_list"] = df["keywords"].apply(safe_parse_list)
    df = df[["id", "keywords_list"]].explode("keywords_list")
    df = df[df["keywords_list"].notna()].copy()

    df["keyword_id"] = df["keywords_list"].apply(
        lambda x: x.get("id") if isinstance(x, dict) else np.nan
    )
    df["keyword_name"] = df["keywords_list"].apply(
        lambda x: x.get("name") if isinstance(x, dict) else np.nan
    )

    df = df.drop(columns=["keywords_list"])
    df = df.dropna(subset=["keyword_id", "keyword_name"])
    df["keyword_id"] = pd.to_numeric(df["keyword_id"], errors="coerce")
    df = df.dropna(subset=["keyword_id"]).copy()
    df["keyword_id"] = df["keyword_id"].astype(int)
    df = df.drop_duplicates(subset=["id", "keyword_id", "keyword_name"])
    df = df[["id", "keyword_id", "keyword_name"]]

    print(f"\n  Final row count: {len(df)}  (original: {original_len})")
    save_csv(df, "keywords_cleaned.csv")
    return df


def main() -> None:
    clean_keywords()


if __name__ == "__main__":
    main()