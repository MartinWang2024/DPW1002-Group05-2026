import pandas as pd

try:
    from .clean_utils import load_csv, save_csv
except ImportError:
    from clean_utils import load_csv, save_csv


def clean_links_small() -> pd.DataFrame:
    df = load_csv("links_small.csv")
    original_len = len(df)

    df["movieId"] = pd.to_numeric(df["movieId"], errors="coerce")
    df["tmdbId"] = pd.to_numeric(df["tmdbId"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["movieId", "tmdbId"])
    df = df.drop_duplicates(subset=["movieId"])
    print(f"  Cleaning/deduplication: removed {before - len(df)}")

    df["movieId"] = df["movieId"].astype(int)
    df["tmdbId"] = df["tmdbId"].astype(int)

    print(f"\n  Final row count: {len(df)}  (original: {original_len})")
    save_csv(df, "links_small_cleaned.csv")
    return df


def main() -> None:
    clean_links_small()


if __name__ == "__main__":
    main()