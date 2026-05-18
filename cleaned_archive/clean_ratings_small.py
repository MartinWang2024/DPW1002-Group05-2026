import numpy as np
import pandas as pd

try:
    from .clean_utils import load_csv, save_csv
except ImportError:
    from clean_utils import load_csv, save_csv


def clean_ratings_small() -> pd.DataFrame:
    df = load_csv("ratings_small.csv")
    original_len = len(df)

    before = len(df)
    df = df.drop_duplicates(subset=["userId", "movieId"])
    print(f"  Deduplication (userId+movieId): removed {before - len(df)}")

    valid_ratings = np.arange(0.5, 5.5, 0.5)
    before = len(df)
    df = df[df["rating"].isin(valid_ratings)].copy()
    print(f"  Invalid ratings: removed {before - len(df)}")

    df["rating_datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    missing = df.isnull().sum().sum()
    print(f"  Missing values: {missing}")

    print(f"\n  Final row count: {len(df)}  (original: {original_len})")
    save_csv(df, "ratings_small_cleaned.csv")
    return df


def main() -> None:
    clean_ratings_small()


if __name__ == "__main__":
    main()