import numpy as np
import pandas as pd

try:
    from .clean_utils import extract_names, iqr_bounds, load_csv, save_csv
except ImportError:
    from clean_utils import extract_names, iqr_bounds, load_csv, save_csv


def clean_movies_metadata() -> pd.DataFrame:
    df = load_csv("movies_metadata.csv", low_memory=False)
    original_len = len(df)

    df = df[df["id"].astype(str).str.match(r"^\d+$")].copy()
    df["id"] = df["id"].astype(int)
    print(f"  Removed malformed rows (non-integer id): {original_len - len(df)}")

    before = len(df)
    df = df.drop_duplicates(subset=["id"])
    df = df.drop_duplicates(subset=["imdb_id"])
    print(f"  Deduplication (id/imdb_id): removed {before - len(df)}")

    before = len(df)
    df = df[df["adult"].isin(["True", "False"])].copy()
    df["adult"] = df["adult"] == "True"
    print(f"  adult column cleaning: removed {before - len(df)} non-boolean rows")

    df["budget"] = pd.to_numeric(df["budget"], errors="coerce").fillna(0).astype(np.int64)
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0).astype(np.int64)

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year.astype("Int64")

    before = len(df)
    df = df.dropna(subset=["release_date"])
    print(f"  Removed missing release_date rows: {before - len(df)}")

    before = len(df)
    df = df[df["runtime"].fillna(0) > 0].copy()
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
    print(f"  Removed runtime=0 rows: {before - len(df)}")

    lo, hi = iqr_bounds(df["runtime"], k=1.5)
    lo = max(lo, 20.0)
    outlier_mask = (df["runtime"] < lo) | (df["runtime"] > hi)
    df["runtime_outlier"] = outlier_mask
    print(f"  runtime outliers (IQR k=1.5, range [{lo:.0f}, {hi:.0f}]): {outlier_mask.sum()} flagged")

    for col in ["budget", "revenue"]:
        lo_c, hi_c = iqr_bounds(df[col], k=3.0)
        lo_c = max(lo_c, 0)
        mask = (df[col] > 0) & ((df[col] < lo_c) | (df[col] > hi_c))
        df[f"{col}_outlier"] = mask
        print(f"  {col} outliers (IQR k=3.0, range [{lo_c:.0f}, {hi_c:.0f}]): {mask.sum()} flagged")

    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
    median_pop = df["popularity"].median()
    df["popularity"] = df["popularity"].fillna(median_pop)

    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce")
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0).astype(np.int64)

    before = len(df)
    df = df[~((df["vote_average"] == 0) & (df["vote_count"] == 0))].copy()
    print(f"  Removed vote=0 & count=0 rows: {before - len(df)}")

    df["genres_list"] = df["genres"].apply(lambda x: extract_names(x, "name"))
    df["production_companies_list"] = df["production_companies"].apply(
        lambda x: extract_names(x, "name")
    )
    df["production_countries_list"] = df["production_countries"].apply(
        lambda x: extract_names(x, "name")
    )
    df["spoken_languages_list"] = df["spoken_languages"].apply(
        lambda x: extract_names(x, "name")
    )

    df["is_collection"] = df["belongs_to_collection"].apply(
        lambda x: False if pd.isna(x) or x == "" else True
    )

    drop_cols = [
        "genres",
        "production_companies",
        "production_countries",
        "spoken_languages",
        "belongs_to_collection",
        "homepage",
        "poster_path",
        "tagline",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    column_order = [
        "id",
        "title",
        "original_title",
        "budget",
        "revenue",
        "runtime",
        "release_date",
        "release_year",
        "popularity",
        "vote_average",
        "vote_count",
        "budget_outlier",
        "revenue_outlier",
        "runtime_outlier",
        "adult",
        "video",
        "status",
        "original_language",
        "imdb_id",
        "genres_list",
        "production_companies_list",
        "production_countries_list",
        "spoken_languages_list",
        "is_collection",
        "overview",
    ]
    remain_cols = [c for c in df.columns if c not in column_order]
    df = df[[c for c in column_order if c in df.columns] + remain_cols]

    save_csv(df, "movies_metadata_cleaned.csv")

    df_box = df[(df["revenue"] > 0) & (df["budget"] > 0)].copy()
    print(f"  Box office subset (revenue>0 & budget>0): {len(df_box)}")
    save_csv(df_box, "movies_metadata_box_office.csv")

    print(f"\n  Final cleaned row count: {len(df)}  (original: {original_len})")
    return df


def main() -> None:
    clean_movies_metadata()


if __name__ == "__main__":
    main()