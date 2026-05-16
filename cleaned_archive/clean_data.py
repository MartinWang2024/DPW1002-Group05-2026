"""
Data Cleaning Script - Movie Box Office Factor Analysis Project
DPW1002 Group 05 · 2026

Data source: MovieLens / TMDB datasets under archive/
Output directory: cleaned_archive/

清理步骤:
  1. movies_metadata.csv  —— 主表，去重、类型转换、异常值处理、JSON 解析
  2. ratings_small.csv    —— 用户评分，校验评分范围与重复值
  3. keywords.csv         —— 关键词，JSON 解析展开
  4. credits.csv          —— 演职人员，JSON 解析展开
  5. links_small.csv      —— ID 映射，去重、关联校验
"""

import ast
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────

def load_csv(filename: str, **kwargs) -> pd.DataFrame:
    """Read a CSV from archive/ and print basic information."""
    path = f"archive/{filename}"
    df = pd.read_csv(path, **kwargs)
    print(f"\n{'='*60}")
    print(f"[Load] {filename}  shape={df.shape}")
    return df


def save_csv(df: pd.DataFrame, filename: str) -> None:
    """Save to cleaned_archive/."""
    path = f"cleaned_archive/{filename}"
    df.to_csv(path, index=False)
    print(f"[Save] {filename}  shape={df.shape}  -> {path}")


def iqr_bounds(series: pd.Series, k: float = 1.5):
    """Return IQR-based (lower, upper) bounds computed on positive values only."""
    pos = series[series > 0].dropna()
    q1, q3 = pos.quantile(0.25), pos.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def safe_parse_list(val) -> list:
    """Parse stringified JSON / Python-literal list into Python list; return [] on failure."""
    if pd.isna(val) or val == "":
        return []
    try:
        result = ast.literal_eval(val)
        return result if isinstance(result, list) else []
    except Exception:
        return []


def extract_names(val, key: str = "name") -> list:
    """Extract a list of values for the given key from a JSON list of dicts."""
    items = safe_parse_list(val)
    return [item.get(key, "") for item in items if isinstance(item, dict) and item.get(key)]


# ─────────────────────────────────────────────
# 1. movies_metadata.csv
# ─────────────────────────────────────────────

def clean_movies_metadata() -> pd.DataFrame:
    df = load_csv("movies_metadata.csv", low_memory=False)
    original_len = len(df)

    # ── 1-1. 删除行偏移造成的错误行（id 为日期字符串）──────────────
    df = df[df["id"].astype(str).str.match(r"^\d+$")].copy()
    df["id"] = df["id"].astype(int)
    print(f"  Removed malformed rows (non-integer id): {original_len - len(df)}")

    # 1-2. Deduplication
    before = len(df)
    df = df.drop_duplicates(subset=["id"])
    df = df.drop_duplicates(subset=["imdb_id"])
    print(f"  Deduplication (id/imdb_id): removed {before - len(df)}")

    # ── 1-3. adult 列：过滤非布尔值行，转换为 bool ────────────────
    before = len(df)
    df = df[df["adult"].isin(["True", "False"])].copy()
    df["adult"] = df["adult"] == "True"
    print(f"  adult column cleaning: removed {before - len(df)} non-boolean rows")

    # ── 1-4. budget 转数值 ────────────────────────────────────────
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce").fillna(0).astype(np.int64)

    # ── 1-5. revenue 填充缺失 ─────────────────────────────────────
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0).astype(np.int64)

    # ── 1-6. release_date 转 datetime，提取年份 ───────────────────
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year.astype("Int64")

    # 删除缺失 release_date 的行（无法判断年代）
    before = len(df)
    df = df.dropna(subset=["release_date"])
    print(f"  Removed missing release_date rows: {before - len(df)}")

    # ── 1-7. runtime 异常值处理（IQR 法） ────────────────────────
    # 先删去 runtime=0 的行（无效值）
    before = len(df)
    df = df[df["runtime"].fillna(0) > 0].copy()
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
    print(f"  Removed runtime=0 rows: {before - len(df)}")

    lo, hi = iqr_bounds(df["runtime"], k=1.5)
    # runtime 下界不低于 20 分钟（短片/广告片排除）
    lo = max(lo, 20.0)
    outlier_mask = (df["runtime"] < lo) | (df["runtime"] > hi)
    df["runtime_outlier"] = outlier_mask
    print(f"  runtime outliers (IQR k=1.5, range [{lo:.0f}, {hi:.0f}]): {outlier_mask.sum()} flagged")

    # ── 1-8. budget / revenue 异常值标记（仅对有效正值集合） ──────
    for col in ["budget", "revenue"]:
        lo_c, hi_c = iqr_bounds(df[col], k=3.0)  # Use a looser k=3 to avoid over-filtering
        lo_c = max(lo_c, 0)
        mask = (df[col] > 0) & ((df[col] < lo_c) | (df[col] > hi_c))
        df[f"{col}_outlier"] = mask
        print(f"  {col} outliers (IQR k=3.0, range [{lo_c:.0f}, {hi_c:.0f}]): {mask.sum()} flagged")

    # ── 1-9. popularity 缺失填充中位数 ──────────────────────────
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
    median_pop = df["popularity"].median()
    df["popularity"] = df["popularity"].fillna(median_pop)

    # ── 1-10. vote_average / vote_count 缺失处理 ─────────────────
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce")
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0).astype(np.int64)
    # 删除 vote_average 为 0 且 vote_count 为 0 的行（无评分意义）
    before = len(df)
    df = df[~((df["vote_average"] == 0) & (df["vote_count"] == 0))].copy()
    print(f"  Removed vote=0 & count=0 rows: {before - len(df)}")

    # ── 1-11. JSON 列解析：genres / production_companies / production_countries ──
    df["genres_list"] = df["genres"].apply(lambda x: extract_names(x, "name"))

    df["production_companies_list"] = df["production_companies"].apply(lambda x: extract_names(x, "name"))

    df["production_countries_list"] = df["production_countries"].apply(lambda x: extract_names(x, "name"))

    df["spoken_languages_list"] = df["spoken_languages"].apply(lambda x: extract_names(x, "name"))

    # 提取 belongs_to_collection 布尔标志（是否属于系列电影）
    df["is_collection"] = df["belongs_to_collection"].apply(
        lambda x: False if pd.isna(x) or x == "" else True
    )

    # ── 1-12. 保留核心分析列，删除冗余原始列 ─────────────────────
    drop_cols = [
        "genres", "production_companies", "production_countries",
        "spoken_languages", "belongs_to_collection",
        "homepage", "poster_path", "tagline",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 按票房分析优先级重排列：id 和 title 最前，关键数值指标靠前
    column_order = [
        "id", "title", "original_title",
        "budget", "revenue", "runtime", "release_date", "release_year",
        "popularity", "vote_average", "vote_count",
        "budget_outlier", "revenue_outlier", "runtime_outlier",
        "adult", "video", "status", "original_language", "imdb_id",
        "genres_list", "production_companies_list", "production_countries_list", "spoken_languages_list",
        "is_collection", "overview",
    ]
    remain_cols = [c for c in df.columns if c not in column_order]
    df = df[[c for c in column_order if c in df.columns] + remain_cols]

    # ── 1-13. 输出为"全量清理版"和"票房分析版"（revenue > 0 且 budget > 0） ────
    save_csv(df, "movies_metadata_cleaned.csv")

    df_box = df[(df["revenue"] > 0) & (df["budget"] > 0)].copy()
    print(f"  Box office subset (revenue>0 & budget>0): {len(df_box)}")
    save_csv(df_box, "movies_metadata_box_office.csv")

    print(f"\n  Final cleaned row count: {len(df)}  (original: {original_len})")
    return df


# ─────────────────────────────────────────────
# 2. ratings_small.csv
# ─────────────────────────────────────────────

def clean_ratings_small() -> pd.DataFrame:
    df = load_csv("ratings_small.csv")
    original_len = len(df)

    # ── 2-1. 去重 ─────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["userId", "movieId"])
    print(f"  Deduplication (userId+movieId): removed {before - len(df)}")

    # ── 2-2. 校验评分范围 (MovieLens 标准: 0.5 ~ 5.0，步长 0.5) ──
    valid_ratings = np.arange(0.5, 5.5, 0.5)
    before = len(df)
    df = df[df["rating"].isin(valid_ratings)].copy()
    print(f"  Invalid ratings: removed {before - len(df)}")

    # ── 2-3. timestamp 转 datetime ────────────────────────────────
    df["rating_datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    # ── 2-4. 缺失值检查 ───────────────────────────────────────────
    missing = df.isnull().sum().sum()
    print(f"  Missing values: {missing}")

    print(f"\n  Final row count: {len(df)}  (original: {original_len})")
    save_csv(df, "ratings_small_cleaned.csv")
    return df


# ─────────────────────────────────────────────
# 3. keywords.csv
# ─────────────────────────────────────────────

def clean_keywords() -> pd.DataFrame:
    df = load_csv("keywords.csv")
    original_len = len(df)

    # ── 3-1. id 转整数，去重 ──────────────────────────────────────
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype(int)
    df = df.drop_duplicates(subset=["id"])
    print(f"  ID cleaning/deduplication: removed {before - len(df)}")

    # ── 3-2. 解析 keywords JSON 并展开为长表 ──────────────────────
    # 输出列: id(电影id), keyword_id, keyword_name
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

    # id 放第一列，后面接展开后的 JSON 字段
    df = df[["id", "keyword_id", "keyword_name"]]

    print(f"\n  Final row count: {len(df)}  (original: {original_len})")
    save_csv(df, "keywords_cleaned.csv")
    return df


# ─────────────────────────────────────────────
# 4. credits.csv
# ─────────────────────────────────────────────

def clean_credits() -> pd.DataFrame:
    df = load_csv("credits.csv")
    original_len = len(df)

    # ── 4-1. id 转整数，去重 ──────────────────────────────────────
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype(int)
    df = df.drop_duplicates(subset=["id"])
    print(f"  ID cleaning/deduplication: removed {before - len(df)}")

    # ── 4-2. 解析 cast：提取前3位主演姓名 ────────────────────────
    def top_cast(val, n=3):
        items = safe_parse_list(val)
        return [item["name"] for item in items[:n] if isinstance(item, dict) and "name" in item]

    df["cast_list"] = df["cast"].apply(top_cast)
    df["cast_count"] = df["cast"].apply(lambda v: len(safe_parse_list(v)))

    # ── 4-3. 解析 crew：提取导演、编剧 ──────────────────────────
    def extract_role(val, job_filter):
        items = safe_parse_list(val)
        return [
            item["name"]
            for item in items
            if isinstance(item, dict) and item.get("job") in job_filter
        ]

    df["directors"] = df["crew"].apply(lambda v: extract_role(v, {"Director"}))

    df["writers"] = df["crew"].apply(
        lambda v: extract_role(v, {"Screenplay", "Writer", "Story"})
    )

    df = df.drop(columns=["cast", "crew"])

    print(f"\n  Final row count: {len(df)}  (original: {original_len})")
    save_csv(df, "credits_cleaned.csv")
    return df


# ─────────────────────────────────────────────
# 5. links_small.csv
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Movie Box Office Factor Analysis - Data Cleaning")
    print("  Using pandas", pd.__version__, "/ numpy", np.__version__)
    print("=" * 60)

    movies = clean_movies_metadata()
    ratings = clean_ratings_small()
    keywords = clean_keywords()
    credits = clean_credits()
    links = clean_links_small()

    # Summary
    print("\n" + "=" * 60)
    print("  Cleaning completed. Output files:")
    print("=" * 60)
    print("  movies_metadata_cleaned.csv")
    print("  movies_metadata_box_office.csv")
    print("  ratings_small_cleaned.csv")
    print("  keywords_cleaned.csv")
    print("  credits_cleaned.csv")
    print("  links_small_cleaned.csv")

    print("\n  Key statistics:")
    print(f"  movies_metadata after full cleaning: {len(movies):,}")
    print(f"  movies_metadata box office subset:   {((movies['revenue'] > 0) & (movies['budget'] > 0)).sum():,}")
    print(f"  ratings_small after cleaning:        {len(ratings):,}")
    print(f"  keywords after cleaning:             {len(keywords):,}")
    print(f"  credits after cleaning:              {len(credits):,}")
    print(f"  links_small after cleaning:          {len(links):,}")

    print("\n  runtime / budget / revenue outlier flag columns are kept in")
    print("  movies_metadata_cleaned.csv for optional filtering in later analysis.")


if __name__ == "__main__":
    main()
