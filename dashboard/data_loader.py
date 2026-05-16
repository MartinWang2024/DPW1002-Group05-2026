"""
data_loader.py
公共数据加载函数与路径常量，供所有 tab 模块共享使用。
"""

from pathlib import Path
import ast
import base64
import json

import pandas as pd
import streamlit as st

# ── 路径常量 ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
CLEANED_DIR = ROOT_DIR / "cleaned_archive"
ANALYSIS_DIR = ROOT_DIR / "analysis"
MEDIA_FILES = [
    ROOT_DIR / "academic_country_3_charts.png",
    ROOT_DIR / "regression_results_fixed.png",
]


# ── 数据加载 ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_movies_data() -> pd.DataFrame:
    path = CLEANED_DIR / "movies_metadata_box_office.csv"
    df = pd.read_csv(path, low_memory=False)

    for col in ["budget", "revenue", "popularity", "vote_average", "vote_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["release_date"] = pd.to_datetime(df.get("release_date"), errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    df = df.dropna(subset=["release_year", "budget", "revenue"]).copy()
    df = df[(df["budget"] > 0) & (df["revenue"] > 0)]
    df["budget_M"] = df["budget"] / 1_000_000
    df["revenue_M"] = df["revenue"] / 1_000_000
    df["profit_M"] = (df["revenue"] - df["budget"]) / 1_000_000
    return df


@st.cache_data(show_spinner=False)
def load_rating_data() -> pd.DataFrame:
    movies = pd.read_csv(CLEANED_DIR / "movies_metadata_cleaned.csv", low_memory=False)
    ratings = pd.read_csv(CLEANED_DIR / "ratings_small_cleaned.csv")

    movies["id"] = pd.to_numeric(movies.get("id"), errors="coerce")
    movies = movies.dropna(subset=["id"]).copy()
    movies["id"] = movies["id"].astype(int)

    ratings_agg = (
        ratings.groupby("movieId")["rating"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_rating", "count": "rating_count"})
    )

    merged = pd.merge(movies, ratings_agg, left_on="id", right_on="movieId", how="inner")
    merged["revenue"] = pd.to_numeric(merged.get("revenue"), errors="coerce")
    merged = merged[merged["revenue"] > 0].copy()
    merged["revenue_M"] = merged["revenue"] / 1_000_000
    return merged


# ── 辅助函数 ──────────────────────────────────────────────────────────────

def parse_countries(cell: str) -> list[str]:
    try:
        value = ast.literal_eval(cell)
        if isinstance(value, list):
            return [str(item) for item in value if item]
        return []
    except (ValueError, SyntaxError, TypeError):
        return []


def build_country_stats(df: pd.DataFrame) -> pd.DataFrame:
    if "production_countries_list" not in df.columns:
        return pd.DataFrame(columns=["country", "movie_count"])

    tmp = df[["id", "production_countries_list"]].copy()
    tmp["country"] = tmp["production_countries_list"].apply(parse_countries)
    tmp = tmp.explode("country").dropna(subset=["country"])

    return (
        tmp.groupby("country")["id"]
        .count()
        .reset_index()
        .rename(columns={"id": "movie_count"})
        .sort_values("movie_count", ascending=False)
        .head(15)
    )


@st.cache_data(show_spinner=False)
def load_box_office_full() -> pd.DataFrame:
    """加载完整票房数据（不做 revenue/budget 强过滤，供专项分析使用）。"""
    path = CLEANED_DIR / "movies_metadata_box_office.csv"
    df = pd.read_csv(path, low_memory=False)
    for col in ["budget", "revenue", "popularity", "vote_average", "vote_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["release_date"] = pd.to_datetime(df.get("release_date"), errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    df["profit"] = df["revenue"] - df["budget"]
    return df


@st.cache_data(show_spinner=False)
def load_credits_data() -> pd.DataFrame:
    return pd.read_csv(CLEANED_DIR / "credits_cleaned.csv")


@st.cache_data(show_spinner=False)
def load_keywords_data() -> pd.DataFrame:
    return pd.read_csv(CLEANED_DIR / "keywords_cleaned.csv")


def discover_notebooks() -> list[Path]:
    return sorted(ANALYSIS_DIR.glob("*.ipynb"))


def extract_notebook_sections(notebook_path: Path) -> tuple[list[str], list[bytes]]:
    with notebook_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    headings: list[str] = []
    images: list[bytes] = []

    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            source = cell.get("source", [])
            text = "".join(source) if isinstance(source, list) else str(source)
            for line in text.splitlines():
                if line.strip().startswith("#"):
                    headings.append(line.strip().lstrip("#").strip())

        if cell.get("cell_type") == "code":
            for out in cell.get("outputs", []):
                png_data = out.get("data", {}).get("image/png")
                if png_data:
                    if isinstance(png_data, list):
                        png_data = "".join(png_data)
                    try:
                        images.append(base64.b64decode(png_data))
                    except (ValueError, TypeError):
                        continue

    headings = list(dict.fromkeys(h for h in headings if h))
    return headings, images
