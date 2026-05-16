"""
tabs/tab_country.py
Tab: 国家分析
来源逻辑: analysis/Data_A&V_country.py
"""

import ast

import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import load_box_office_full


def _clean_list(s) -> list:
    try:
        lst = ast.literal_eval(s)
        return lst if isinstance(lst, list) and lst else []
    except Exception:
        return []


def render() -> None:
    st.header("国家分析 (Country Analysis)")
    st.caption("数据来源: movies_metadata_box_office.csv")

    with st.spinner("加载数据中..."):
        df = load_box_office_full()

    df["countries_clean"] = df["production_countries_list"].apply(_clean_list)
    df_c = df.explode("countries_clean").dropna(subset=["countries_clean"])
    df_c = df_c[df_c["countries_clean"].astype(str).str.strip() != ""]

    # ── 统计 ──────────────────────────────────────────────────────────────
    top_count = (
        df_c["countries_clean"].value_counts().head(10).reset_index()
        .rename(columns={"index": "country", "countries_clean": "movie_count",
                         "count": "movie_count"})
    )
    # pandas value_counts() returns Series; convert properly
    vc = df_c["countries_clean"].value_counts().head(10)
    top_count = pd.DataFrame({"country": vc.index, "movie_count": vc.values})

    top_profit = (
        df_c.groupby("countries_clean")["profit"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={"countries_clean": "country", "profit": "total_profit"})
    )
    top_profit["total_profit_B"] = (top_profit["total_profit"] / 1e9).round(2)

    country_stats = df_c.groupby("countries_clean").agg(
        movie_count=("id", "count"),
        total_profit=("profit", "sum"),
    ).reset_index().rename(columns={"countries_clean": "country"})
    country_stats["avg_profit_M"] = (
        country_stats["total_profit"] / country_stats["movie_count"] / 1e6
    ).round(1)
    top_avg = (
        country_stats[country_stats["movie_count"] >= 20]
        .sort_values("avg_profit_M", ascending=False)
        .head(10)
    )

    # ── 图1: 发片数量（交互式） ───────────────────────────────────────────
    st.subheader("Top 10 国家 — 发片数量")
    fig1 = px.bar(
        top_count.sort_values("movie_count"),
        x="movie_count", y="country",
        orientation="h",
        text="movie_count",
        color="movie_count",
        color_continuous_scale="Blues",
        labels={"movie_count": "电影数量", "country": "国家"},
    )
    fig1.update_traces(textposition="outside")
    fig1.update_layout(coloraxis_showscale=False, height=420)
    st.plotly_chart(fig1, use_container_width=True)

    # ── 图2: 总利润（交互式） ─────────────────────────────────────────────
    st.subheader("Top 10 国家 — 总利润")
    fig2 = px.bar(
        top_profit.sort_values("total_profit_B"),
        x="total_profit_B", y="country",
        orientation="h",
        text="total_profit_B",
        color="total_profit_B",
        color_continuous_scale="Teal",
        labels={"total_profit_B": "总利润 (十亿$)", "country": "国家"},
    )
    fig2.update_traces(texttemplate="%{text:.1f}B", textposition="outside")
    fig2.update_layout(coloraxis_showscale=False, height=420)
    st.plotly_chart(fig2, use_container_width=True)

    # ── 图3: 平均利润（交互式） ───────────────────────────────────────────
    st.subheader("Top 10 国家 — 平均利润（≥20 部）")
    fig3 = px.bar(
        top_avg.sort_values("avg_profit_M"),
        x="avg_profit_M", y="country",
        orientation="h",
        text="avg_profit_M",
        color="avg_profit_M",
        color_continuous_scale="Oranges",
        labels={"avg_profit_M": "平均利润 (百万$)", "country": "国家"},
    )
    fig3.update_traces(texttemplate="%{text:.0f}M", textposition="outside")
    fig3.update_layout(coloraxis_showscale=False, height=420)
    st.plotly_chart(fig3, use_container_width=True)

    # ── 动图: 年度发片量 Bar Chart Race ───────────────────────────────────
    st.subheader("动态图 — 各国年度发片量变化")
    st.caption("点击播放按钮查看各年份各国电影产量变化")

    top_countries = top_count["country"].tolist()
    df_year = df_c[df_c["countries_clean"].isin(top_countries)].copy()
    df_year["release_year"] = pd.to_numeric(df_year["release_year"], errors="coerce")
    df_year = df_year.dropna(subset=["release_year"])
    df_year["release_year"] = df_year["release_year"].astype(int)

    yearly = (
        df_year.groupby(["release_year", "countries_clean"])
        .size()
        .reset_index(name="movie_count")
        .rename(columns={"countries_clean": "country"})
    )
    # 补全缺失年份/国家组合为 0，使动画连续
    all_years = range(int(yearly["release_year"].min()), int(yearly["release_year"].max()) + 1)
    full_idx = pd.MultiIndex.from_product(
        [all_years, top_countries], names=["release_year", "country"]
    )
    yearly = (
        yearly.set_index(["release_year", "country"])
        .reindex(full_idx, fill_value=0)
        .reset_index()
    )

    fig_anim = px.bar(
        yearly.sort_values(["release_year", "movie_count"]),
        x="movie_count",
        y="country",
        orientation="h",
        animation_frame="release_year",
        animation_group="country",
        color="country",
        range_x=[0, yearly["movie_count"].max() + 3],
        labels={"movie_count": "电影数量", "country": "国家", "release_year": "年份"},
        title="Top 10 国家年度发片量",
    )
    fig_anim.update_layout(
        height=480,
        showlegend=False,
        sliders=[{"currentvalue": {"prefix": "年份: "}}],
    )
    st.plotly_chart(fig_anim, use_container_width=True)

