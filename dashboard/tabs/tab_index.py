"""
tabs/tab_index.py
Tab: 数据概览
整合「预算与票房」「评分关系」「国家分布」三个分析板块，用内部子标题分隔。
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import build_country_stats


def render(movies_df: pd.DataFrame, rating_df: pd.DataFrame) -> None:

    # ── 筛选控件（仅本 Tab 生效） ─────────────────────────────────────────
    with st.expander("筛选条件", expanded=True):
        year_min = int(movies_df["release_year"].min())
        year_max = int(movies_df["release_year"].max())
        col_y, col_b = st.columns(2)
        with col_y:
            selected_years = st.slider(
                "上映年份范围", min_value=year_min, max_value=year_max, value=(1980, year_max),
                key="idx_years",
            )
        with col_b:
            budget_cap = st.slider(
                "预算上限 (M$)", min_value=10, max_value=500, value=120, step=10,
                key="idx_budget",
            )

    filtered = movies_df[
        (movies_df["release_year"] >= selected_years[0])
        & (movies_df["release_year"] <= selected_years[1])
        & (movies_df["budget_M"] <= budget_cap)
    ].copy()

    # ── 指标卡 ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("电影数量", f"{len(filtered):,}")
    c2.metric("预算中位数", f"${filtered['budget_M'].median():.1f}M")
    c3.metric("票房中位数", f"${filtered['revenue_M'].median():.1f}M")
    c4.metric("利润中位数", f"${filtered['profit_M'].median():.1f}M")

    st.divider()

    # ── 预算与票房 ────────────────────────────────────────────────────────
    st.subheader("预算与票房")

    left, right = st.columns(2)
    with left:
        fig_hist = px.histogram(
            filtered,
            x="budget_M",
            nbins=40,
            title="预算分布 (Budget Distribution)",
            labels={"budget_M": "Budget (M$)"},
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with right:
        trend = (
            filtered.groupby("release_year")["budget_M"]
            .median()
            .reset_index()
            .rename(columns={"budget_M": "median_budget_M"})
        )
        fig_trend = px.line(
            trend,
            x="release_year",
            y="median_budget_M",
            markers=True,
            title="年度预算中位数趋势",
            labels={"release_year": "Year", "median_budget_M": "Median Budget (M$)"},
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    corr = filtered["budget_M"].corr(filtered["revenue_M"])
    fig_scatter = px.scatter(
        filtered,
        x="budget_M",
        y="revenue_M",
        title=f"预算 vs 票房 (相关系数: {corr:.2f})",
        labels={"budget_M": "Budget (M$)", "revenue_M": "Revenue (M$)"},
        opacity=0.5,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    # ── 评分关系 ──────────────────────────────────────────────────────────
    st.subheader("评分关系")

    rating_df = rating_df[
        (rating_df["avg_rating"] >= 0) & (rating_df["avg_rating"] <= 5)
    ].copy()

    fig_rating = px.scatter(
        rating_df,
        x="avg_rating",
        y="revenue_M",
        color="rating_count",
        title="评分与票房关系",
        labels={
            "avg_rating": "Average Rating",
            "revenue_M": "Revenue (M$)",
            "rating_count": "Rating Count",
        },
        opacity=0.45,
    )
    st.plotly_chart(fig_rating, use_container_width=True)

    pearson = rating_df["avg_rating"].corr(rating_df["revenue_M"], method="pearson")
    spearman = rating_df["avg_rating"].corr(rating_df["revenue_M"], method="spearman")
    st.info(f"Pearson: {pearson:.4f} | Spearman: {spearman:.4f}")

    st.divider()

    # ── 国家分布 ──────────────────────────────────────────────────────────
    st.subheader("国家分布")

    countries = build_country_stats(filtered)
    fig_country = px.bar(
        countries,
        x="movie_count",
        y="country",
        orientation="h",
        title="Top Countries by Movie Count",
        labels={"movie_count": "Movie Count", "country": "Country"},
    )
    fig_country.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_country, use_container_width=True)
    st.dataframe(countries, use_container_width=True)
