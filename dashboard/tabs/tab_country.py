"""
tabs/tab_country.py
Tab: 国家分析
来源逻辑: analysis/Data_A&V_country.py
"""

import ast

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from matplotlib.ticker import FuncFormatter

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

    # ── Core Insights ─────────────────────────────────────────
    st.subheader("Core Insights")
    st.info(
        "Based on the analysis of cleaned movie box office and metadata, the following "
        "key insights highlight the productivity and profitability of production countries."
    )
    col_a, col_b, col_c = st.columns(3)
    col_a.success(
        "Absolute Dominance 🇺🇸\n\n"
        "The United States leads with 4,379 films produced and $303.1 billion in "
        "total profit — entirely dwarfing every other country and confirming the "
        "unshakable central position of the Hollywood film industry."
    )
    col_b.warning(
        "Spillover Effect of Super IPs 🇬🇧\n\n"
        "The UK ranks second in total profit ($47.8 billion), largely due to the deep "
        "integration of major Hollywood studios and British production companies for global "
        "super IPs — a testament to the ROI power of transnational collaboration."
    )
    col_c.error(
        "High-ROI Anomalies 🇳🇿🇨🇳\n\n"
        "New Zealand tops average profit per film at 215.7M USD (driven by fantasy "
        "blockbusters like 'The Lord of the Rings'). China ranks second at 95.9M USD, "
        "reflecting the rapid expansion of its domestic box office."
    )

    st.divider()

    with st.spinner("加载数据中..."):
        df = load_box_office_full()

    df["countries_clean"] = df["production_countries_list"].apply(_clean_list)
    df_c = df.explode("countries_clean").dropna(subset=["countries_clean"])
    df_c = df_c[df_c["countries_clean"].astype(str).str.strip() != ""]

    # ── 统计 ──────────────────────────────────────────────────────────────
    top_countries_count = df_c["countries_clean"].value_counts().head(10)

    top_countries_profit = (
        df_c.groupby("countries_clean")["profit"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    country_stats = df_c.groupby("countries_clean").agg(
        movie_count=("id", "count"),
        total_profit=("profit", "sum"),
    )
    country_stats["avg_profit"] = country_stats["total_profit"] / country_stats["movie_count"]
    top_countries_avg_profit = (
        country_stats[country_stats["movie_count"] >= 20]
        .sort_values("avg_profit", ascending=False)
        .head(10)
    )

    # ── 配色 ──────────────────────────────────────────────────────────────
    color_count = sns.color_palette("ch:s=-.2,r=.6", 10)[::-1]
    color_profit = sns.color_palette("mako", 10)[::-1]
    color_avg = sns.color_palette("flare", 10)[::-1]

    def billion_fmt(x, pos):
        return f"${x * 1e-9:.1f}B"

    def million_fmt(x, pos):
        return f"${x * 1e-6:.0f}M"

    sns.set_theme(style="ticks", rc={"axes.facecolor": "#FFFFFF", "figure.facecolor": "#FFFFFF"})

    # ── 三连图 ────────────────────────────────────────────────────────────
    # ── 图1: 发片数量 ─────────────────────────────────────────────────────
    st.subheader("Top 10 Countries by Movie Count")
    fig1, ax1 = plt.subplots(figsize=(14, 7))
    sns.barplot(ax=ax1, x=top_countries_count.values, y=top_countries_count.index,
                palette=color_count, edgecolor="black", linewidth=0.5)
    ax1.set_title("Top 10 Countries by Movie Count", fontsize=18, fontweight="bold", pad=15)
    ax1.set_xlabel("Number of Movies", fontsize=14)
    ax1.set_ylabel("")
    ax1.tick_params(axis="both", labelsize=13)
    ax1.grid(axis="x", linestyle="--", alpha=0.6, color="#E0E0E0")
    ax1.set_axisbelow(True)
    for i, v in enumerate(top_countries_count.values):
        ax1.text(v + 10, i, f" {v}", va="center", fontsize=12, fontweight="bold", color="#333333")
    sns.despine(ax=ax1)
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    with st.expander("📊 图表解读：发片数量", expanded=True):
        st.markdown(
            "Movie Count (发片数量)\n\n"
            "This bar chart illustrates absolute production volume across the top 10 countries. "
            "The top bar for the United States is dramatically longer than the rest "
            "(nearly 7× that of the UK), visually highlighting its unmatched production capacity "
            "and the undisputed central position of Hollywood in the global film industry."
        )

    st.divider()

    # ── 图2: 总利润 ───────────────────────────────────────────────────────
    st.subheader("Top 10 Countries by Total Profit")
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    sns.barplot(ax=ax2, x=top_countries_profit.values, y=top_countries_profit.index,
                palette=color_profit, edgecolor="black", linewidth=0.5)
    ax2.set_title("Top 10 Countries by Total Profit", fontsize=18, fontweight="bold", pad=15)
    ax2.set_xlabel("Total Profit (USD)", fontsize=14)
    ax2.set_ylabel("")
    ax2.tick_params(axis="both", labelsize=13)
    ax2.xaxis.set_major_formatter(FuncFormatter(billion_fmt))
    ax2.grid(axis="x", linestyle="--", alpha=0.6, color="#E0E0E0")
    ax2.set_axisbelow(True)
    for i, v in enumerate(top_countries_profit.values):
        ax2.text(v + 1e9, i, f" ${v * 1e-9:.1f}B", va="center", fontsize=12, fontweight="bold", color="#333333")
    sns.despine(ax=ax2)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    with st.expander("Visuals Explanation", expanded=True):
        st.markdown(
            "This chart visualizes accumulated net profit across countries. "
            "Notice the \"visual compression\" effect: the US's 303.1B USD stretches "
            "to the right edge, making even the impressive 47.8B USD of the UK look "
            "relatively small — emphasizing the monopolistic scale of Hollywood.\n\n"
            "The UK's second-place position reflects the deep integration of major Hollywood "
            "studios with British production companies for global super IPs, demonstrating "
            "the exceptional ROI power of transnational collaboration."
        )

    st.divider()

    # ── 图3: 平均利润 ─────────────────────────────────────────────────────
    st.subheader("Top 10 Countries by Average Profit (Min. 20 Movies)")
    fig3, ax3 = plt.subplots(figsize=(14, 7))
    sns.barplot(ax=ax3, x=top_countries_avg_profit["avg_profit"].values, y=top_countries_avg_profit.index,
                palette=color_avg, edgecolor="black", linewidth=0.5)
    ax3.set_title("Top 10 Countries by Average Profit\n(Min. 20 Movies)", fontsize=18, fontweight="bold", pad=15)
    ax3.set_xlabel("Average Profit (USD)", fontsize=14)
    ax3.set_ylabel("")
    ax3.tick_params(axis="both", labelsize=13)
    ax3.xaxis.set_major_formatter(FuncFormatter(million_fmt))
    ax3.grid(axis="x", linestyle="--", alpha=0.6, color="#E0E0E0")
    ax3.set_axisbelow(True)
    for i, v in enumerate(top_countries_avg_profit["avg_profit"].values):
        ax3.text(v + 2e6, i, f" ${v * 1e-6:.1f}M", va="center", fontsize=12, fontweight="bold", color="#333333")
    sns.despine(ax=ax3)
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    with st.expander("📊 图表解读：平均利润 (ROI)", expanded=True):
        st.markdown(
            "Average Profit (平均利润, min. 20 films)\n\n"
            "The most crucial chart for understanding ROI efficiency. By filtering for "
            "countries with significant output (≥ 20 films), the focus shifts from sheer "
            "volume to per-film profitability.\n\n"
            "New Zealand tops the chart at 215.7M USD per film, driven by phenomenal "
            "fantasy blockbusters such as 'The Lord of the Rings'. "
            "China ranks second at 95.9M USD, demonstrating the rapid expansion and "
            "massive consumption power of its domestic box office. "
            "This chart proves that high-budget visual effects films and rapidly growing "
            "domestic markets play a decisive role in elevating a country's average cinematic profit."
        )

