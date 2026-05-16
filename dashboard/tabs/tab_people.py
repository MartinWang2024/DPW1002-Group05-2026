"""
tabs/tab_people.py
Tab: 人物分析（演员 / 导演）
来源逻辑: analysis/Data_A&V_peo.py
"""

import ast

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from matplotlib.ticker import FuncFormatter

from data_loader import load_box_office_full, load_credits_data


def _clean_list(s) -> list:
    try:
        lst = ast.literal_eval(s)
        return lst if isinstance(lst, list) and lst else []
    except Exception:
        return []


def render() -> None:
    st.header("人物分析 (Actors & Directors)")
    st.caption("数据来源: movies_metadata_box_office.csv + credits_cleaned.csv")

    # ── 核心发现 ──────────────────────────────────────────────────────────
    st.subheader("Core Insights")
    st.info(
        "We shifted our analytical perspective down to specific cast and crew members to explore the relationship between personal branding, productivity, and box office appeal."
    )
    col_a, col_b, col_c = st.columns(3)
    col_a.success(
        "**Box Office Money Printers**\n\n"
        "Commercial monetization reveals two distinct paths to wealth. Emma Watson tops the list with a staggering 7.8 billion USD in accumulated profit, driven purely by long-running global super fantasy IPs (e.g., Harry Potter). However, she is immediately followed by veteran superstars Tom Cruise (6.8 billion USD) and Tom Hanks (6.7 billion USD)."
    )
    col_b.warning(
        "**The \"Dual-Threat\" Superstars vs. Pure Hardworkers**\n\n"
        "While some of the most industrious actors (e.g., Robert De Niro with 49 films, Nicolas Cage) do not appear on the most profitable list, there is a rare tier of \"Dual-Threat\" superstars. Tom Cruise, Tom Hanks, and Johnny Depp successfully secure spots in the Top 10 for both movie count and total profit. This proves that while sheer volume alone doesn't guarantee massive wealth, combining decades of sustained high output with strategic blockbuster choices is a proven formula for absolute commercial dominance."
    )
    col_c.error(
        "**The Ultimate Dominance**\n\n"
        "Steven Spielberg exhibits unbelievable industry dominance. He is the only \"Double Crown\" winner in the directorial category: ranking first in output (30 films) and dominating the total profit chart (7.5 billion USD)."
    )
    st.divider()

    with st.spinner("加载数据中..."):
        df_box = load_box_office_full()
        df_credits = load_credits_data()

    df = df_box.merge(df_credits, on="id", how="inner")
    df["profit"] = df["revenue"] - df["budget"]
    df["cast_list_clean"] = df["cast_list"].apply(_clean_list)
    df["directors_clean"] = df["directors"].apply(_clean_list)

    df_cast = df.explode("cast_list_clean").dropna(subset=["cast_list_clean"])
    df_cast = df_cast[df_cast["cast_list_clean"].astype(str).str.strip() != ""]
    df_dir = df.explode("directors_clean").dropna(subset=["directors_clean"])
    df_dir = df_dir[df_dir["directors_clean"].astype(str).str.strip() != ""]

    top_actors_count = df_cast["cast_list_clean"].value_counts().head(10)
    top_actors_profit = (
        df_cast.groupby("cast_list_clean")["profit"].sum()
        .sort_values(ascending=False).head(10)
    )
    top_dir_count = df_dir["directors_clean"].value_counts().head(10)
    top_dir_profit = (
        df_dir.groupby("directors_clean")["profit"].sum()
        .sort_values(ascending=False).head(10)
    )

    # ── 演员分析 ──────────────────────────────────────────────────────────────
    st.subheader("演员分析 (Actors)")
    fig_actors, axes_actors = plt.subplots(1, 2, figsize=(20, 7))

    color_count = sns.color_palette("ch:s=.25,rot=-.25", 10)[::-1]
    color_profit = sns.color_palette("crest", 10)[::-1]

    def billion_fmt(x, _):
        return f"${x * 1e-9:.1f}B"

    # 图1 演员出演数量
    sns.barplot(ax=axes_actors[0], x=top_actors_count.values, y=top_actors_count.index,
                palette=color_count, edgecolor="black", linewidth=0.5)
    axes_actors[0].set_title("Top 10 Actors by Movie Count", fontsize=16, fontweight="bold", pad=12)
    axes_actors[0].set_xlabel("Number of Movies", fontsize=13)
    axes_actors[0].set_ylabel("")
    for i, v in enumerate(top_actors_count.values):
        axes_actors[0].text(v + 0.3, i, f" {v}", va="center", fontsize=12, fontweight="bold", color="#333333")

    # 图2 演员累计利润
    sns.barplot(ax=axes_actors[1], x=top_actors_profit.values, y=top_actors_profit.index,
                palette=color_profit, edgecolor="black", linewidth=0.5)
    axes_actors[1].set_title("Top 10 Actors by Total Profit", fontsize=16, fontweight="bold", pad=12)
    axes_actors[1].set_xlabel("Total Profit (USD)", fontsize=13)
    axes_actors[1].set_ylabel("")
    axes_actors[1].xaxis.set_major_formatter(FuncFormatter(billion_fmt))
    for i, v in enumerate(top_actors_profit.values):
        axes_actors[1].text(v + 1e8, i, f" ${v * 1e-9:.1f}B", va="center", fontsize=12, fontweight="bold", color="#333333")

    for ax in axes_actors.flat:
        ax.grid(axis="x", linestyle="--", alpha=0.6, color="#E0E0E0")
        ax.set_axisbelow(True)

    sns.despine(fig_actors)
    plt.tight_layout(pad=3.0)
    st.pyplot(fig_actors)
    plt.close(fig_actors)

    with st.expander("📊 Visuals Explanation", expanded=True):
        st.markdown(
            "**Top Row (Actors Analysis):** By cross-referencing the left chart (most prolific) and the right chart (highest-grossing), we can visually categorize actors into three distinct success models: the \"Franchise Winners\" (e.g., Emma Watson, Daniel Radcliffe on the right), the \"Pure Hardworkers\" (e.g., Robert De Niro, Bruce Willis exclusively on the left), and the \"Dual-Threat Legends\" (Tom Cruise, Tom Hanks, Johnny Depp, who impressively bridge the gap and appear on both lists)."
        )

    # ── 导演分析 ──────────────────────────────────────────────────────────────
    st.subheader("导演分析 (Directors)")
    fig_directors, axes_directors = plt.subplots(1, 2, figsize=(20, 7))

    # 图3 导演执导数量
    sns.barplot(ax=axes_directors[0], x=top_dir_count.values, y=top_dir_count.index,
                palette=color_count, edgecolor="black", linewidth=0.5)
    axes_directors[0].set_title("Top 10 Directors by Movie Count", fontsize=16, fontweight="bold", pad=12)
    axes_directors[0].set_xlabel("Number of Movies", fontsize=13)
    axes_directors[0].set_ylabel("")
    for i, v in enumerate(top_dir_count.values):
        axes_directors[0].text(v + 0.2, i, f" {v}", va="center", fontsize=12, fontweight="bold", color="#333333")

    # 图4 导演累计利润
    sns.barplot(ax=axes_directors[1], x=top_dir_profit.values, y=top_dir_profit.index,
                palette=color_profit, edgecolor="black", linewidth=0.5)
    axes_directors[1].set_title("Top 10 Directors by Total Profit", fontsize=16, fontweight="bold", pad=12)
    axes_directors[1].set_xlabel("Total Profit (USD)", fontsize=13)
    axes_directors[1].set_ylabel("")
    axes_directors[1].xaxis.set_major_formatter(FuncFormatter(billion_fmt))
    for i, v in enumerate(top_dir_profit.values):
        axes_directors[1].text(v + 1e8, i, f" ${v * 1e-9:.1f}B", va="center", fontsize=12, fontweight="bold", color="#333333")

    for ax in axes_directors.flat:
        ax.grid(axis="x", linestyle="--", alpha=0.6, color="#E0E0E0")
        ax.set_axisbelow(True)

    sns.despine(fig_directors)
    plt.tight_layout(pad=3.0)
    st.pyplot(fig_directors)
    plt.close(fig_directors)

    # ── 图表解读 ────────────────────────────────────────────────────────────
    with st.expander("📊 Visuals Explanation", expanded=True):
        st.markdown(
            "**Bottom Row (Directors Analysis):** The presence of Steven Spielberg at the very top of both charts visually cements his status as the undisputed king of commercial cinema, while the high rankings of Peter Jackson and James Cameron on the right chart emphasize the immense financial impact of top-tier blockbuster directors."
        )
