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

    # ── 绘图 ──────────────────────────────────────────────────────────────
    plt.style.use("default")
    sns.set_theme(
        style="ticks",
        rc={"axes.facecolor": "#FFFFFF", "figure.facecolor": "#FFFFFF"},
    )
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    color_count = sns.color_palette("ch:s=.25,rot=-.25", 10)[::-1]
    color_profit = sns.color_palette("crest", 10)[::-1]

    def billion_fmt(x, _):
        return f"${x * 1e-9:.1f}B"

    # 图1 演员出演数量
    sns.barplot(ax=axes[0, 0], x=top_actors_count.values, y=top_actors_count.index,
                palette=color_count, edgecolor="black", linewidth=0.5)
    axes[0, 0].set_title("Top 10 Actors by Movie Count", fontsize=16, fontweight="bold", pad=12)
    axes[0, 0].set_xlabel("Number of Movies", fontsize=13)
    axes[0, 0].set_ylabel("")
    for i, v in enumerate(top_actors_count.values):
        axes[0, 0].text(v + 0.3, i, f" {v}", va="center", fontsize=12, fontweight="bold", color="#333333")

    # 图2 演员累计利润
    sns.barplot(ax=axes[0, 1], x=top_actors_profit.values, y=top_actors_profit.index,
                palette=color_profit, edgecolor="black", linewidth=0.5)
    axes[0, 1].set_title("Top 10 Actors by Total Profit", fontsize=16, fontweight="bold", pad=12)
    axes[0, 1].set_xlabel("Total Profit (USD)", fontsize=13)
    axes[0, 1].set_ylabel("")
    axes[0, 1].xaxis.set_major_formatter(FuncFormatter(billion_fmt))
    for i, v in enumerate(top_actors_profit.values):
        axes[0, 1].text(v + 1e8, i, f" ${v * 1e-9:.1f}B", va="center", fontsize=12, fontweight="bold", color="#333333")

    # 图3 导演执导数量
    sns.barplot(ax=axes[1, 0], x=top_dir_count.values, y=top_dir_count.index,
                palette=color_count, edgecolor="black", linewidth=0.5)
    axes[1, 0].set_title("Top 10 Directors by Movie Count", fontsize=16, fontweight="bold", pad=12)
    axes[1, 0].set_xlabel("Number of Movies", fontsize=13)
    axes[1, 0].set_ylabel("")
    for i, v in enumerate(top_dir_count.values):
        axes[1, 0].text(v + 0.2, i, f" {v}", va="center", fontsize=12, fontweight="bold", color="#333333")

    # 图4 导演累计利润
    sns.barplot(ax=axes[1, 1], x=top_dir_profit.values, y=top_dir_profit.index,
                palette=color_profit, edgecolor="black", linewidth=0.5)
    axes[1, 1].set_title("Top 10 Directors by Total Profit", fontsize=16, fontweight="bold", pad=12)
    axes[1, 1].set_xlabel("Total Profit (USD)", fontsize=13)
    axes[1, 1].set_ylabel("")
    axes[1, 1].xaxis.set_major_formatter(FuncFormatter(billion_fmt))
    for i, v in enumerate(top_dir_profit.values):
        axes[1, 1].text(v + 1e8, i, f" ${v * 1e-9:.1f}B", va="center", fontsize=12, fontweight="bold", color="#333333")

    for ax in axes.flat:
        ax.grid(axis="x", linestyle="--", alpha=0.6, color="#E0E0E0")
        ax.set_axisbelow(True)

    sns.despine(fig)
    plt.tight_layout(pad=3.0)
    st.pyplot(fig)
    plt.close(fig)
