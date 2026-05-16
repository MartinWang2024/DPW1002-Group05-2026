"""
tabs/tab_correlation.py
Tab: 相关性分析（Pearson / Spearman）
来源逻辑: analysis/pearson_and_spearman.py + pearson_and_spearman_visulization.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats

from data_loader import load_rating_data


def render() -> None:
    st.header("相关性分析 (Pearson & Spearman)")
    st.caption("数据来源: movies_metadata_cleaned.csv + ratings_small_cleaned.csv")

    with st.spinner("加载数据中..."):
        df = load_rating_data()

    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df = df[(df["revenue"] > 0) & df["revenue"].notna()].copy()
    df["revenue_m"] = df["revenue"] / 1_000_000

    # ── 相关系数计算 ──────────────────────────────────────────────────────
    corr_pearson, p_pearson = stats.pearsonr(df["avg_rating"], df["revenue_m"])
    corr_spearman, p_spearman = stats.spearmanr(df["avg_rating"], df["revenue_m"])

    # ── 指标展示 ──────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("有效样本", f"{len(df):,}")
    c2.metric("Pearson r", f"{corr_pearson:.4f}", f"p = {p_pearson:.4f}")
    c3.metric("Spearman ρ", f"{corr_spearman:.4f}", f"p = {p_spearman:.4f}")

    with st.expander("相关系数含义说明"):
        st.markdown(
            """
- **Pearson r**：衡量线性相关，接近 0 表示评分与票房不存在线性规律。  
- **Spearman ρ**：衡量单调秩相关，不依赖正态分布假设，更鲁棒。  
- p-value < 0.05 表示结果在统计上显著。
"""
        )

    # ── 可视化 ────────────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=100)
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

    # 左图：Pearson 线性回归
    sns.regplot(
        x="avg_rating", y="revenue_m", data=df, ax=axes[0],
        scatter_kws={"alpha": 0.35, "color": "#4C72B0", "s": 25},
        line_kws={"color": "#C44E52", "linewidth": 2},
    )
    axes[0].set_title("Linear Correlation (Pearson)", fontsize=15, pad=12)
    axes[0].set_xlabel("Average Rating (1–5 Stars)", fontsize=13)
    axes[0].set_ylabel("Box Office Revenue (Millions USD)", fontsize=13)
    axes[0].text(
        0.05, 0.95,
        f"Pearson r = {corr_pearson:.4f}\np-value = {p_pearson:.4f}\n"
        "Measures linear relationship.\nNear 0: no straight line fits well.",
        transform=axes[0].transAxes, fontsize=11,
        verticalalignment="top", bbox=props,
    )

    # 右图：Spearman 分位数趋势
    df["rating_quantile"] = pd.qcut(df["avg_rating"], q=10, duplicates="drop")
    q_stats = (
        df.groupby("rating_quantile", observed=False)["revenue_m"]
        .median()
        .reset_index()
    )
    q_stats["rating_mid"] = q_stats["rating_quantile"].apply(lambda x: x.mid)

    sns.scatterplot(
        x="avg_rating", y="revenue_m", data=df, ax=axes[1],
        alpha=0.2, color="gray", s=20,
    )
    sns.lineplot(
        x="rating_mid", y="revenue_m", data=q_stats, ax=axes[1],
        color="#55A868", linewidth=3, marker="o", markersize=9,
    )
    axes[1].set_title("Rank Correlation (Spearman)", fontsize=15, pad=12)
    axes[1].set_xlabel("Average Rating (1–5 Stars)", fontsize=13)
    axes[1].set_ylabel("Box Office Revenue (Millions USD)", fontsize=13)
    axes[1].set_ylim(axes[0].get_ylim())
    axes[1].text(
        0.05, 0.95,
        f"Spearman ρ = {corr_spearman:.4f}\np-value = {p_spearman:.4f}\n"
        "Measures monotonic (rank) relationship.\n"
        "Green line: median revenue trend.",
        transform=axes[1].transAxes, fontsize=11,
        verticalalignment="top", bbox=props,
    )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
