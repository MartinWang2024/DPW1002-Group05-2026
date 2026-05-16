"""
tabs/tab_buget.py
Tab: Budget Analysis
直接从 cleaned_archive/movies_metadata_box_office.csv 读取数据并生成图表。
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from data_loader import load_movies_data


def render() -> None:
    st.header("Budget Analysis")
    st.caption("数据来源: movies_metadata_box_office.csv")

    with st.spinner("加载数据中..."):
        df = load_movies_data()

    # ── ROI & 利润计算 ────────────────────────────────────────────────────
    df = df.copy()
    df["ROI_pct"] = ((df["revenue"] - df["budget"]) / df["budget"]) * 100

    # 筛选 1950 年以后
    df_analysis = df[df["release_year"] >= 1950].copy()

    # 预算分箱（10 个等宽区间）
    df_analysis["budget_bin"] = pd.cut(df_analysis["budget_M"], bins=10, labels=False)
    budget_mid = (
        df_analysis.groupby("budget_bin", observed=False)["budget_M"]
        .median()
        .reset_index()
    )

    sns.set_style("whitegrid")

    # ── 指标概览 ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("有效样本", f"{len(df_analysis):,} 部")
    c2.metric("平均预算", f"${df_analysis['budget_M'].mean():.1f}M")
    c3.metric("平均 ROI", f"{df_analysis['ROI_pct'].mean():.1f}%")
    c4.metric("中位数 ROI", f"{df_analysis['ROI_pct'].median():.1f}%")

    st.divider()

    # ── 图1：预算分布直方图 ───────────────────────────────────────────────
    st.subheader("1. Movie Budget Distribution")
    max_budget_M = 100
    data_hist = df_analysis[df_analysis["budget_M"] <= max_budget_M]["budget_M"]
    fig1, ax1 = plt.subplots(figsize=(12, 5), dpi=100)
    sns.histplot(data_hist, bins=50, color="teal", edgecolor="white", alpha=0.85,
                 kde=True, ax=ax1)
    ax1.set_title(f"Movie Budget Distribution (≤ {max_budget_M}M$)", fontweight="bold")
    ax1.set_xlabel("Budget (M$)")
    ax1.set_ylabel("Number of Movies")
    ax1.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    # ── 图2：年度中位数预算趋势 ───────────────────────────────────────────
    st.subheader("2. Median Movie Budget Trend (Since 1950)")
    budget_trend = (
        df_analysis.groupby("release_year", observed=False)
        .agg(Median_Budget_M=("budget_M", "median"), Sample_Count=("id", "count"))
        .reset_index()
    )
    budget_trend = budget_trend[budget_trend["Sample_Count"] >= 5]
    fig2, ax2 = plt.subplots(figsize=(12, 5), dpi=100)
    ax2.plot(budget_trend["release_year"], budget_trend["Median_Budget_M"],
             marker="o", linewidth=2.5, color="crimson", markersize=4)
    ax2.set_title("Median Movie Budget Trend (Since 1950)", fontweight="bold")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Median Budget (M$)")
    ax2.set_xticks(np.arange(1950, budget_trend["release_year"].max() + 1, 10))
    ax2.set_xlim(1950, budget_trend["release_year"].max())
    ax2.grid(linestyle="--", alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # ── 图3：预算 vs 全球票房 相关性 ──────────────────────────────────────
    st.subheader("3. Budget vs Revenue Correlation")
    corr = df_analysis["budget_M"].corr(df_analysis["revenue_M"])
    fig3, ax3 = plt.subplots(figsize=(10, 6), dpi=100)
    sns.regplot(
        x="budget_M", y="revenue_M", data=df_analysis, ax=ax3,
        scatter_kws={"alpha": 0.45, "s": 20},
        line_kws={"color": "red", "linewidth": 2},
        ci=95,
    )
    ax3.set_title(f"Budget vs Revenue  |  Pearson r = {corr:.2f}", fontweight="bold")
    ax3.set_xlabel("Budget (M$)")
    ax3.set_ylabel("Revenue (M$)")
    ax3.set_xlim(0, df_analysis["budget_M"].quantile(0.99))
    ax3.set_ylim(0, df_analysis["revenue_M"].quantile(0.99))
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # ── 图4：不同预算区间的票房表现 ───────────────────────────────────────
    st.subheader("4. Box Office Performance by Budget Tier")
    budget_revenue_trend = (
        df_analysis.groupby("budget_bin", observed=False)
        .agg(Median_Revenue_M=("revenue_M", "median"), Mean_Revenue_M=("revenue_M", "mean"),
             Sample_Count=("id", "count"))
        .reset_index()
    )
    budget_revenue_trend = pd.merge(budget_revenue_trend, budget_mid, on="budget_bin")
    fig4, ax4 = plt.subplots(figsize=(11, 5), dpi=100)
    ax4.plot(budget_revenue_trend["budget_M"], budget_revenue_trend["Median_Revenue_M"],
             marker="o", linewidth=2.5, color="#377eb8", markersize=6, label="Median Revenue")
    ax4.plot(budget_revenue_trend["budget_M"], budget_revenue_trend["Mean_Revenue_M"],
             marker="s", linewidth=2, color="#4daf4a", markersize=4, alpha=0.8,
             label="Mean Revenue")
    ax4.set_title("Box Office Performance by Budget Tier", fontweight="bold")
    ax4.set_xlabel("Budget (M$)")
    ax4.set_ylabel("Revenue (M$)")
    ax4.grid(linestyle="--", alpha=0.6)
    ax4.set_xlim(0, df_analysis["budget_M"].quantile(0.99))
    ax4.legend()
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    # ── 图5：ROI 分布直方图 ───────────────────────────────────────────────
    st.subheader("5. Movie ROI Distribution")
    max_roi = df_analysis["ROI_pct"].quantile(0.95)
    min_roi = df_analysis["ROI_pct"].quantile(0.05)
    roi_data = df_analysis[
        (df_analysis["ROI_pct"] >= min_roi) & (df_analysis["ROI_pct"] <= max_roi)
    ]["ROI_pct"]
    median_roi = roi_data.median()
    mean_roi = roi_data.mean()
    fig5, ax5 = plt.subplots(figsize=(12, 5), dpi=100)
    sns.histplot(roi_data, bins=50, color="purple", edgecolor="white", alpha=0.85,
                 kde=True, ax=ax5)
    ax5.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Break-even (0% ROI)")
    ax5.axvline(x=median_roi, color="darkblue", linewidth=2,
                label=f"Median ROI: {median_roi:.1f}%")
    ax5.axvline(x=mean_roi, color="orange", linestyle="--", linewidth=2,
                label=f"Mean ROI: {mean_roi:.1f}%")
    ax5.set_title("Movie Investment Return (ROI) Distribution", fontweight="bold")
    ax5.set_xlabel("ROI (%)")
    ax5.set_ylabel("Number of Movies")
    ax5.grid(axis="y", alpha=0.4)
    ax5.legend()
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)

    # ── 图6：不同预算区间中位数 ROI ───────────────────────────────────────
    st.subheader("6. Median ROI by Budget Tier")
    budget_roi_trend = (
        df_analysis.groupby("budget_bin", observed=False)
        .agg(Median_ROI_pct=("ROI_pct", "median"), Mean_ROI_pct=("ROI_pct", "mean"),
             Sample_Count=("id", "count"))
        .reset_index()
    )
    budget_roi_trend = pd.merge(budget_roi_trend, budget_mid, on="budget_bin")
    fig6, ax6 = plt.subplots(figsize=(12, 5), dpi=100)
    sns.barplot(
        x="budget_M", y="Median_ROI_pct", hue="budget_M",
        data=budget_roi_trend, palette="Purples_d", legend=False, ax=ax6,
    )
    ax6.axhline(y=0, color="red", linestyle="--", linewidth=2, label="Break-even (0% ROI)")
    ax6.set_title("Median ROI by Budget Tier", fontweight="bold")
    ax6.set_xlabel("Budget (M$)")
    ax6.set_ylabel("Median ROI (%)")
    ax6.grid(axis="y", alpha=0.4)
    ax6.legend()
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close(fig6)
