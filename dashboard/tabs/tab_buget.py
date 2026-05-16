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

    # ── Core Insights & Summary ───────────────────────────────────────────
    st.subheader("Core Insights")
    st.info(
        "基于有效电影样本数据，本研究系统分析了预算、票房与投资回报率（ROI）之间的关系。"
        "结果表明：电影行业以中小成本为主，预算呈右偏分布；1950 年以来制作成本持续上涨，"
        "2000 年后增速加快。预算与票房呈强正相关，高投入整体带来高产出，但边际回报递减；"
        "行业 ROI 呈右偏分布，半数以上电影可盈利，盈利高度依赖头部爆款。"
        "低预算（0–3000 万美元）区间投资性价比最高，而高预算项目回报效率偏低、风险更高。"
    )

    a1, a2, a3 = st.columns(3)
    a1.success(
        "**成本结构**\n\n"
        "预算分布显著右偏，\n"
        "行业主流集中在中小成本。"
    )
    a2.warning(
        "**长期趋势**\n\n"
        "1950 年以来制作成本上升，\n"
        "2000 年后增速明显加快。"
    )
    a3.error(
        "**回报效率**\n\n"
        "预算越高票房通常越高，\n"
        "但 ROI 边际效率递减。"
    )

    st.divider()

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
    c1.metric("有效样本", f"{len(df_analysis):,} ")
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

    with st.expander("图表 1 解读：电影预算分布直方图（≤1 亿美元）", expanded=True):
        st.markdown(
            "**图表目的**：展示 1 亿美元预算以内的电影数量分布特征，识别行业主流预算区间。"
        )
        st.markdown(
            "1. 预算呈显著右偏分布，电影数量随预算升高快速下降，符合“中小成本为主、大制作稀少”的结构。\n"
            "2. 主流预算集中在 0-4000 万美元，且 1000-2000 万美元通常是密度最高区间。\n"
            "3. 8000 万美元以上电影占比很低，属于头部高成本项目。"
        )

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

    with st.expander("图表 2 解读：1950 年以来电影年度预算趋势", expanded=True):
        st.markdown(
            "**图表目的**：展示 1950 年之后中位数制作预算的年度变化，反映长期成本演进。"
        )
        st.markdown(
            "1. 中位数预算整体持续上升，电影制作成本门槛不断提高。\n"
            "2. 2000 年后预算上涨加快，2010 年后增长更明显。\n"
            "3. 局部波动受当年供给结构影响，但不改变长期上升趋势。"
        )

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

    with st.expander("图表 3 解读：预算 vs 全球票房相关性", expanded=True):
        st.markdown(
            "**图表目的**：量化预算与全球票房的线性关系，并用 95% 置信区间评估稳定性。"
        )
        st.markdown(
            "1. 预算与票房呈显著正相关，整体符合“高投入高产出”的行业规律。\n"
            "2. 回归线及其置信区间显示该关系具有统计稳定性。\n"
            "3. 图中剔除了前 1% 极端值，结论更贴近主流电影样本。"
        )

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

    with st.expander("图表 4 解读：不同预算区间的票房表现", expanded=True):
        st.markdown(
            "**图表目的**：对比不同预算层级的中位数/平均票房，观察预算投入的边际产出变化。"
        )
        st.markdown(
            "1. 随预算提升，中位数和平均票房总体上升，预算投入对票房有正向拉动。\n"
            "2. 票房边际回报递减：低预算阶段每增加同等预算的拉动更明显，高预算阶段趋缓。\n"
            "3. 平均值普遍高于中位数，说明区间内存在爆款拉高均值，头部效应明显。"
        )

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

    with st.expander("图表 5 解读：电影投资回报率（ROI）分布", expanded=True):
        st.markdown(
            "**图表目的**：展示行业 ROI 分布结构，识别盈亏平衡线两侧样本特征。"
        )
        st.markdown(
            "1. ROI 呈右偏分布，少数爆款显著抬高整体均值。\n"
            "2. 中位数 ROI 为正，说明过半电影可盈利；但均值明显高于中位数，盈利依赖头部。\n"
            "3. 仍有可观比例电影 ROI 为负，行业投资风险客观存在。\n"
            "4. 本图剔除前后 5% 极端 ROI，提升了主流样本可读性。"
        )

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

    ranking_df = (
        budget_roi_trend.sort_values("Median_ROI_pct", ascending=False)
        [["budget_M", "Median_ROI_pct", "Mean_ROI_pct", "Sample_Count"]]
        .copy()
    )
    ranking_df["budget_M"] = ranking_df["budget_M"].round(1)
    ranking_df["Median_ROI_pct"] = ranking_df["Median_ROI_pct"].round(1)
    ranking_df["Mean_ROI_pct"] = ranking_df["Mean_ROI_pct"].round(1)

    st.markdown("**Budget Tier ROI Ranking (by Median ROI)**")
    st.dataframe(
        ranking_df.rename(
            columns={
                "budget_M": "Budget Mid (M$)",
                "Median_ROI_pct": "Median ROI (%)",
                "Mean_ROI_pct": "Mean ROI (%)",
                "Sample_Count": "Movie Count",
            }
        ),
        use_container_width=True,
    )

    with st.expander("图表 6 解读：不同预算区间 ROI 排名与对比", expanded=True):
        st.markdown(
            "**图表目的**：比较 10 个预算区间的 ROI 效率，识别高性价比预算带。"
        )
        st.markdown(
            "1. ROI 与预算并非同向增长，低预算区间常见更高中位数 ROI。\n"
            "2. 0-3000 万美元区间通常具有更高投资性价比。\n"
            "3. 高预算区间中位数 ROI 往往更低，部分接近盈亏平衡，回报效率与风险压力更大。"
        )
