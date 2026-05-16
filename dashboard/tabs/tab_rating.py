"""
tabs/tab_rating.py
Tab: 评分分析 + 相关性分析（Pearson / Spearman）
来源逻辑: analysis/main.py + analysis/main_visulization.py
         analysis/pearson_and_spearman.py + pearson_and_spearman_visulization.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats

from data_loader import load_rating_data


def render() -> None:
    st.header("评分分析 (Rating Tier Analysis)")
    st.caption("数据来源: movies_metadata_cleaned.csv + ratings_small_cleaned.csv")

    # ── 核心发现 ──────────────────────────────────────────────────────────
    st.subheader("🔍 核心发现 (Core Insights)")
    st.info(
        "通过对 **1203 部**有效电影样本的分析，我们得到了一个相当反直觉的结论："
        "**\u201c叫好\u201d并不等于\u201c叫座\u201d**。"
    )
    col_a, col_b, col_c = st.columns(3)
    col_a.error(
        "**相关性极低**\n\n"
        "Pearson 与 Spearman 相关系数均约 **−0.01**，\n"
        "评分高低与票房收入几乎没有任何统计关联。"
    )
    col_b.warning(
        "**评分倒挂现象**\n\n"
        "低分电影（<3.0）的平均票房与中位数票房，\n"
        "反而**高于**高分电影（>4.0）。"
    )
    col_c.success(
        "**爆款逻辑**\n\n"
        "票房超 3.3 亿美元的\u201c超级爆款\u201d平均评分（3.30），\n"
        "甚至略低于普通电影（3.34）。"
    )
    st.divider()

    with st.spinner("加载数据中..."):
        df = load_rating_data()

    df = df[(df["avg_rating"] >= 0) & (df["avg_rating"] <= 5)].copy()
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df = df[df["revenue"] > 0].copy()
    df["revenue_m"] = df["revenue"] / 1_000_000

    # ── 评分分层 ──────────────────────────────────────────────────────────
    bins = [0, 2.999, 3.999, 5.01]
    labels = ["Low (<3.0)", "Medium (3.0-4.0)", "High (>4.0)"]
    df["rating_tier"] = pd.cut(df["avg_rating"], bins=bins, labels=labels)

    tier_stats = (
        df.groupby("rating_tier", observed=False)["revenue_m"]
        .agg(avg_revenue="mean", median_revenue="median", movie_count="count")
        .reset_index()
    )

    # ── 异常值分析 ────────────────────────────────────────────────────────
    Q1, Q3 = df["revenue"].quantile(0.25), df["revenue"].quantile(0.75)
    upper = Q3 + 1.5 * (Q3 - Q1)
    blockbusters = df[df["revenue"] > upper]
    normal = df[df["revenue"] <= upper]

    # ── Top 5 ─────────────────────────────────────────────────────────────
    top5 = df.nlargest(5, "revenue")[["title", "revenue_m", "avg_rating"]].copy()
    top5.columns = ["Title", "Revenue ($M)", "Avg Rating"]
    top5["Revenue ($M)"] = top5["Revenue ($M)"].round(1)
    top5["Avg Rating"] = top5["Avg Rating"].round(2)

    # ── 统计摘要 ──────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("有效样本", f"{len(df):,}")
    col2.metric(
        f"大片 (>{upper / 1e6:.0f}M$)",
        f"{len(blockbusters):,}",
        f"均评 {blockbusters['avg_rating'].mean():.2f}",
    )
    col3.metric(
        "普通电影",
        f"{len(normal):,}",
        f"均评 {normal['avg_rating'].mean():.2f}",
    )

    st.subheader("评分分层统计")
    st.dataframe(
        tier_stats.rename(columns={
            "rating_tier": "评分层",
            "avg_revenue": "平均票房 ($M)",
            "median_revenue": "中位票房 ($M)",
            "movie_count": "电影数量",
        }).style.format({"平均票房 ($M)": "{:.1f}", "中位票房 ($M)": "{:.1f}"}),
        use_container_width=True,
    )

    st.subheader("Top 5 票房电影")
    st.dataframe(top5, use_container_width=True)

    # ── 可视化 ────────────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=100)

    # 左图：均值 & 中位数柱状图
    melted = tier_stats.melt(
        id_vars="rating_tier",
        value_vars=["avg_revenue", "median_revenue"],
        var_name="Metric",
        value_name="Revenue ($M)",
    )
    melted["Metric"] = melted["Metric"].map({"avg_revenue": "Mean", "median_revenue": "Median"})
    sns.barplot(
        x="rating_tier", y="Revenue ($M)", hue="Metric",
        data=melted, palette=["#4C72B0", "#55A868"], ax=axes[0],
    )
    axes[0].set_title("Average & Median Revenue by Rating Tier", pad=12)
    axes[0].set_xlabel("Rating Tier")
    axes[0].set_ylabel("Box Office Revenue (Millions USD)")
    for p in axes[0].patches:
        h = p.get_height()
        if pd.notnull(h) and h > 0:
            axes[0].annotate(
                f"${h:.0f}M",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center", va="bottom", xytext=(0, 4),
                textcoords="offset points", fontsize=10,
            )

    # 右图：箱线图 + 散点
    sns.boxplot(
        x="rating_tier", y="revenue_m", data=df,
        palette="pastel", showfliers=False, ax=axes[1],
    )
    sns.stripplot(
        x="rating_tier", y="revenue_m", data=df,
        color="#C44E52", alpha=0.4, jitter=True, size=3, ax=axes[1],
    )
    axes[1].set_title("Revenue Distribution by Rating Tier", pad=12)
    axes[1].set_xlabel("Rating Tier")
    axes[1].set_ylabel("Box Office Revenue (Millions USD)")
    axes[1].set_ylim(0, 2000)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── 第一组图表解读 ────────────────────────────────────────────────────
    with st.expander("📊 第一组图表解读：分层对比图（柱状图 & 箱线图）", expanded=True):
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(
                "**左图（柱状图）：** 展示了低、中、高三个评分段的平均和中位数票房。"
                "最左侧的\u201c低分段\u201d柱子最高，说明市场上的商业大片往往评分并不高，但吸金能力最强。"
            )
        with col_r:
            st.markdown(
                "**右图（箱线图 + 散点图）：** 箱子代表大部分电影票房都在低位徘徊。"
                "上方的红点（离群点）代表《泰坦尼克号》等超级爆款——无论评分高低，每个阶段都有红点飞得极高，"
                "说明**爆款的产生具有随机性，并不依赖高评分**。"
            )

    # ══════════════════════════════════════════════════════════════════════
    # 相关性分析 (Pearson & Spearman)
    # ══════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("相关性分析 (Pearson & Spearman)")

    corr_pearson, p_pearson = stats.pearsonr(df["avg_rating"], df["revenue_m"])
    corr_spearman, p_spearman = stats.spearmanr(df["avg_rating"], df["revenue_m"])

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

    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7), dpi=100)
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

    # 左图：Pearson 线性回归
    sns.regplot(
        x="avg_rating", y="revenue_m", data=df, ax=axes2[0],
        scatter_kws={"alpha": 0.35, "color": "#4C72B0", "s": 25},
        line_kws={"color": "#C44E52", "linewidth": 2},
    )
    axes2[0].set_title("Linear Correlation (Pearson)", fontsize=15, pad=12)
    axes2[0].set_xlabel("Average Rating (1–5 Stars)", fontsize=13)
    axes2[0].set_ylabel("Box Office Revenue (Millions USD)", fontsize=13)
    axes2[0].text(
        0.05, 0.95,
        f"Pearson r = {corr_pearson:.4f}\np-value = {p_pearson:.4f}\n"
        "Measures linear relationship.\nNear 0: no straight line fits well.",
        transform=axes2[0].transAxes, fontsize=11,
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
        x="avg_rating", y="revenue_m", data=df, ax=axes2[1],
        alpha=0.2, color="gray", s=20,
    )
    sns.lineplot(
        x="rating_mid", y="revenue_m", data=q_stats, ax=axes2[1],
        color="#55A868", linewidth=3, marker="o", markersize=9,
    )
    axes2[1].set_title("Rank Correlation (Spearman)", fontsize=15, pad=12)
    axes2[1].set_xlabel("Average Rating (1–5 Stars)", fontsize=13)
    axes2[1].set_ylabel("Box Office Revenue (Millions USD)", fontsize=13)
    axes2[1].set_ylim(axes2[0].get_ylim())
    axes2[1].text(
        0.05, 0.95,
        f"Spearman ρ = {corr_spearman:.4f}\np-value = {p_spearman:.4f}\n"
        "Measures monotonic (rank) relationship.\n"
        "Green line: median revenue trend.",
        transform=axes2[1].transAxes, fontsize=11,
        verticalalignment="top", bbox=props,
    )

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # ── 第二组图表解读 ────────────────────────────────────────────────────
    with st.expander("📊 第二组图表解读：相关性深度分析（Pearson & Spearman）", expanded=True):
        col_l2, col_r2 = st.columns(2)
        with col_l2:
            st.markdown(
                "**左图（Pearson 线性拟合）：** 中间那条红色的拟合线几乎是**水平的**。"
                "如果评分越高票房越高，线应向右上方倾斜。"
                "平直的线说明：评分增加，票房并不会随之增加。"
            )
        with col_r2:
            st.markdown(
                "**右图（Spearman 等级趋势）：** 绿色折线代表随评分上升的票房中位数走势。"
                "折线起伏不定，甚至在高分段出现下滑，进一步证明**\u201c高排名评分\u201d并不能带来\u201c高排名票房\u201d**。"
            )
