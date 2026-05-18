"""
tabs/tab_keywords.py
Tab: 关键词分析
来源逻辑: analysis/movie_analysis.ipynb
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.ticker import FuncFormatter

from data_loader import load_box_office_full, load_keywords_data


def render() -> None:
    st.header("关键词分析 (Keyword Analysis)")
    st.caption("数据来源: movies_metadata_box_office.csv + keywords_cleaned.csv")

    st.subheader("Core Insights")
    st.info(
        "基于电影关键词与票房数据的交叉分析可以看出，不同题材设定之间存在明显的市场表现差异。"
        "部分关键词不仅拥有更高的平均票房，也更容易产生头部爆款；而另一些关键词虽然均值较高，"
        "但样本集中度强，解释时需要结合分布情况一起判断。"
    )
    col_a, col_b = st.columns(2)
    col_a.success(
        "高票房关键词\n\n"
        "平均票房排名靠前的关键词显示出显著的市场吸引力，说明特定故事设定会对商业表现产生明显拉动。"
    )
    col_b.warning(
        "均值之外看分布\n\n"
        "仅看均值并不足够，箱线图能进一步区分是持续稳定高票房，还是由少数爆款或小样本抬高平均值。"
    )

    st.divider()

    with st.spinner("加载数据中..."):
        movies_df = load_box_office_full()
        keywords_df = load_keywords_data()

    movies_df = movies_df.copy()
    movies_df["revenue"] = pd.to_numeric(movies_df["revenue"], errors="coerce")
    movies_df = movies_df[movies_df["revenue"] > 0].copy()

    merged_df = pd.merge(movies_df, keywords_df, on="id", how="left")
    merged_df = merged_df.dropna(subset=["keyword_name", "revenue"]).copy()
    merged_df["revenue_m"] = merged_df["revenue"] / 1_000_000

    keyword_revenue_df = (
        merged_df.groupby("keyword_name", observed=False)["revenue_m"]
        .agg(mean="mean", count="count")
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    top_keywords = keyword_revenue_df.head(10).copy()
    top_keyword_names = top_keywords["keyword_name"].tolist()
    filtered_df = merged_df[merged_df["keyword_name"].isin(top_keyword_names)].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("有效样本", f"{len(merged_df):,}")
    c2.metric("关键词数量", f"{keyword_revenue_df['keyword_name'].nunique():,}")
    c3.metric("Top 关键词", top_keywords.iloc[0]["keyword_name"] if not top_keywords.empty else "N/A")

    sns.set_theme(style="whitegrid", context="talk")

    st.subheader("① 按关键词划分的平均票房 Top10")
    fig1, ax1 = plt.subplots(figsize=(14, 7), dpi=100)
    sns.barplot(
        data=top_keywords,
        x="mean",
        y="keyword_name",
        palette="viridis",
        ax=ax1,
    )
    ax1.set_title("Top 10 Keywords by Average Revenue", fontsize=16, pad=12)
    ax1.set_xlabel("Average Revenue (Millions USD)")
    ax1.set_ylabel("Keyword")
    for patch in ax1.patches:
        width = patch.get_width()
        if pd.notna(width):
            ax1.annotate(
                f"{width:.1f}",
                (width, patch.get_y() + patch.get_height() / 2),
                ha="left",
                va="center",
                fontsize=10,
                xytext=(6, 0),
                textcoords="offset points",
            )
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    with st.expander("📊 图表解读：按关键词划分的平均票房 Top10", expanded=True):
        st.markdown(
            "这张图展示了平均票房最高的 10 个电影关键词 / 题材，直观呈现了不同主题的市场表现差异。\n\n"
            "1. 'anti war（反战）' 题材的电影平均票房遥遥领先，显著高于其他关键词，是最具票房潜力的主题。\n"
            "2. 紧随其后的是 'steerage'、'rich woman - poor man（贫富恋）'、'salvage（救援）' 等题材，说明特定类型的故事设定对票房有较强的拉动作用，而 'mysterious woman（神秘女性）' 等题材的平均票房相对较低。"
        )

    st.divider()

    st.subheader("② Top 关键词的票房分布")
    fig2, ax2 = plt.subplots(figsize=(14, 7), dpi=100)
    sns.boxplot(
        data=filtered_df,
        x="keyword_name",
        y="revenue",
        showfliers=False,
        width=0.6,
        color="lightblue",
        ax=ax2,
    )
    sns.stripplot(
        data=filtered_df,
        x="keyword_name",
        y="revenue",
        color="darkred",
        size=7,
        jitter=0.25,
        alpha=0.75,
        ax=ax2,
    )
    ax2.set_title("Revenue Distribution by Top Keywords", fontsize=16, pad=12)
    ax2.set_xlabel("Keyword")
    ax2.set_ylabel("Revenue (Billion USD)")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x / 1e9:.1f}"))
    ax2.scatter([], [], color="red", label="Individual Data Points")
    ax2.legend(loc="upper right")
    ax2.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    with st.expander("📊 图表解读：Top 关键词的票房分布", expanded=True):
        st.markdown(
            "这张箱线图进一步展示了高票房关键词题材的票房分布情况，补充了均值之外的细节信息。\n\n"
            "1. 'anti war（反战）' 和 'space war（太空战争）'、'power relations（权力关系）' 等题材的票房分布范围更广，且存在极高的上限值，说明这些题材不仅平均票房高，也更容易诞生票房爆款。\n"
            "2. 相比之下，'rich woman - poor man（贫富恋）'、'steerage' 等题材的票房数据几乎是单一值，说明样本量较少或票房表现高度集中，数据的代表性较弱。"
        )