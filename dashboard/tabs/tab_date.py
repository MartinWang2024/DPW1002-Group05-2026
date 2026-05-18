"""
tabs/tab_date.py
Tab: 上映日期分析
来源逻辑: analysis/movie_analysis.ipynb
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from data_loader import load_box_office_full


MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def render() -> None:
    st.header("上映日期分析 (Release Date Analysis)")
    st.caption("数据来源: movies_metadata_box_office.csv")

    st.subheader("Core Insights")
    st.info(
        "基于电影上映时间与票房数据的分析可以看出，电影票房不仅具有显著的长期增长趋势，"
        "也呈现出明显的季节性规律。年份维度反映产业规模扩张，月份维度则揭示档期选择对"
        "商业表现的直接影响。"
    )
    col_a, col_b = st.columns(2)
    col_a.success(
        "长期增长趋势\n\n"
        "从 1920 年至今，电影平均票房整体显著上升，尤其 2000 年后增长更快，"
        "反映出电影市场规模和商业化程度持续扩张。"
    )
    col_b.warning(
        "档期效应明显\n\n"
        "5-7 月及 11 月是平均票房最高的黄金档，1 月和 9 月表现最弱，说明上映月份"
        "会显著影响商业表现。"
    )

    st.divider()

    with st.spinner("加载数据中..."):
        df = load_box_office_full()

    df = df.copy()
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    df["release_month"] = pd.to_datetime(df["release_date"], errors="coerce").dt.month
    df = df[(df["revenue"] > 0) & df["release_year"].notna() & df["release_month"].notna()].copy()
    df["release_year"] = df["release_year"].astype(int)
    df["release_month"] = df["release_month"].astype(int)
    df["revenue_m"] = df["revenue"] / 1_000_000

    yearly_revenue = (
        df.groupby("release_year", observed=False)["revenue_m"]
        .mean()
        .reset_index()
        .sort_values("release_year")
    )

    monthly_revenue = (
        df.groupby("release_month", observed=False)["revenue_m"]
        .mean()
        .reindex(range(1, 13))
        .reset_index()
    )

    model_df = df[["release_month", "budget", "revenue_m"]].dropna().copy()
    X = model_df[["release_month", "budget"]]
    y = model_df["revenue_m"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    c1, c2, c3 = st.columns(3)
    c1.metric("有效样本", f"{len(df):,}")
    c2.metric("最高票房月份", MONTH_LABELS[int(monthly_revenue.loc[monthly_revenue['revenue_m'].idxmax(), 'release_month']) - 1])
    c3.metric("年份跨度", f"{yearly_revenue['release_year'].min()} - {yearly_revenue['release_year'].max()}")

    c4, c5 = st.columns(2)
    c4.metric("回归样本", f"{len(model_df):,}")
    c5.metric("回归 MSE", f"{mse:,.1f}")

    sns.set_theme(style="whitegrid", context="talk")

    st.subheader("① 按上映年份划分的平均票房")
    fig1, ax1 = plt.subplots(figsize=(14, 6), dpi=100)
    sns.lineplot(
        data=yearly_revenue,
        x="release_year",
        y="revenue_m",
        color="#4C72B0",
        linewidth=2.5,
        ax=ax1,
    )
    ax1.set_title("Average Revenue by Release Year", fontsize=16, pad=12)
    ax1.set_xlabel("Release Year")
    ax1.set_ylabel("Average Revenue (Millions USD)")
    ax1.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    with st.expander("📊 图表解读：按上映年份划分的平均票房", expanded=True):
        st.markdown(
            "这张折线图展示了电影平均票房随上映年份的变化趋势。\n\n"
            "1. 整体来看，从 1920 年至今，电影的平均票房呈现出长期显著的上升趋势，尤其在 2000 年后增长速度明显加快，反映出电影市场规模的持续扩张。\n"
            "2. 数据中存在明显的波动和局部峰值，例如 1940 年前后的一个高值点，以及 2010 年后持续走高的曲线，说明票房增长并非匀速，受时代背景、产业发展和头部爆款影片的影响较大。"
        )

    st.divider()

    st.subheader("② 按上映月份划分的平均票房")
    fig2, ax2 = plt.subplots(figsize=(14, 6), dpi=100)
    sns.barplot(
        data=monthly_revenue,
        x="release_month",
        y="revenue_m",
        palette="crest",
        ax=ax2,
    )
    ax2.set_title("Average Revenue by Release Month", fontsize=16, pad=12)
    ax2.set_xlabel("Release Month")
    ax2.set_ylabel("Average Revenue (Millions USD)")
    ax2.set_xticks(range(12))
    ax2.set_xticklabels(MONTH_LABELS)
    for patch in ax2.patches:
        height = patch.get_height()
        if pd.notna(height):
            ax2.annotate(
                f"{height:.0f}",
                (patch.get_x() + patch.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=10,
                xytext=(0, 4),
                textcoords="offset points",
            )
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    with st.expander("📊 图表解读：按上映月份划分的平均票房", expanded=True):
        st.markdown(
            "这张柱状图对比了不同月份上映电影的平均票房表现，揭示了明显的季节性规律：\n\n"
            "1. 票房表现的黄金档集中在 5 月、6 月、7 月和 11 月，其中 5 月和 6 月的平均票房最高，是全年的票房高峰。\n"
            "2. 相比之下，1 月和 9 月的平均票房为全年最低，说明上映月份对电影票房有显著影响，暑期档和年末档更易产出高票房作品。"
        )

    st.divider()

    st.subheader("③ 回归分析：实际票房 vs 预测票房")
    fig3, ax3 = plt.subplots(figsize=(14, 6), dpi=100)
    sns.scatterplot(
        x=y_test,
        y=y_pred,
        color="#4C72B0",
        alpha=0.5,
        s=35,
        ax=ax3,
    )
    max_val = max(float(y_test.max()), float(pd.Series(y_pred).max()))
    ax3.plot([0, max_val], [0, max_val], linestyle="--", color="#C44E52", linewidth=2)
    ax3.set_title("Actual Revenue vs Predicted Revenue", fontsize=16, pad=12)
    ax3.set_xlabel("Actual Revenue (Millions USD)")
    ax3.set_ylabel("Predicted Revenue (Millions USD)")
    ax3.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    with st.expander("📊 图表解读：实际票房 vs 预测票房", expanded=True):
        st.markdown(
            "这张散点图用于评估票房预测模型的效果，横轴为电影的实际票房，纵轴为模型预测的票房。\n\n"
            "1. 数据点整体呈现出从左下到右上的正相关趋势，说明模型的预测结果与实际票房有一定的正相关性，能够捕捉到票房的大致规律。\n"
            "2. 但数据点分布较为分散，尤其是在高票房区间（实际票房 > 5 亿），预测值与实际值的偏差明显增大，说明模型对头部爆款电影的预测能力较弱，整体预测精度仍有提升空间。"
        )