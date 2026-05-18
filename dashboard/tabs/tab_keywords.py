import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.ticker import FuncFormatter

from data_loader import load_box_office_full, load_keywords_data


def render() -> None:
    st.header("Keyword Analysis")
    st.caption("Data Sources: movies_metadata_box_office.csv + keywords_cleaned.csv")

    st.subheader("Core Insights")
    st.info(
        "Cross-analysis of film keywords and box office data reveals significant differences in market performance among different themes and settings."
        "Some keywords not only have higher average box office revenue but also are more likely to produce blockbuster hits; while other keywords, although having high averages, "
        "have concentrated samples, and their interpretation needs to be considered in conjunction with the distribution."
    )
    col_a, col_b = st.columns(2)
    col_a.success(
        "High Revenue Keywords\n\n"
        "IP-related keywords have stable high revenue potential, such as space wars, fantasy, superheroes, etc. With sufficient samples, their box office performance is more robust, making them high-yield themes."
    )
    col_b.warning(
        "High Popularity Keywords\n\n"
        "High popularity keywords can significantly raise the box office ceiling. Quality theme tags can effectively boost box office expectations, and when combined with release timing, can further amplify revenue."
    )

    st.divider()

    with st.spinner("Loading data..."):
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
    c1.metric("Valid Samples", f"{len(merged_df):,}")
    c2.metric("Number of Keywords", f"{keyword_revenue_df['keyword_name'].nunique():,}")
    c3.metric("Top Keyword", top_keywords.iloc[0]["keyword_name"] if not top_keywords.empty else "N/A")

    sns.set_theme(style="whitegrid", context="talk")

    st.subheader("1. Top 10 Keywords by Average Revenue")
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

    with st.expander("Visuals Explanation", expanded=True):
        st.markdown(
            "This chart shows the top 10 movie keywords/themes by average box office revenue, providing a clear view of market performance differences among different themes.\n\n"
            "1. The 'anti war' theme leads in average box office revenue, significantly higher than other keywords, making it the most promising theme.\n"
            "2. Following closely are themes like 'steerage', 'rich woman - poor man', 'salvage', indicating that specific story settings have a strong impact on box office performance, while themes like 'mysterious woman' have relatively lower average box office revenue."
        )

    st.divider()

    st.subheader("2. Revenue Distribution of Top Keywords")
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

    with st.expander("Visuals Explanation", expanded=True):
        st.markdown(
            "This boxplot further illustrates the revenue distribution of high-grossing keyword themes, providing additional details beyond the mean.\n\n"
            "1. Themes like 'anti war', 'space war', and 'power relations' have a wider revenue distribution and extremely high upper limits, indicating that these themes not only have high average revenue but are also more likely to produce box office hits.\n"
            "2. In contrast, themes like 'rich woman - poor man' and 'steerage' have almost singular revenue values, suggesting a smaller sample size or highly concentrated box office performance, making the data less representative."
        )