import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats

from data_loader import load_rating_data


def render() -> None:
    st.header("Rating Analysis")
    st.caption("Data source: movies_metadata_cleaned.csv + ratings_small_cleaned.csv")

    # ── 核心发现 ──────────────────────────────────────────────────────────
    st.subheader("Core Insights")
    st.info(
        "Based on an analysis of **1,203 valid movie samples**, we reached a fairly counterintuitive conclusion: "
        "**critical acclaim does not necessarily translate into box office success**."
    )
    col_a, col_b, col_c = st.columns(3)
    col_a.error(
        "**Extremely Low Correlation**\n\n"
        "Both Pearson and Spearman correlation coefficients are approximately **−0.01**,\n"
        "indicating almost no statistical relationship between ratings and box office revenue."
    )
    col_b.warning(
        "**Rating Reversal Phenomenon**\n\n"
        "Movies with low ratings (<3.0) have average and median box office revenues\n"
        "that are actually **higher** than those of highly rated movies (>4.0)."
    )
    col_c.success(
        "**Blockbuster Logic**\n\n"
        "Movies with box office revenue exceeding $330M have an average rating of 3.30,\n"
        "which is slightly lower than that of regular movies (3.34)."
    )
    st.divider()

    with st.spinner("Loading data..."):
        df = load_rating_data()

    df = df[(df["avg_rating"] >= 0) & (df["avg_rating"] <= 5)].copy()
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df = df[df["revenue"] > 0].copy()
    df["revenue_m"] = df["revenue"] / 1_000_000

    # ── Rating Tiers ──────────────────────────────────────────────────────────
    bins = [0, 2.999, 3.999, 5.01]
    labels = ["Low (<3.0)", "Medium (3.0-4.0)", "High (>4.0)"]
    df["rating_tier"] = pd.cut(df["avg_rating"], bins=bins, labels=labels)

    tier_stats = (
        df.groupby("rating_tier", observed=False)["revenue_m"]
        .agg(avg_revenue="mean", median_revenue="median", movie_count="count")
        .reset_index()
    )

    # ── Outlier Analysis ────────────────────────────────────────────────────────
    Q1, Q3 = df["revenue"].quantile(0.25), df["revenue"].quantile(0.75)
    upper = Q3 + 1.5 * (Q3 - Q1)
    blockbusters = df[df["revenue"] > upper]
    normal = df[df["revenue"] <= upper]

    # ── Top 5 ─────────────────────────────────────────────────────────────
    top5 = df.nlargest(5, "revenue")[["title", "revenue_m", "avg_rating"]].copy()
    top5.columns = ["Title", "Revenue ($M)", "Avg Rating"]
    top5["Revenue ($M)"] = top5["Revenue ($M)"].round(1)
    top5["Avg Rating"] = top5["Avg Rating"].round(2)

    # ── Statistical Summary ──────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Valid Samples", f"{len(df):,}")
    col2.metric(
        f"Blockbusters (>{upper / 1e6:.0f}M$)",
        f"{len(blockbusters):,}",
        f"Avg Rating {blockbusters['avg_rating'].mean():.2f}",
    )
    col3.metric(
        "Regular Movies",
        f"{len(normal):,}",
        f"Avg Rating {normal['avg_rating'].mean():.2f}",
    )

    st.subheader("Rating Tier Statistics")
    st.dataframe(
        tier_stats.rename(columns={
            "rating_tier": "Rating Tier",
            "avg_revenue": "Average Revenue ($M)",
            "median_revenue": "Median Revenue ($M)",
            "movie_count": "Movie Count",
        }).style.format({"Average Revenue ($M)": "{:.1f}", "Median Revenue ($M)": "{:.1f}"}),
        use_container_width=True,
    )

    st.subheader("Top 5 Box Office Movies")
    st.dataframe(top5, use_container_width=True)

    # ── Visualization ────────────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=100)

    # Left Plot: Mean & Median Bar Chart
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

    # Right Plot: Boxplot + Scatter
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

    # ── First Set of Chart Interpretations ────────────────────────────────────────────────────
    with st.expander("Visuals Explanation", expanded=True):
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(
                "**Left Chart (Bar):** Shows the average and median box office revenue for low, medium, and high rating tiers."
                " The tallest bar on the left indicates that blockbuster movies often have lower ratings but higher revenue."
            )
        with col_r:
            st.markdown(
                "**Right Chart (Boxplot + Scatter):** The boxes represent the majority of movie revenues staying low."
                " The red dots above (outliers) represent super blockbusters like 'Titanic' — regardless of rating, each tier has red dots flying high,"
                " indicating that **the occurrence of blockbusters is random and not dependent on high ratings**."
            )

    # ══════════════════════════════════════════════════════════════════════
    # Correlation Analysis (Pearson & Spearman)
    # ══════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Correlation Analysis (Pearson & Spearman)")

    corr_pearson, p_pearson = stats.pearsonr(df["avg_rating"], df["revenue_m"])
    corr_spearman, p_spearman = stats.spearmanr(df["avg_rating"], df["revenue_m"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Sample Size", f"{len(df):,}")
    c2.metric("Pearson r", f"{corr_pearson:.4f}", f"p = {p_pearson:.4f}")
    c3.metric("Spearman ρ", f"{corr_spearman:.4f}", f"p = {p_spearman:.4f}")

    with st.expander("Correlation Coefficients Explanation"):
        st.markdown(
            """
- **Pearson r**: Measures linear correlation. Close to 0 indicates no linear relationship between rating and box office.  
- **Spearman ρ**: Measures monotonic rank correlation, more robust as it doesn't assume normal distribution.  
- p-value < 0.05 indicates statistical significance.
"""
        )

    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7), dpi=100)
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

    # Left Plot: Pearson Linear Regression
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

    # Right Plot: Spearman Quantile Trend
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

    # ── Interpretation of the Second Set of Charts ────────────────────────────────────────────────────
    with st.expander("Visuals Explanation", expanded=True):
        col_l2, col_r2 = st.columns(2)
        with col_l2:
            st.markdown(
                "**Left Chart (Pearson Linear Fit):** The red fit line in the middle is almost **horizontal**."
                " If higher ratings led to higher box office, the line should slope upwards to the right."
                " The flat line indicates that increasing ratings do not correspond to higher box office."
            )
        with col_r2:
            st.markdown(
                "**Right Chart (Spearman Rank Trend):** The green line represents the median revenue trend as ratings increase."
                " The line fluctuates and even declines in the high-rating segment, further confirming that **\u201chigh ratings\u201d do not necessarily lead to \u201chigh box office\u201d**."
            )
