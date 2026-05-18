import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from data_loader import load_movies_data


def render() -> None:
    st.header("Budget Analysis")
    st.caption("Data Source: movies_metadata_box_office.csv")

    # ── Core Insights & Summary ───────────────────────────────────────────
    st.subheader("Core Insights")
    st.info(
        "Based on valid movie sample data, this study systematically analyzes the relationship between budget, box office, and return on investment (ROI)."

        "The results indicate that the film industry is dominated by small and medium budgets, with a right-skewed distribution; production costs have been rising since 1950,"
        "with an accelerated increase after 2000. Budget and box office are strongly positively correlated, with high investment generally leading to high output, but with diminishing marginal returns;"
        "industry ROI is right-skewed, with more than half of the films being profitable, heavily reliant on blockbuster hits."
        "Low-budget (0–30 million USD) films offer the highest investment efficiency, while high-budget projects have lower returns and higher risks."
    )

    a1, a2, a3 = st.columns(3)
    a1.success(
        "**Cost Structure**\n\n"
        "Budget distribution is significantly right-skewed,\n"
        "with the industry mainly concentrated in small and medium budgets."
    )
    a2.warning(
        "**Long-term Trend**\n\n"
        "Production costs have been rising since 1950,\n"
        "with an accelerated increase after 2000."
    )
    a3.error(
        "**Return Efficiency**\n\n"
        "Higher budgets generally lead to higher box office,\n"
        "but ROI exhibits diminishing marginal returns."
    )

    st.divider()

    with st.spinner("Loading data..."):
        df = load_movies_data()

    # ── ROI & Profit Calculation ─────────────────────────────────────────────
    df = df.copy()
    df["ROI_pct"] = ((df["revenue"] - df["budget"]) / df["budget"]) * 100

    # Filter movies released after 1950
    df_analysis = df[df["release_year"] >= 1950].copy()

    # Budget binning (10 equal-width bins)
    df_analysis["budget_bin"] = pd.cut(df_analysis["budget_M"], bins=10, labels=False)
    budget_mid = (
        df_analysis.groupby("budget_bin", observed=False)["budget_M"]
        .median()
        .reset_index()
    )

    sns.set_style("whitegrid")

    # ── Metrics Overview ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Valid Samples", f"{len(df_analysis):,} ")
    c2.metric("Average Budget", f"${df_analysis['budget_M'].mean():.1f}M")
    c3.metric("Average ROI", f"{df_analysis['ROI_pct'].mean():.1f}%")
    c4.metric("Median ROI", f"{df_analysis['ROI_pct'].median():.1f}%")

    st.divider()

    # ── Figure 1: Movie Budget Distribution ───────────────────────────────────────────────
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

    with st.expander("Visuals Explanation", expanded=True):
        st.markdown(
            "**Purpose**: Show the distribution of movies with budgets under $100M, identifying the mainstream budget range."
        )
        st.markdown(
            "1. The budget distribution is significantly right-skewed, with the number of movies decreasing rapidly as the budget increases, reflecting a structure dominated by small and medium budgets with few blockbusters.\n"
            "2. The mainstream budget is concentrated between $0-40M, with $10-20M typically being the densest range.\n"
            "3. Movies with budgets over $80M are rare, representing high-cost blockbusters."
        )

    # ── Figure 2: Median Movie Budget Trend ───────────────────────────────────────────
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

    with st.expander("Visuals Explanation", expanded=True):
        st.markdown(
            "**Purpose**: Show the annual change in median production budget since 1950, reflecting long-term cost evolution."
        )
        st.markdown(
            "1. The median budget has been steadily increasing, indicating a rising threshold for movie production costs.\n"
            "2. The budget increase accelerated after 2000, with more noticeable growth after 2010.\n"
            "3. Local fluctuations are influenced by the supply structure of the year but do not change the long-term upward trend."
        )

    # ── Figure 3: Budget vs Revenue Correlation ──────────────────────────────────────
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

    with st.expander("Visuals Explanation", expanded=True):
        st.markdown(
            "**Purpose**: Quantify the linear relationship between budget and global box office revenue, and assess stability using a 95% confidence interval."
        )
        st.markdown(
            "1. Budget and revenue are significantly positively correlated, consistent with the industry rule of 'high investment, high return'.\n"
            "2. The regression line and its confidence interval indicate statistical stability of this relationship.\n"
            "3. The top 1% of extreme values are excluded, making the conclusion more representative of mainstream movie samples."
        )

    # ── Figure 4: Box Office Performance by Budget Tier ───────────────────────────────────────
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

    with st.expander("Visuals Explanation", expanded=True):
        st.markdown(
            "**Purpose**: Compare median/mean box office revenue across different budget tiers to observe the marginal returns of budget investment."
        )
        st.markdown(
            "1. As the budget increases, both median and mean box office revenue generally rise, indicating a positive impact of budget on revenue.\n"
            "2. Diminishing marginal returns: In the low-budget stage, each additional budget increment has a more noticeable impact, while in the high-budget stage, the effect tapers off.\n"
            "3. The mean is generally higher than the median, indicating the presence of blockbusters that elevate the mean, highlighting the head effect."
        )

    # ── Figure 5: ROI Distribution Histogram ───────────────────────────────────────────────
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

    with st.expander("Visuals Explanation", expanded=True):
        st.markdown(
            "**Purpose**: Display the industry ROI distribution structure and identify sample characteristics on both sides of the break-even line."
        )
        st.markdown(
            "1. ROI is right-skewed, with a few blockbusters significantly raising the overall mean.\n"
            "2. The median ROI is positive, indicating that more than half of the movies are profitable; however, the mean is significantly higher than the median, showing that profitability relies on the head.\n"
            "3. A considerable proportion of movies still have negative ROI, indicating inherent investment risks in the industry.\n"
            "4. The top and bottom 5% of extreme ROI values are excluded, improving the readability of mainstream samples."
        )

    # ── Figure 6: Median ROI by Budget Tier ───────────────────────────────────────
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

    with st.expander("Visuals Explanation", expanded=True):
        st.markdown(
            "**Purpose**: Compare the ROI efficiency across 10 budget tiers and identify high cost-performance budget ranges."
        )
        st.markdown(
            "1. ROI does not increase monotonically with budget; lower budget tiers often have higher median ROI.\n"
            "2. The $0-30M budget range typically offers higher investment cost-performance.\n"
            "3. High budget tiers tend to have lower median ROI, some approaching break-even, indicating higher risk and lower return efficiency."
        )
