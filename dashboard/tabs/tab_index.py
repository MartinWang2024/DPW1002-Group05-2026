import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import build_country_stats


def render(movies_df: pd.DataFrame, rating_df: pd.DataFrame) -> None:

    # ── Filter Control ─────────────────────────────────────────
    with st.expander("Filtering criteria", expanded=True):
        year_min = int(movies_df["release_year"].min())
        year_max = int(movies_df["release_year"].max())
        col_y, col_b = st.columns(2)
        with col_y:
            selected_years = st.slider(
                "Release Year Range", min_value=year_min, max_value=year_max, value=(1980, year_max),
                key="idx_years",
            )
        with col_b:
            budget_cap = st.slider(
                "Budget Cap (M$)", min_value=10, max_value=500, value=120, step=10,
                key="idx_budget",
            )

    filtered = movies_df[
        (movies_df["release_year"] >= selected_years[0])
        & (movies_df["release_year"] <= selected_years[1])
        & (movies_df["budget_M"] <= budget_cap)
    ].copy()

    # ── Metrics ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Movie Count", f"{len(filtered):,}")
    c2.metric("Median Budget", f"${filtered['budget_M'].median():.1f}M")
    c3.metric("Median Revenue", f"${filtered['revenue_M'].median():.1f}M")
    c4.metric("Median Profit", f"${filtered['profit_M'].median():.1f}M")

    st.divider()

    # ── Budget vs Revenue ────────────────────────────────────────────────────────
    st.subheader("Budget vs Revenue")

    left, right = st.columns(2)
    with left:
        fig_hist = px.histogram(
            filtered,
            x="budget_M",
            nbins=40,
            title="Budget Distribution",
            labels={"budget_M": "Budget (M$)"},
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption("Note: The horizontal axis represents the budget (millions of US dollars), showing the distribution of movie budgets in the sample to help identify common budget ranges and tail values.")

    with right:
        trend = (
            filtered.groupby("release_year")["budget_M"]
            .median()
            .reset_index()
            .rename(columns={"budget_M": "median_budget_M"})
        )
        fig_trend = px.line(
            trend,
            x="release_year",
            y="median_budget_M",
            markers=True,
            title="Median Budget Trend by Year",
            labels={"release_year": "Year", "median_budget_M": "Median Budget (M$)"},
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        st.caption("Note: Shows the median budget of movies each year, helping to observe the trend of budget changes over time (median reduces the impact of extreme values).")

    corr = filtered["budget_M"].corr(filtered["revenue_M"])
    fig_scatter = px.scatter(
        filtered,
        x="budget_M",
        y="revenue_M",
        title=f"Budget vs Revenue (Correlation: {corr:.2f})",
        labels={"budget_M": "Budget (M$)", "revenue_M": "Revenue (M$)"},
        opacity=0.5,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption("Note: The scatter plot shows the relationship between budget and revenue; the distribution of points and the correlation coefficient above reflect the strength of their linear association.")

    st.divider()

    # ── Rating vs Revenue ──────────────────────────────────────────────────────────
    st.subheader("Rating vs Revenue")

    rating_df = rating_df[
        (rating_df["avg_rating"] >= 0) & (rating_df["avg_rating"] <= 5)
    ].copy()

    fig_rating = px.scatter(
        rating_df,
        x="avg_rating",
        y="revenue_M",
        color="rating_count",
        title="Rating vs Revenue",
        labels={
            "avg_rating": "Average Rating",
            "revenue_M": "Revenue (M$)",
            "rating_count": "Rating Count",
        },
        opacity=0.45,
    )
    st.plotly_chart(fig_rating, use_container_width=True)
    st.caption("Note: The horizontal axis represents the average rating (0–5), and the color indicates the rating count; this helps to observe the relationship between ratings and revenue, considering the sample weight.")

    pearson = rating_df["avg_rating"].corr(rating_df["revenue_M"], method="pearson")
    spearman = rating_df["avg_rating"].corr(rating_df["revenue_M"], method="spearman")
    st.info(f"Pearson: {pearson:.4f} | Spearman: {spearman:.4f}")

    st.divider()

    # ── Country Distribution ──────────────────────────────────────────────────────────
    st.subheader("Country Distribution")

    countries = build_country_stats(filtered)
    fig_country = px.bar(
        countries,
        x="movie_count",
        y="country",
        orientation="h",
        title="Top Countries by Movie Count",
        labels={"movie_count": "Movie Count", "country": "Country"},
    )
    fig_country.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_country, use_container_width=True)
    st.caption("Note: The bar chart shows the distribution of countries by movie count; you can refer to the table below for specific counts and proportions.")
    st.dataframe(countries, use_container_width=True)
