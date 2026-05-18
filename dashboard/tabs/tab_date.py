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
    st.header("Release Date Analysis")
    st.caption("Data Source: movies_metadata_box_office.csv")

    st.subheader("Core Insights")
    st.info(
        "Analysis based on movie release dates and box office data shows that movie box office not only has a significant long-term growth trend, "
        "but also exhibits clear seasonal patterns. The yearly dimension reflects the expansion of industry scale, while the monthly dimension reveals the direct impact of release timing on commercial performance."
    )
    col_a, col_b = st.columns(2)
    col_a.success(
        "Long-term Growth Trend\n\n"
        "Global box office shows a long-term upward trend, with accelerated growth after 2000, indicating continuous expansion of market scale and commercial value."
    )
    col_b.warning(
        "Significant Release Timing Effect\n\n"
        "The summer months of May–June and the year-end period of November are golden windows, with significantly higher average box office, making them the optimal release periods of the year."
    )

    st.divider()

    with st.spinner("Loading data..."):
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
    c1.metric("Valid Samples", f"{len(df):,}")
    c2.metric("Highest Revenue Month", MONTH_LABELS[int(monthly_revenue.loc[monthly_revenue['revenue_m'].idxmax(), 'release_month']) - 1])
    c3.metric("Year Span", f"{yearly_revenue['release_year'].min()} - {yearly_revenue['release_year'].max()}")

    c4, c5 = st.columns(2)
    c4.metric("Regression Samples", f"{len(model_df):,}")
    c5.metric("Regression MSE", f"{mse:,.1f}")

    sns.set_theme(style="whitegrid", context="talk")

    st.subheader("1. Average Revenue by Release Year")
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

    with st.expander("Visuals Explanation", expanded=True):
        st.markdown(
            "This line chart shows the trend of average movie revenue over the years.\n\n"
            "1. Overall, from 1920 to the present, the average movie revenue shows a long-term significant upward trend, especially after 2000, indicating continuous expansion of the movie market.\n"
            "2. There are noticeable fluctuations and local peaks in the data, such as a high point around 1940 and a continuous rise after 2010, indicating that revenue growth is not uniform and is influenced by historical context, industry development, and blockbuster movies."
        )

    st.divider()

    st.subheader("2. Average Revenue by Release Month")
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

    with st.expander("Visuals Explanation", expanded=True):
        st.markdown(
            "This bar chart compares the average movie revenue across different release months, revealing clear seasonal patterns:\n\n"
            "1. The golden periods for box office performance are concentrated in May, June, July, and November, with May and June having the highest average revenue, representing the peak months of the year.\n"
            "2. In contrast, January and September have the lowest average revenue, indicating that the release month significantly impacts movie revenue, with summer and year-end periods more likely to produce high-grossing films."
        )

    st.divider()

    st.subheader("3. Regression Analysis: Actual Revenue vs Predicted Revenue")
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

    with st.expander("Visuals Explanation", expanded=True):
        st.markdown(
            "This scatter plot is used to evaluate the performance of the box office prediction model, with the x-axis representing the actual revenue of the movies and the y-axis representing the predicted revenue by the model.\n\n"
            "1. The data points generally show a positive correlation trend from the bottom left to the top right, indicating that the model's predictions have a certain positive correlation with the actual revenue and can capture the general pattern of box office performance.\n"
            "2. However, the data points are relatively scattered, especially in the high-revenue range (actual revenue > 500 million), where the deviation between predicted and actual values is significant, indicating that the model's ability to predict blockbuster movies is weak, and there is still room for improvement in overall prediction accuracy."
        )