import sys
from pathlib import Path

import streamlit as st

_DASHBOARD_DIR = Path(__file__).resolve().parent
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

from data_loader import (  
    ANALYSIS_DIR,
    load_movies_data,
    load_rating_data,
)
from tabs import ( 
    tab_index,
    tab_budget,
    tab_country,
    tab_date,
    tab_keywords,
    tab_people,
    tab_rating,
    tab_ml,
)

st.set_page_config(page_title="Movie Analytics Dashboard", layout="wide")


def main() -> None:
    st.title("Movie Data Analytics Dashboard")
    st.caption("Built from the cleaned_archive dataset to present analyses of budget, box office, ratings, and countries.")

    movies_df = load_movies_data()
    rating_df = load_rating_data()

    # ── Sidebar Navigation ────────────────────────────────────────────────
    with st.sidebar:
        st.header("Analysis Sections")
        page = st.radio(
            "Select Page",
            options=[
                "Data Overview",
                "Country Analysis",
                "Release Date Analysis",
                "Keyword Analysis",
                "Actors & Directors Analysis",
                "Rating Analysis",
                "Budget Analysis",
                "Machine Learning Models",
            ],
            index=0,
        )

    if page == "Data Overview":
        tab_index.render(movies_df, rating_df)
    elif page == "Country Analysis":
        tab_country.render()
    elif page == "Release Date Analysis":
        tab_date.render()
    elif page == "Keyword Analysis":
        tab_keywords.render()
    elif page == "Actors & Directors Analysis":
        tab_people.render()
    elif page == "Rating Analysis":
        tab_rating.render()
    elif page == "Budget Analysis":
        tab_budget.render()
    elif page == "Machine Learning Models":
        tab_ml.render()


if __name__ == "__main__":
    main()
