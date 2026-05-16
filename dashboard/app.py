"""
app.py
Movie Analytics Dashboard — 主框架与启动器
每个 Tab 的逻辑独立维护在 dashboard/tabs/ 目录下。
"""

import sys
from pathlib import Path

import streamlit as st

# 确保 dashboard/ 目录在 sys.path 中，以便 tabs/* 能直接 import data_loader
_DASHBOARD_DIR = Path(__file__).resolve().parent
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

from data_loader import (  # noqa: E402
    ANALYSIS_DIR,
    load_movies_data,
    load_rating_data,
)
from tabs import (  # noqa: E402
    tab_index,
    tab_buget,
    tab_country,
    tab_people,
    tab_rating,
    tab_regression,
    tab_ml,
)

st.set_page_config(page_title="Movie Analytics Dashboard", page_icon="🎬", layout="wide")


def main() -> None:
    st.title("Movie Data Analytics Dashboard")
    st.caption("基于 cleaned_archive 数据构建，用于展示预算、票房、评分和国家维度分析。")

    movies_df = load_movies_data()
    rating_df = load_rating_data()

    # ── Sidebar Navigation ────────────────────────────────────────────────
    with st.sidebar:
        st.header("📊 分析模块")
        page = st.radio(
            "选择页面",
            options=[
                "数据概览",
                "国家分析",
                "人物分析",
                "评分分析",
                "回归模型",
                "预算分析",
                "机器学习模型",
            ],
            index=0,
        )

    if page == "数据概览":
        tab_index.render(movies_df, rating_df)
    elif page == "国家分析":
        tab_country.render()
    elif page == "人物分析":
        tab_people.render()
    elif page == "评分分析":
        tab_rating.render()
    elif page == "回归模型":
        tab_regression.render()
    elif page == "预算分析":
        tab_buget.render(ANALYSIS_DIR)
    elif page == "机器学习模型":
        tab_ml.render()


if __name__ == "__main__":
    main()
