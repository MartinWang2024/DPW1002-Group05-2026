"""
tabs/tab_ml.py
Tab: 机器学习模型
"""

from pathlib import Path

import streamlit as st


def render() -> None:
    st.header("机器学习模型 (Machine Learning Models)")

    st.caption("本页面展示电影盈利能力预测的机器学习模型结果。")

    st.markdown(
        """
        ### Profitability Prediction Model

        In this section, we present a regression-based machine learning model for
        movie profitability analysis. The model is used to explore how movie-related
        features may help explain or predict profitability outcomes.

        The model code and output figure are stored in the `regression_model` folder.
        """
    )

    st.divider()

    st.subheader("Model Result Visualization")

    image_path = (
        Path(__file__).resolve().parents[2]
        / "regression_model"
        / "profitability_model_results.png"
    )

    if image_path.exists():
        st.image(
            str(image_path),
            caption="Profitability Model Results",
            use_container_width=True,
        )
    else:
        st.warning(
            "The result image was not found. "
            "Please check whether regression_model/profitability_model_results.png exists."
        )

    st.divider()

    st.subheader("Model File")

    st.markdown(
        """
        The machine learning script is available at:

        `regression_model/profitability_model.py`
        """
    )

    st.info(
        "This section connects the machine learning output to the dashboard, "
        "so users can view the model result directly from the web app."
    )
