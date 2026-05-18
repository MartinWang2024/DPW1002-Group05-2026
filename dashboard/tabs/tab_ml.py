"""
tabs/tab_ml.py
Tab: 机器学习模型
"""

import ast
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Colour palette ────────────────────────────────────────────
BLUE = "#4C72B0"
GREEN = "#55A868"
RED = "#C44E52"
ORANGE = "#DD8452"
PURPLE = "#8172B2"

DATA_ROOT = Path(__file__).resolve().parents[2] / "cleaned_archive"


def _train_models():
    """Load data, engineer features, train LR & RF, return all artefacts."""
    df = pd.read_csv(DATA_ROOT / "movies_metadata_cleaned.csv", low_memory=False)

    for col in ["budget", "revenue", "runtime", "release_year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()
    df["profitable"] = (df["revenue"] > df["budget"]).astype(int)

    def parse_list(s):
        try:
            return ast.literal_eval(s)
        except Exception:
            return []

    df["genres_parsed"] = df["genres_list"].apply(parse_list)

    TOP_GENRES = [
        "Drama", "Comedy", "Thriller", "Action", "Romance",
        "Adventure", "Crime", "Science Fiction", "Horror",
        "Family", "Fantasy", "Animation", "Mystery", "History", "War",
    ]
    for genre in TOP_GENRES:
        df[f'genre_{genre.replace(" ", "_")}'] = df["genres_parsed"].apply(
            lambda g, _g=genre: 1 if _g in g else 0
        )

    TOP_LANGS = ["en", "hi", "fr", "ru", "es", "ja", "zh", "it", "ta", "ko", "de"]
    for lang in TOP_LANGS:
        df[f"lang_{lang}"] = (df["original_language"] == lang).astype(int)
    df["lang_other"] = (~df["original_language"].isin(TOP_LANGS)).astype(int)

    df["runtime"] = df["runtime"].fillna(df["runtime"].median())
    df["is_collection"] = df["is_collection"].fillna(0).astype(int)
    df["log_budget"] = np.log1p(df["budget"])
    df["release_year"] = df["release_year"].fillna(df["release_year"].median())

    genre_cols = [f'genre_{g.replace(" ", "_")}' for g in TOP_GENRES]
    lang_cols = [f"lang_{l}" for l in TOP_LANGS] + ["lang_other"]
    num_cols = ["log_budget", "runtime", "is_collection", "release_year"]
    FEATURES = num_cols + genre_cols + lang_cols

    X = df[FEATURES].copy()
    y = df["profitable"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train_sc, y_train)
    y_pred_lr = lr.predict(X_test_sc)
    y_prob_lr = lr.predict_proba(X_test_sc)[:, 1]
    acc_lr = accuracy_score(y_test, y_pred_lr)
    auc_lr = roc_auc_score(y_test, y_prob_lr)
    cv_lr_all = cross_val_score(lr, X_train_sc, y_train, cv=5, scoring="roc_auc")

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    acc_rf = accuracy_score(y_test, y_pred_rf)
    auc_rf = roc_auc_score(y_test, y_prob_rf)
    cv_rf_all = cross_val_score(rf, X_train, y_train, cv=5, scoring="roc_auc")

    return dict(
        acc_lr=acc_lr, auc_lr=auc_lr, cv_lr_all=cv_lr_all,
        acc_rf=acc_rf, auc_rf=auc_rf, cv_rf_all=cv_rf_all,
        y_test=y_test,
        y_pred_lr=y_pred_lr, y_prob_lr=y_prob_lr,
        y_pred_rf=y_pred_rf, y_prob_rf=y_prob_rf,
        rf=rf, FEATURES=FEATURES,
    )


def _fig1_comparison_roc(m: dict) -> plt.Figure:
    """Figure 1: Model Performance Comparison + ROC Curves."""
    sns.set_theme(style="whitegrid", context="talk")
    fig, (ax_bar, ax_roc) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Model Performance Comparison & ROC Curves",
        fontsize=15, fontweight="bold", y=1.02,
    )

    # ── Panel A: bar chart ──
    bar_data = pd.DataFrame({
        "Model":  ["LR", "RF", "LR", "RF"],
        "Metric": ["Accuracy", "Accuracy", "ROC-AUC", "ROC-AUC"],
        "Score":  [m["acc_lr"], m["acc_rf"], m["auc_lr"], m["auc_rf"]],
    })
    sns.barplot(
        data=bar_data, x="Model", y="Score", hue="Metric",
        palette={"Accuracy": BLUE, "ROC-AUC": GREEN}, ax=ax_bar,
    )
    ax_bar.set_ylim(0.5, 1.0)
    ax_bar.set_title("Model Performance Comparison", fontweight="bold", pad=10)
    ax_bar.set_ylabel("Score")
    ax_bar.set_xlabel("")
    for p in ax_bar.patches:
        h = p.get_height()
        if h > 0.5:
            ax_bar.annotate(
                f"{h:.3f}",
                (p.get_x() + p.get_width() / 2, h),
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )
    ax_bar.legend(loc="lower right", fontsize=10)

    # ── Panel B: ROC curves ──
    fpr_lr, tpr_lr, _ = roc_curve(m["y_test"], m["y_prob_lr"])
    fpr_rf, tpr_rf, _ = roc_curve(m["y_test"], m["y_prob_rf"])
    ax_roc.plot(fpr_lr, tpr_lr, color=BLUE, lw=2.5,
                label=f'Logistic Regression (AUC = {m["auc_lr"]:.3f})')
    ax_roc.plot(fpr_rf, tpr_rf, color=GREEN, lw=2.5,
                label=f'Random Forest      (AUC = {m["auc_rf"]:.3f})')
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Baseline")
    ax_roc.fill_between(fpr_lr, tpr_lr, alpha=0.08, color=BLUE)
    ax_roc.fill_between(fpr_rf, tpr_rf, alpha=0.08, color=GREEN)
    ax_roc.set_title("ROC Curves", fontweight="bold", pad=10)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right", fontsize=10)

    fig.tight_layout()
    return fig


def _fig2_cv_cm_lr(m: dict) -> plt.Figure:
    """Figure 2: 5-Fold Cross-Validation + Confusion Matrix (LR)."""
    sns.set_theme(style="whitegrid", context="talk")
    fig, (ax_cv, ax_cm_lr) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Cross-Validation & Logistic Regression Confusion Matrix",
        fontsize=15, fontweight="bold", y=1.02,
    )

    # ── Panel C: CV boxplot ──
    cv_df = pd.DataFrame({
        "Fold":  list(range(1, 6)) * 2,
        "AUC":   list(m["cv_lr_all"]) + list(m["cv_rf_all"]),
        "Model": ["Logistic Regression"] * 5 + ["Random Forest"] * 5,
    })
    sns.boxplot(data=cv_df, x="Model", y="AUC",
                palette=[BLUE, GREEN], ax=ax_cv, width=0.4)
    sns.stripplot(data=cv_df, x="Model", y="AUC",
                  palette=[BLUE, GREEN], ax=ax_cv, size=8, jitter=False)
    ax_cv.set_title("5-Fold Cross-Validation (ROC-AUC)", fontweight="bold", pad=10)
    ax_cv.set_xlabel("")
    ax_cv.set_ylabel("ROC-AUC")
    ax_cv.set_ylim(0.55, 0.95)

    # ── Panel D: Confusion matrix LR ──
    cm_lr = confusion_matrix(m["y_test"], m["y_pred_lr"])
    ConfusionMatrixDisplay(
        cm_lr, display_labels=["Not Profitable", "Profitable"]
    ).plot(ax=ax_cm_lr, colorbar=False, cmap="Blues")
    ax_cm_lr.set_title(
        "Confusion Matrix\nLogistic Regression", fontweight="bold", pad=10
    )

    fig.tight_layout()
    return fig


def _fig3_cm_rf_fi(m: dict) -> plt.Figure:
    """Figure 3: Confusion Matrix (RF) + Feature Importances."""
    sns.set_theme(style="whitegrid", context="talk")
    fig, (ax_cm_rf, ax_fi) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Random Forest Confusion Matrix & Feature Importances",
        fontsize=15, fontweight="bold", y=1.02,
    )

    # ── Panel E: Confusion matrix RF ──
    cm_rf = confusion_matrix(m["y_test"], m["y_pred_rf"])
    ConfusionMatrixDisplay(
        cm_rf, display_labels=["Not Profitable", "Profitable"]
    ).plot(ax=ax_cm_rf, colorbar=False, cmap="Greens")
    ax_cm_rf.set_title(
        "Confusion Matrix\nRandom Forest", fontweight="bold", pad=10
    )

    # ── Panel F: Feature importances ──
    fi = (
        pd.Series(m["rf"].feature_importances_, index=m["FEATURES"])
        .sort_values(ascending=False)
        .head(15)
    )
    clean_labels = (
        fi.index
        .str.replace("genre_", "", regex=False)
        .str.replace("lang_", "Lang:", regex=False)
        .str.replace("_", " ", regex=False)
    )
    colors_fi = [
        RED if "log_budget" in i
        else ORANGE if "genre" in i
        else PURPLE
        for i in fi.index
    ]
    ax_fi.barh(range(len(fi)), fi.values[::-1],
               color=colors_fi[::-1], edgecolor="white")
    ax_fi.set_yticks(range(len(fi)))
    ax_fi.set_yticklabels(clean_labels[::-1], fontsize=11)
    ax_fi.set_title(
        "Top 15 Feature Importances\n(Random Forest)", fontweight="bold", pad=10
    )
    ax_fi.set_xlabel("Importance Score")
    ax_fi.legend(
        handles=[
            Patch(facecolor=RED,    label="Numeric"),
            Patch(facecolor=ORANGE, label="Genre"),
            Patch(facecolor=PURPLE, label="Language"),
        ],
        loc="lower right", fontsize=9,
    )

    fig.tight_layout()
    return fig


def render() -> None:
    st.header("Machine Learning Models")
    st.caption("This page presents the results of machine learning models predicting movie profitability.")

    st.markdown(
        """
        ### Profitability Prediction Model

        A classification model (Logistic Regression vs Random Forest) is trained to
        predict whether a movie will be profitable (revenue > budget).
        Features include budget, runtime, release year, genres, and language.
        *Post-release signals (vote_count, popularity) are excluded to prevent data leakage.*
        """
    )

    # ── Core Insights ─────────────────────────────────────────
    st.subheader("Core Insights")
    st.success(
        "The profitability prediction model demonstrates that machine learning techniques "
        "can effectively identify profitability patterns using structured movie metadata. "
        "Random Forest achieves the strongest overall predictive performance due to its "
        "ability to capture **complex non-linear relationships** among movie features, "
        "providing practical insights into the major factors influencing movie profitability."
    )
    col_a, col_b, col_c = st.columns(3)
    col_a.success(
        "**Random Forest Leads**\n\n"
        "Random Forest achieves **Accuracy 0.757** and **ROC-AUC 0.852**, outperforming "
        "Logistic Regression across all metrics due to its stronger ability to capture "
        "non-linear relationships."
    )
    col_b.warning(
        "**Features Beyond Budget**\n\n"
        "Release year, log-budget, and runtime are the strongest predictors. "
        "Genre and language also contribute, suggesting that **production strategy and "
        "market accessibility** matter more than spending alone."
    )
    col_c.error(
        "**Profitability is Complex**\n\n"
        "Both models achieve ROC-AUC **> 0.84**, yet profitability cannot be explained by "
        "a single variable. It emerges from the **combined effects** of genre, timing, "
        "strategy, and market dynamics."
    )

    st.divider()

    m = _train_models()

    # ── Figure 1 ──────────────────────────────────────────────
    st.subheader("1. Model Performance Comparison & ROC Curves")
    fig1 = _fig1_comparison_roc(m)
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    with st.expander("Visuals Explanation", expanded=True):
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(
                "**Left Chart — Model Accuracy Comparison (Bar Chart)**\n\n"
                "The chart compares the prediction performance of Logistic Regression and "
                "Random Forest.\n\n"
                "| Metric | Logistic Regression | Random Forest |\n"
                "|--------|-------------------|---------------|\n"
                "| Accuracy | 0.725 | 0.757 |\n"
                "| ROC-AUC | 0.846 | 0.852 |\n"
                "| CV ROC-AUC | ~0.83 | ~0.84 |\n\n"
                "The results show that Random Forest slightly outperforms Logistic Regression "
                "across most evaluation metrics."
            )
        with col_r:
            st.markdown(
                "**Right Chart — ROC Curve Comparison**\n\n"
                "The ROC curves visualize the classification capability of both machine "
                "learning models.\n\n"
                "Random Forest achieves the highest ROC-AUC score (0.852), indicating "
                "stronger discrimination capability between profitable and non-profitable movies.\n\n"
                "Although the performance improvement is moderate, Random Forest demonstrates "
                "stronger capability in capturing non-linear relationships among movie attributes."
            )

    st.divider()

    # ── Figure 2 ──────────────────────────────────────────────
    st.subheader("2. Cross-Validation & Logistic Regression Confusion Matrix")
    fig2 = _fig2_cv_cm_lr(m)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    with st.expander("Visuals Explanation", expanded=True):
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(
                "**Left Chart — Cross Validation ROC-AUC (Boxplot)**\n\n"
                "The boxplot compares cross-validation performance stability between models.\n\n"
                "Both models maintain relatively stable ROC-AUC scores across validation folds, "
                "indicating good generalization performance and limited overfitting."
            )
        with col_r:
            st.markdown(
                "**Right Chart — Logistic Regression Confusion Matrix**\n\n"
                "The confusion matrix visualizes prediction distribution for profitable and "
                "non-profitable movies using Logistic Regression.\n\n"
                "The model demonstrates acceptable classification capability but shows weaker "
                "performance when identifying more complex profitability patterns."
            )

    st.divider()

    # ── Figure 3 ──────────────────────────────────────────────
    st.subheader("3. Random Forest Confusion Matrix & Feature Importances")
    fig3 = _fig3_cm_rf_fi(m)
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    with st.expander("Visuals Explanation", expanded=True):
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(
                "**Left Chart — Random Forest Confusion Matrix**\n\n"
                "The confusion matrix demonstrates the stronger classification capability of "
                "Random Forest.\n\n"
                "Compared with Logistic Regression, Random Forest produces more balanced "
                "predictions and reduces classification error for profitable movies."
            )
        with col_r:
            st.markdown(
                "**Right Chart — Feature Importance (Random Forest)**\n\n"
                "The chart visualizes the most influential variables contributing to "
                "profitability prediction.\n\n"
                "Feature importance analysis reveals that **release year**, **budget** (log), "
                "and **runtime** are among the strongest predictors of profitability. "
                "Genre and language features also contribute meaningful signals, suggesting "
                "that audience engagement and market accessibility play important roles beyond "
                "production spending alone."
            )

    st.divider()

    st.info(
        "The complete training script can be found in `regression_model/profitability_model.py`."
    )
