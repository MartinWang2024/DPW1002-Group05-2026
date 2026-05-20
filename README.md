# Movie Data Analytics Platform

DPW1002 Group Project (Group 05, 2026) — A comprehensive movie data analysis platform built on the TMDB MovieLens dataset, featuring data cleaning, statistical analysis, machine learning, and an interactive Streamlit dashboard.

## Tech Stack

- **Python 3.14**
- **Data processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Dashboard**: Streamlit
- **Machine Learning**: scikit-learn (LogisticRegression, RandomForestClassifier)
- **Notebooks**: Jupyter

## Dataset

This project uses the [TMDB MovieLens dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset), which contains metadata for 45,000 movies released on or before July 2017, including:

- **movies_metadata.csv** — budget, revenue, genres, release dates, languages, countries
- **credits.csv** — cast and crew information
- **keywords.csv** — plot keywords
- **ratings_small.csv** — 100,000 ratings from 700 users on 9,000 movies
- **links.csv / links_small.csv** — TMDB and IMDB ID mappings

See [Dataset.md](Dataset.md) for full dataset documentation.

## Project Structure

```
├── archive/                  # Raw CSV data files
├── cleaned_archive/          # Data cleaning scripts & cleaned datasets
│   ├── clean_movies_metadata.py
│   ├── clean_credits.py
│   ├── clean_keywords.py
│   ├── clean_ratings_small.py
│   ├── clean_links_small.py
│   └── *.csv                 # Cleaned output CSVs
├── analysis/                 # Statistical analysis & EDA
│   ├── Data_A&V_country.py           # Country-level analysis
│   ├── Data_A&V_peo.py               # Actor & director analysis
│   ├── budget_analysis.ipynb         # Budget & revenue deep dive
│   ├── movie_analysis.ipynb          # General EDA
│   ├── pearson_and_spearman.py       # Correlation coefficients
│   ├── pearson_and_spearman_visulization.py
│   ├── rating.py                     # Rating analysis
│   └── rating_visulization.py
├── cross_analysis/           # Cross-dimensional analysis
│   └── cross_analysis.py     # Genre × Budget Tier × Language
├── regression_model/         # Machine learning models
│   └── profitability_model.py  # LR + RF profitability prediction
├── dashboard/                # Streamlit interactive dashboard
│   ├── app.py                # Main entry point (8 tabs)
│   ├── data_loader.py        # Shared data loading utilities
│   ├── requirements-dashboard.txt
│   └── tabs/
│       ├── tab_index.py      # Data overview (budget/revenue/rating/country)
│       ├── tab_country.py    # Country analysis
│       ├── tab_date.py       # Release date analysis
│       ├── tab_keywords.py   # Keyword analysis
│       ├── tab_people.py     # Actors & directors analysis
│       ├── tab_rating.py     # Rating analysis & correlations
│       ├── tab_budget.py     # Budget trend analysis
│       └── tab_ml.py         # ML model results (LR vs RF)
└── outputs/                  # Generated visualizations (PNG)
```

## Features

### Data Cleaning

Scripts that parse JSON-like string fields, handle missing values, normalize formats, and produce analysis-ready CSVs.

### Statistical Analysis

- **\**
- **Rating analysis**: Distribution patterns, vote count vs. vote average relationships
- **Country analysis**: Geographic distribution of movie production
- **People analysis**: Actor/director impact on movie performance
- **Budget analysis**: Trends, ROI profiling, and profitability patterns
- **Release & keywords analysis**: Temporal release patterns and plot keyword impact

### Cross Analysis

Multi-dimensional analysis examining profitability across combinations of genre, budget tier, and language group, with heatmap visualization.

### Machine Learning

Trains Logistic Regression and Random Forest classifiers to predict movie profitability (revenue > budget) using pre-release features:

- **Features**: log-budget, runtime, release year, genres, language, collection membership
- **Avoids data leakage**: Post-release signals (vote_count, popularity) are excluded
- **Key results**: Random Forest achieves ROC-AUC ~0.85

### Interactive Dashboard

An 8-tab Streamlit app providing interactive charts and filters:

| Tab                   | Description                                            |
| --------------------- | ------------------------------------------------------ |
| Data Overview         | Budget/revenue/rating distributions, country breakdown |
| Country Analysis      | Geographic production patterns                         |
| Release Date Analysis | Temporal trends in movie releases                      |
| Keyword Analysis      | Plot keyword frequency & impact                        |
| Actors & Directors    | Cast/crew influence on performance                     |
| Rating Analysis       | Correlations between ratings and revenue               |
| Budget Analysis       | Budget trends, ROI, and profitability                  |
| Machine Learning      | LR vs RF model comparison, feature importance          |

## Quick Start

### 1. Clone and set up environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 2. Install dependencies

```bash
pip install -r dashboard/requirements-dashboard.txt
```

Additional dependencies needed for analysis and ML scripts:

```bash
pip install matplotlib seaborn scikit-learn scipy jupyter
```

### 3. Run the dashboard

```bash
streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser.

### 4. Run analysis scripts

```bash
# Correlation analysis
python analysis/pearson_and_spearman.py

# Cross analysis
python cross_analysis/cross_analysis.py

# Machine learning model
python regression_model/profitability_model.py
```

## Team

| Member | Contribution |
|--------|-------------|
| 1.     |             |
| 2.     |             |
| 3.     |             |
| 4.     |             |
| 5.     |             |
| 6.     |             |

DPW1002 Group 05, 2026
