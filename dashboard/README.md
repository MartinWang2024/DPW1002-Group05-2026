# Dashboard Quick Start

## 1) Install dependencies

From project root:

```powershell
pip install -r dashboard/requirements-dashboard.txt
```

## 2) Run dashboard

```powershell
streamlit run dashboard/app.py
```

## 3) Open in browser

Streamlit will print a local URL, usually:

- http://localhost:8501

```
dashboard/
├── app.py                ← 7 个 Tab 的主框架
├── data_loader.py        ← 新增 load_box_office_full / load_credits_data / load_keywords_data
└── tabs/
    ├── tab_index.py      → 数据概览（预算/票房/评分/国家）
    ├── tab_country.py    → 国家分析（Data_A&V_country.py）
    ├── tab_people.py     → 人物分析（Data_A&V_peo.py）
    ├── tab_rating.py     → 评分分析（main.py + main_visulization.py）
    ├── tab_correlation.py → 相关性分析（pearson_and_spearman*.py）
    ├── tab_regression.py → 回归模型（movie_analysis_fixed.py）
    └── tab_buget.py      → Budget 预算（budget_analysis.ipynb）
```