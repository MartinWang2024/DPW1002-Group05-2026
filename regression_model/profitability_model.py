import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os

from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score,
    ConfusionMatrixDisplay
)

# 1. DATA LOADING & FEATURE ENGINEERING
print("STEP 1: Loading Data...")

df = pd.read_csv(
    r'cleaned_archive/movies_metadata_cleaned.csv',
    low_memory=False
)

df_ratings = pd.read_csv(
    r'cleaned_archive/ratings_small_cleaned.csv',
    low_memory=False
)

# Convert numeric columns
numeric_cols = ['budget', 'revenue', 'runtime', 'release_year']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Keep only movies with valid budget & revenue
df = df[(df['budget'] > 0) & (df['revenue'] > 0)].copy()
print(f"Valid records (budget + revenue both > 0): {len(df)}")

# ── Target variable: profitable if revenue > budget ──────────
df['profitable'] = (df['revenue'] > df['budget']).astype(int)
profit_rate = df['profitable'].mean()
print(f"Profitable movies: {df['profitable'].sum()} ({profit_rate:.1%})")
print(f"Non-profitable:    {(df['profitable']==0).sum()} ({1-profit_rate:.1%})")

# ── Parse genres list ─────────────────────────────────────────
def parse_list(s):
    try:
        return ast.literal_eval(s)
    except:
        return []

df['genres_parsed'] = df['genres_list'].apply(parse_list)

TOP_GENRES = [
    'Drama', 'Comedy', 'Thriller', 'Action', 'Romance',
    'Adventure', 'Crime', 'Science Fiction', 'Horror',
    'Family', 'Fantasy', 'Animation', 'Mystery', 'History', 'War'
]

for genre in TOP_GENRES:
    df[f'genre_{genre.replace(" ", "_")}'] = df['genres_parsed'].apply(
        lambda g: 1 if genre in g else 0
    )

# ── Language encoding ─────────────────────────────────────────
TOP_LANGS = ['en', 'hi', 'fr', 'ru', 'es', 'ja', 'zh', 'it', 'ta', 'ko', 'de']
for lang in TOP_LANGS:
    df[f'lang_{lang}'] = (df['original_language'] == lang).astype(int)

df['lang_other'] = (~df['original_language'].isin(TOP_LANGS)).astype(int)

# ── Other numeric features ────────────────────────────────────
df['runtime'] = df['runtime'].fillna(df['runtime'].median())

# NOTE: vote_count and popularity are POST-RELEASE data
# removed to prevent data leakage
df['is_collection'] = df['is_collection'].fillna(0).astype(int)

# Log transform budget (handles skew)
df['log_budget'] = np.log1p(df['budget'])
df['release_decade'] = (df['release_year'] // 10) * 10

# ── Feature matrix ────────────────────────────────────────────
# vote_count and popularity EXCLUDED: both are formed after release
# and would cause data leakage (AUC inflated by ~0.11)
genre_cols  = [f'genre_{g.replace(" ", "_")}' for g in TOP_GENRES]
lang_cols   = [f'lang_{l}' for l in TOP_LANGS] + ['lang_other']
num_cols    = ['log_budget', 'runtime', 'is_collection', 'release_year']

FEATURES = num_cols + genre_cols + lang_cols
X = df[FEATURES].copy()
y = df['profitable']

print(f"\nFeature matrix shape: {X.shape}")
print(f"  - Numeric features:  {len(num_cols)}  (vote_count & popularity removed — post-release leakage)")
print(f"  - Genre features:    {len(genre_cols)}")
print(f"  - Language features: {len(lang_cols)}")

# 2. TRAIN / TEST SPLIT
print("\nSTEP 2: Train/Test Split (80/20, stratified)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set:  {X_train.shape[0]} samples")
print(f"Test set:      {X_test.shape[0]} samples")

# Scale numeric features for Logistic Regression
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# 3. LOGISTIC REGRESSION
print("\nSTEP 3: Logistic Regression")

lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)

lr.fit(X_train_sc, y_train)

y_pred_lr  = lr.predict(X_test_sc)
y_prob_lr  = lr.predict_proba(X_test_sc)[:, 1]

acc_lr  = accuracy_score(y_test, y_pred_lr)
auc_lr  = roc_auc_score(y_test, y_prob_lr)

cv_lr   = cross_val_score(
    lr,
    X_train_sc,
    y_train,
    cv=5,
    scoring='roc_auc'
).mean()

print(f"  Accuracy:   {acc_lr:.4f}")
print(f"  ROC-AUC:    {auc_lr:.4f}")
print(f"  CV AUC (5-fold): {cv_lr:.4f}")

print("\nClassification Report (Logistic Regression):")

print(classification_report(
    y_test,
    y_pred_lr,
    target_names=['Not Profitable', 'Profitable']
))

# 4. RANDOM FOREST
print("\nSTEP 4: Random Forest")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred_rf  = rf.predict(X_test)
y_prob_rf  = rf.predict_proba(X_test)[:, 1]

acc_rf  = accuracy_score(y_test, y_pred_rf)
auc_rf  = roc_auc_score(y_test, y_prob_rf)

cv_rf   = cross_val_score(
    rf,
    X_train,
    y_train,
    cv=5,
    scoring='roc_auc'
).mean()

print(f"  Accuracy:   {acc_rf:.4f}")
print(f"  ROC-AUC:    {auc_rf:.4f}")
print(f"  CV AUC (5-fold): {cv_rf:.4f}")

print("\nClassification Report (Random Forest):")

print(classification_report(
    y_test,
    y_pred_rf,
    target_names=['Not Profitable', 'Profitable']
))

# 5. VISUALIZATION (4-panel figure)
print("\nSTEP 5: Generating Visualizations...")

sns.set_theme(style='whitegrid', context='talk')

BLUE   = '#4C72B0'
GREEN  = '#55A868'
RED    = '#C44E52'
ORANGE = '#DD8452'

fig = plt.figure(figsize=(20, 16))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# ─── Panel A: Model Comparison Bar Chart ─────────────────────
ax_bar = fig.add_subplot(gs[0, 0])

metrics_df = pd.DataFrame({
    'Model': ['Logistic\nRegression', 'Random\nForest',
              'Logistic\nRegression', 'Random\nForest'],
    'Metric': ['Accuracy', 'Accuracy', 'ROC-AUC', 'ROC-AUC'],
    'Score':  [acc_lr, acc_rf, auc_lr, auc_rf]
})

palette = {
    'Accuracy': BLUE,
    'ROC-AUC': GREEN
}

bar_data = pd.DataFrame({
    'Model':    ['LR', 'RF', 'LR', 'RF'],
    'Metric':   ['Accuracy', 'Accuracy', 'ROC-AUC', 'ROC-AUC'],
    'Score':    [acc_lr, acc_rf, auc_lr, auc_rf]
})

bars = sns.barplot(
    data=bar_data,
    x='Model',
    y='Score',
    hue='Metric',
    palette=palette,
    ax=ax_bar
)

ax_bar.set_ylim(0.5, 1.0)
ax_bar.set_title('Model Performance Comparison', fontweight='bold', pad=10)
ax_bar.set_ylabel('Score')
ax_bar.set_xlabel('')

for p in ax_bar.patches:
    h = p.get_height()

    if h > 0.5:
        ax_bar.annotate(
            f'{h:.3f}',
            (p.get_x() + p.get_width()/2, h),
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )

ax_bar.legend(loc='lower right', fontsize=10)

# ─── Panel B: ROC Curves ─────────────────────────────────────
ax_roc = fig.add_subplot(gs[0, 1])

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

ax_roc.plot(
    fpr_lr,
    tpr_lr,
    color=BLUE,
    lw=2.5,
    label=f'Logistic Regression (AUC = {auc_lr:.3f})'
)

ax_roc.plot(
    fpr_rf,
    tpr_rf,
    color=GREEN,
    lw=2.5,
    label=f'Random Forest      (AUC = {auc_rf:.3f})'
)

ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random Baseline')

ax_roc.fill_between(fpr_lr, tpr_lr, alpha=0.08, color=BLUE)
ax_roc.fill_between(fpr_rf, tpr_rf, alpha=0.08, color=GREEN)

ax_roc.set_title('ROC Curves', fontweight='bold', pad=10)
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')

ax_roc.legend(loc='lower right', fontsize=10)

# ─── Panel C: CV Scores (5-fold) ─────────────────────────────
ax_cv = fig.add_subplot(gs[0, 2])

cv_lr_all = cross_val_score(
    lr,
    X_train_sc,
    y_train,
    cv=5,
    scoring='roc_auc'
)

cv_rf_all = cross_val_score(
    rf,
    X_train,
    y_train,
    cv=5,
    scoring='roc_auc'
)

cv_df = pd.DataFrame({
    'Fold': list(range(1, 6)) * 2,
    'AUC':  list(cv_lr_all) + list(cv_rf_all),
    'Model': ['Logistic Regression'] * 5 + ['Random Forest'] * 5
})

sns.boxplot(
    data=cv_df,
    x='Model',
    y='AUC',
    palette=[BLUE, GREEN],
    ax=ax_cv,
    width=0.4
)

sns.stripplot(
    data=cv_df,
    x='Model',
    y='AUC',
    palette=[BLUE, GREEN],
    ax=ax_cv,
    size=8,
    jitter=False
)

ax_cv.set_title('5-Fold Cross-Validation (ROC-AUC)', fontweight='bold', pad=10)
ax_cv.set_xlabel('')
ax_cv.set_ylabel('ROC-AUC')
ax_cv.set_ylim(0.55, 0.95)

# ─── Panel D: Confusion Matrix LR ────────────────────────────
ax_cm_lr = fig.add_subplot(gs[1, 0])

cm_lr = confusion_matrix(y_test, y_pred_lr)

ConfusionMatrixDisplay(
    cm_lr,
    display_labels=['Not Profitable', 'Profitable']
).plot(
    ax=ax_cm_lr,
    colorbar=False,
    cmap='Blues'
)

ax_cm_lr.set_title(
    'Confusion Matrix\nLogistic Regression',
    fontweight='bold',
    pad=10
)

# ─── Panel E: Confusion Matrix RF ────────────────────────────
ax_cm_rf = fig.add_subplot(gs[1, 1])

cm_rf = confusion_matrix(y_test, y_pred_rf)

ConfusionMatrixDisplay(
    cm_rf,
    display_labels=['Not Profitable', 'Profitable']
).plot(
    ax=ax_cm_rf,
    colorbar=False,
    cmap='Greens'
)

ax_cm_rf.set_title(
    'Confusion Matrix\nRandom Forest',
    fontweight='bold',
    pad=10
)

# ─── Panel F: Feature Importance (RF) ────────────────────────
ax_fi = fig.add_subplot(gs[1, 2])

fi = pd.Series(
    rf.feature_importances_,
    index=FEATURES
).sort_values(ascending=False).head(15)

# Clean labels
clean_labels = (
    fi.index
    .str.replace('genre_', '', regex=False)
    .str.replace('lang_', 'Lang:', regex=False)
    .str.replace('_', ' ', regex=False)
)

colors_fi = [
    RED if 'log_budget' in i
    else ORANGE if 'genre' in i
    else '#8172B2'
    for i in fi.index
]

ax_fi.barh(
    range(len(fi)),
    fi.values[::-1],
    color=colors_fi[::-1],
    edgecolor='white'
)

ax_fi.set_yticks(range(len(fi)))
ax_fi.set_yticklabels(clean_labels[::-1], fontsize=11)

ax_fi.set_title(
    'Top 15 Feature Importances\n(Random Forest)',
    fontweight='bold',
    pad=10
)

ax_fi.set_xlabel('Importance Score')

# Legend patches
from matplotlib.patches import Patch

legend_els = [
    Patch(facecolor=RED,    label='Numeric'),
    Patch(facecolor=ORANGE, label='Genre'),
    Patch(facecolor='#8172B2', label='Language'),
]

ax_fi.legend(handles=legend_els, loc='lower right', fontsize=9)

# ─── Super title ─────────────────────────────────────────────
fig.suptitle(
    'Movie Profitability Prediction: Logistic Regression vs Random Forest\n'
    '(Leak-Free: vote_count & popularity removed)',
    fontsize=17,
    fontweight='bold',
    y=1.02
)


os.makedirs('outputs', exist_ok=True)

plt.savefig(
    'outputs/profitability_model_results.png',
    dpi=180,
    bbox_inches='tight',
    facecolor='white'
)

plt.show()

print("Visualization saved.")

# 6. SUMMARY
print("\nSUMMARY")

print(f"{'Metric':<25} {'Logistic Regression':>20} {'Random Forest':>15}")

print("-" * 62)

print(f"{'Accuracy':<25} {acc_lr:>20.4f} {acc_rf:>15.4f}")
print(f"{'ROC-AUC':<25} {auc_lr:>20.4f} {auc_rf:>15.4f}")
print(f"{'CV ROC-AUC (5-fold)':<25} {cv_lr:>20.4f} {cv_rf:>15.4f}")

print("-" * 62)

winner = "Random Forest" if auc_rf > auc_lr else "Logistic Regression"

print(f"\nBest model by ROC-AUC: {winner}")

print("\nTop 5 most predictive features (Random Forest):")

top5_features = pd.Series(
    rf.feature_importances_,
    index=FEATURES
).sort_values(ascending=False).head(5)

for feat, imp in top5_features.items():
    clean = feat.replace('genre_', 'Genre: ').replace('lang_', 'Language: ').replace('_', ' ')

    print(f"  {clean:<30} {imp:.4f}")