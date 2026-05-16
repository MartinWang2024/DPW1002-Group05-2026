"""
交叉分析：类型 × 预算 × 语言
三步：
  Step 1 - 类型 × 盈利率 & 中位ROI
  Step 2 - 预算分层 × 类型 盈利率热力图
  Step 3 - 语言组 × 类型 盈利率对比
"""

import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── 数据准备 ────────────────────────────────────────────────
df = pd.read_csv(
    '/home/claude/dpw_project/DPW_project_dataAnalsis_rating_and_revenue/movies_metadata_cleaned.csv',
    low_memory=False)
df = df[(df['budget'] > 0) & (df['revenue'] > 0)].copy()
df['profitable'] = (df['revenue'] > df['budget']).astype(int)
df['roi'] = (df['revenue'] - df['budget']) / df['budget']

def parse_list(s):
    try: return ast.literal_eval(s)
    except: return []

df['genres_parsed'] = df['genres_list'].apply(parse_list)

TOP_GENRES = ['Animation','Family','Adventure','Science Fiction','Action',
              'Fantasy','Comedy','Thriller','Horror','Crime','Romance','Drama']

df['budget_tier'] = pd.cut(df['budget'],
    bins=[0, 5e6, 20e6, 60e6, 400e6],
    labels=['Low (<$5M)', 'Mid ($5-20M)', 'High ($20-60M)', 'Blockbuster (>$60M)'])

df['lang_group'] = df['original_language'].apply(
    lambda x: 'English' if x == 'en'
    else ('Asian' if x in ['hi','ja','zh','ko','ta']
    else ('European' if x in ['fr','es','it','de','ru'] else 'Other')))

# 展开类型行
rows = []
for _, row in df.iterrows():
    for g in row['genres_parsed']:
        if g in TOP_GENRES:
            rows.append({'genre': g, 'budget_tier': row['budget_tier'],
                         'lang_group': row['lang_group'],
                         'profitable': row['profitable'],
                         'roi': row['roi'],
                         'budget': row['budget']})
gdf = pd.DataFrame(rows)

# ── 配色系统 ─────────────────────────────────────────────────
PALETTE   = '#4C72B0'
GREEN     = '#55A868'
RED       = '#C44E52'
AMBER     = '#DD8452'
BG        = 'white'

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': BG,
                     'font.family': 'sans-serif'})

fig = plt.figure(figsize=(22, 20))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.38)

# ════════════════════════════════════════════════════════════
# STEP 1A — 各类型盈利率 (按盈利率降序)
# ════════════════════════════════════════════════════════════
ax1a = fig.add_subplot(gs[0, 0])

genre_profit = (gdf.groupby('genre')['profitable']
                .agg(['mean', 'count'])
                .reset_index()
                .rename(columns={'mean':'profit_rate','count':'n'}))
genre_profit = genre_profit[genre_profit['n'] >= 30].sort_values('profit_rate', ascending=True)

colors_bar = [GREEN if r >= 0.70 else PALETTE if r >= 0.55 else RED
              for r in genre_profit['profit_rate']]

bars = ax1a.barh(genre_profit['genre'], genre_profit['profit_rate'],
                 color=colors_bar, edgecolor='white', height=0.65)
ax1a.axvline(x=0.699, color='gray', linestyle='--', linewidth=1.2, alpha=0.6, label='Overall avg (69.9%)')
ax1a.set_xlim(0.4, 1.0)
ax1a.set_xlabel('Profitability Rate')
ax1a.set_title('Step 1A  |  Profitability Rate by Genre', fontweight='bold', pad=10)
for bar, val in zip(bars, genre_profit['profit_rate']):
    ax1a.text(val + 0.008, bar.get_y() + bar.get_height()/2,
              f'{val:.1%}', va='center', fontsize=11)
ax1a.legend(fontsize=10)

# ════════════════════════════════════════════════════════════
# STEP 1B — 各类型中位ROI（截断到500%）
# ════════════════════════════════════════════════════════════
ax1b = fig.add_subplot(gs[0, 1])

genre_roi = (gdf.groupby('genre')['roi']
             .median()
             .reset_index()
             .rename(columns={'roi':'median_roi'}))
# 按 Step1A 的盈利率顺序排列
genre_roi = genre_roi.set_index('genre').loc[genre_profit['genre']].reset_index()
genre_roi['median_roi_pct'] = genre_roi['median_roi'] * 100

colors_roi = [GREEN if r >= 100 else PALETTE if r >= 50 else AMBER
              for r in genre_roi['median_roi_pct']]

bars2 = ax1b.barh(genre_roi['genre'], genre_roi['median_roi_pct'].clip(upper=500),
                  color=colors_roi, edgecolor='white', height=0.65)
ax1b.set_xlabel('Median ROI (%)')
ax1b.set_title('Step 1B  |  Median ROI by Genre', fontweight='bold', pad=10)
for bar, val in zip(bars2, genre_roi['median_roi_pct']):
    ax1b.text(min(val, 500) + 5, bar.get_y() + bar.get_height()/2,
              f'{val:.0f}%', va='center', fontsize=11)

# ════════════════════════════════════════════════════════════
# STEP 2 — 预算分层 × 类型 盈利率热力图
# ════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[1, :])

pivot = (gdf.groupby(['genre','budget_tier'])['profitable']
         .agg(['mean','count'])
         .reset_index())
pivot.columns = ['genre','budget_tier','profit_rate','n']
pivot = pivot[pivot['n'] >= 10]

hmap = pivot.pivot(index='genre', columns='budget_tier', values='profit_rate')
hmap = hmap.loc[genre_profit['genre'][::-1]]   # 同Step1顺序
tier_order = ['Low (<$5M)','Mid ($5-20M)','High ($20-60M)','Blockbuster (>$60M)']
hmap = hmap.reindex(columns=tier_order)

# annotation: rate + sample count
annot_df = pivot.pivot(index='genre', columns='budget_tier', values='profit_rate')
count_df  = pivot.pivot(index='genre', columns='budget_tier', values='n')
annot_df  = annot_df.loc[hmap.index, tier_order]
count_df  = count_df.loc[hmap.index, tier_order]
annot = annot_df.map(lambda v: f'{v:.0%}' if (not isinstance(v, float) or not np.isnan(v)) else 'N/A')

sns.heatmap(hmap, ax=ax2, cmap='RdYlGn', vmin=0.3, vmax=1.0,
            annot=annot, fmt='', annot_kws={'size':13, 'weight':'bold'},
            linewidths=0.5, linecolor='white', cbar_kws={'label':'Profitability Rate'})
ax2.set_title('Step 2  |  Profitability Rate Heatmap: Genre × Budget Tier',
              fontweight='bold', pad=12, fontsize=14)
ax2.set_xlabel('Budget Tier', fontsize=12)
ax2.set_ylabel('')
ax2.tick_params(axis='x', rotation=0, labelsize=11)
ax2.tick_params(axis='y', labelsize=11)

# ════════════════════════════════════════════════════════════
# STEP 3A — 语言组 × 盈利率总览
# ════════════════════════════════════════════════════════════
ax3a = fig.add_subplot(gs[2, 0])

lang_profit = (gdf.groupby('lang_group')['profitable']
               .agg(['mean','count'])
               .reset_index()
               .rename(columns={'mean':'profit_rate','count':'n'})
               .sort_values('profit_rate', ascending=False))

colors_lang = [GREEN if r >= 0.70 else PALETTE if r >= 0.55 else RED
               for r in lang_profit['profit_rate']]

bars3 = ax3a.bar(lang_profit['lang_group'], lang_profit['profit_rate'],
                 color=colors_lang, edgecolor='white', width=0.5)
ax3a.axhline(y=0.699, color='gray', linestyle='--', linewidth=1.2, alpha=0.6)
ax3a.set_ylim(0, 1.0)
ax3a.set_ylabel('Profitability Rate')
ax3a.set_title('Step 3A  |  Profitability Rate by Language Group',
               fontweight='bold', pad=10)
for bar, row in zip(bars3, lang_profit.itertuples()):
    ax3a.text(bar.get_x() + bar.get_width()/2, row.profit_rate + 0.02,
              f'{row.profit_rate:.1%}\n(n={row.n})', ha='center', fontsize=11)

# ════════════════════════════════════════════════════════════
# STEP 3B — English vs Non-English 各类型盈利率对比
# ════════════════════════════════════════════════════════════
ax3b = fig.add_subplot(gs[2, 1])

gdf['is_english'] = gdf['lang_group'] == 'English'
lang_genre = (gdf.groupby(['genre','is_english'])['profitable']
              .agg(['mean','count'])
              .reset_index())
lang_genre.columns = ['genre','is_english','profit_rate','n']
lang_genre = lang_genre[lang_genre['n'] >= 10]

# 只画有两组都存在的类型
genres_both = lang_genre.groupby('genre')['is_english'].nunique()
genres_both = genres_both[genres_both == 2].index.tolist()
plot_df = lang_genre[lang_genre['genre'].isin(genres_both)].copy()
plot_df['Language'] = plot_df['is_english'].map({True:'English', False:'Non-English'})

# 按英语盈利率排序
order = (plot_df[plot_df['Language']=='English']
         .sort_values('profit_rate', ascending=False)['genre'].tolist())

sns.barplot(data=plot_df, y='genre', x='profit_rate', hue='Language',
            order=order, palette={'English': PALETTE, 'Non-English': AMBER},
            ax=ax3b, orient='h')
ax3b.axvline(x=0.699, color='gray', linestyle='--', linewidth=1.2, alpha=0.6)
ax3b.set_xlim(0.3, 1.1)
ax3b.set_xlabel('Profitability Rate')
ax3b.set_ylabel('')
ax3b.set_title('Step 3B  |  English vs Non-English by Genre',
               fontweight='bold', pad=10)
ax3b.legend(loc='lower right', fontsize=10)

# ── 总标题 ──────────────────────────────────────────────────
fig.suptitle('Cross Analysis: Genre × Budget Tier × Language',
             fontsize=18, fontweight='bold', y=1.01)

plt.savefig('/mnt/user-data/outputs/cross_analysis.png',
            dpi=180, bbox_inches='tight', facecolor='white')
print("✅ Saved cross_analysis.png")

# ── 打印关键结论 ─────────────────────────────────────────────
print("\n" + "="*60)
print("KEY FINDINGS")
print("="*60)

print("\n[Step 1] 盈利率最高类型 Top 3:")
top3 = genre_profit.tail(3)[::-1]
for _, r in top3.iterrows():
    roi_val = genre_roi[genre_roi['genre']==r['genre']]['median_roi_pct'].values[0]
    print(f"  {r['genre']:<18} 盈利率={r['profit_rate']:.1%}  中位ROI={roi_val:.0f}%")

print("\n[Step 1] 盈利率最低类型 Bottom 3:")
bot3 = genre_profit.head(3)
for _, r in bot3.iterrows():
    roi_val = genre_roi[genre_roi['genre']==r['genre']]['median_roi_pct'].values[0]
    print(f"  {r['genre']:<18} 盈利率={r['profit_rate']:.1%}  中位ROI={roi_val:.0f}%")

print("\n[Step 2] 热力图亮点（盈利率>85%）:")
high_cells = pivot[pivot['profit_rate'] >= 0.85][['genre','budget_tier','profit_rate','n']]
for _, r in high_cells.iterrows():
    print(f"  {r['genre']:<18} × {r['budget_tier']:<20} = {r['profit_rate']:.1%}  (n={r['n']})")

print("\n[Step 3] 语言组盈利率:")
for _, r in lang_profit.iterrows():
    print(f"  {r['lang_group']:<12} {r['profit_rate']:.1%}  (n={r['n']})")
EOF
