import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# ==========================================
# 1. 数据加载与清洗
# ==========================================
print("正在加载并处理国家数据...")
df_box_office = pd.read_csv("cleaned_archive/movies_metadata_box_office.csv")

def clean_list_string(s):
    try:
        lst = ast.literal_eval(s)
        if isinstance(lst, list) and len(lst) > 0: return lst
        return []
    except:
        return []

df_box_office['countries_clean'] = df_box_office['production_countries_list'].apply(clean_list_string)
# 炸开国家列表，处理多国合拍的情况
df_countries = df_box_office.explode('countries_clean')
df_countries['profit'] = df_countries['revenue'] - df_countries['budget']
df_countries = df_countries.dropna(subset=['countries_clean'])

# ==========================================
# 2. 统计计算
# ==========================================
# 图1数据：发片数量 Top 10
top_countries_count = df_countries['countries_clean'].value_counts().head(10)

# 图2数据：总利润 Top 10
top_countries_profit = df_countries.groupby('countries_clean')['profit'].sum().sort_values(ascending=False).head(10)

# 图3数据：平均单片利润 Top 10 (为了排除极个别偶然高利润电影，设置发片量 >= 20 的门槛)
country_stats = df_countries.groupby('countries_clean').agg(
    movie_count=('id', 'count'),
    total_profit=('profit', 'sum')
)
country_stats['avg_profit'] = country_stats['total_profit'] / country_stats['movie_count']
top_countries_avg_profit = country_stats[country_stats['movie_count'] >= 20].sort_values(by='avg_profit', ascending=False).head(10)

# ==========================================
# 3. 生成学术级三连图表 (1x3 布局)
# ==========================================
plt.style.use('default')
sns.set_theme(style="ticks", rc={"axes.facecolor": "#FFFFFF", "figure.facecolor": "#FFFFFF"})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# 创建 1行3列的宽幅画板 (宽度28，高度8)
fig, axes = plt.subplots(1, 3, figsize=(28, 8))

# 为三张图分别定义高级渐变配色方案
color_count = sns.color_palette("ch:s=-.2,r=.6", 10)[::-1]  # 暖紫渐变
color_profit = sns.color_palette("mako", 10)[::-1]          # 深蓝绿渐变
color_avg = sns.color_palette("flare", 10)[::-1]            # 红橙渐变

# 坐标轴数字格式化函数
def billon_fmt(x, pos):
    return f'${x*1e-9:.1f}B'
def million_fmt(x, pos):
    return f'${x*1e-6:.0f}M'

# ------------------------------------------
# 左图 (图1): 电影产出数量
# ------------------------------------------
sns.barplot(ax=axes[0], x=top_countries_count.values, y=top_countries_count.index, 
            palette=color_count, edgecolor="black", linewidth=0.5)
axes[0].set_title('Top 10 Countries by Movie Count', fontsize=18, fontweight='bold', pad=15)
axes[0].set_xlabel('Number of Movies', fontsize=14, fontweight='medium')
axes[0].set_ylabel('')
axes[0].tick_params(axis='both', which='major', labelsize=13)
# 添加数值标签
for i, v in enumerate(top_countries_count.values):
    axes[0].text(v + 10, i, f" {v}", va='center', fontsize=12, fontweight='bold', color='#333333')

# ------------------------------------------
# 中图 (图2): 累计总利润
# ------------------------------------------
sns.barplot(ax=axes[1], x=top_countries_profit.values, y=top_countries_profit.index, 
            palette=color_profit, edgecolor="black", linewidth=0.5)
axes[1].set_title('Top 10 Countries by Total Profit', fontsize=18, fontweight='bold', pad=15)
axes[1].set_xlabel('Total Profit (USD)', fontsize=14, fontweight='medium')
axes[1].set_ylabel('')
axes[1].tick_params(axis='both', which='major', labelsize=13)
axes[1].xaxis.set_major_formatter(FuncFormatter(billon_fmt))
# 添加数值标签
for i, v in enumerate(top_countries_profit.values):
    axes[1].text(v + 1e9, i, f" ${v*1e-9:.1f}B", va='center', fontsize=12, fontweight='bold', color='#333333')

# ------------------------------------------
# 右图 (图3): 平均利润 (要求发片量>=20)
# ------------------------------------------
sns.barplot(ax=axes[2], x=top_countries_avg_profit['avg_profit'].values, y=top_countries_avg_profit.index, 
            palette=color_avg, edgecolor="black", linewidth=0.5)
axes[2].set_title('Top 10 Countries by Average Profit\n(Min. 20 Movies)', fontsize=18, fontweight='bold', pad=15)
axes[2].set_xlabel('Average Profit (USD)', fontsize=14, fontweight='medium')
axes[2].set_ylabel('')
axes[2].tick_params(axis='both', which='major', labelsize=13)
axes[2].xaxis.set_major_formatter(FuncFormatter(million_fmt))
# 添加数值标签
for i, v in enumerate(top_countries_avg_profit['avg_profit'].values):
    axes[2].text(v + 2e6, i, f" ${v*1e-6:.1f}M", va='center', fontsize=12, fontweight='bold', color='#333333')

# ------------------------------------------
# 整体排版修饰与保存
# ------------------------------------------
sns.despine(fig) # 移除外边框

# 增加 X 轴辅助虚线网格
for ax in axes.flat:
    ax.grid(axis='x', linestyle='--', alpha=0.6, color='#E0E0E0')
    ax.set_axisbelow(True)

plt.tight_layout(pad=4.0) # 增大图与图之间的间距防止重叠

output_filename = 'academic_country_3_charts.png'
plt.savefig(output_filename, dpi=400, bbox_inches='tight')
print(f"✅ 图表已保存为 '{output_filename}'")
plt.show()