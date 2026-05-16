import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# ==========================================
# 第一部分：数据加载与预处理 (Data Loading & Preprocessing)
# ==========================================
print("正在加载并合并数据...")
# 1. 加载所需的两个核心数据表
df_credits = pd.read_csv("cleaned_archive/credits_cleaned.csv")
df_box_office = pd.read_csv("cleaned_archive/movies_metadata_box_office.csv")

# 2. 通过电影ID (id) 合并数据表
df = pd.merge(df_box_office, df_credits, on='id', how='inner')

# 3. 计算核心财务指标：利润 (Profit) = 票房 (Revenue) - 预算 (Budget)
df['profit'] = df['revenue'] - df['budget']

# 4. 定义数据清洗函数：将字符串格式的列表转换为真实的 Python 列表
def clean_list_string(s):
    try:
        lst = ast.literal_eval(s)
        # 确保转换后是列表且不为空
        if isinstance(lst, list) and len(lst) > 0:
            return lst
        return []
    except:
        return []

print("正在清洗并展开数据结构...")
# 应用清洗函数到演员和导演列
df['cast_list_clean'] = df['cast_list'].apply(clean_list_string)
df['directors_clean'] = df['directors'].apply(clean_list_string)


# ==========================================
# 第二部分：数据提取与统计 (Data Extraction & Aggregation)
# ==========================================
print("正在计算各大榜单 Top 10...")
# 使用 explode 将包含多个元素的列表“炸开”，使每个演员/导演独占一行，以便独立统计
df_cast = df.explode('cast_list_clean')
df_directors = df.explode('directors_clean')

# --- 演员榜单统计 ---
# 1. 演出数量排名前 10 的演员
top_actors_count = df_cast['cast_list_clean'].value_counts().head(10)
# 2. 累计利润排名前 10 的演员 (过滤掉利润为 NaN 或 0 的异常值后求和)
top_actors_profit = df_cast.groupby('cast_list_clean')['profit'].sum().sort_values(ascending=False).head(10)

# --- 导演榜单统计 ---
# 3. 执导数量排名前 10 的导演
top_directors_count = df_directors['directors_clean'].value_counts().head(10)
# 4. 累计利润排名前 10 的导演
top_directors_profit = df_directors.groupby('directors_clean')['profit'].sum().sort_values(ascending=False).head(10)


# ==========================================
# 第三部分：学术级数据可视化 (Academic Data Visualization)
# ==========================================
print("正在生成学术级高清图表...")

# 1. 恢复默认设置并启用高级图表风格
plt.style.use('default')
# 设置纯白背景，移除多余网格线，贴近学术期刊标准
sns.set_theme(style="ticks", rc={"axes.facecolor": "#FFFFFF", "figure.facecolor": "#FFFFFF"})
# 设置通用无衬线字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# 创建一个 2x2 的画板，尺寸为宽 20，高 14
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# 2. 定义高级渐变配色方案 (莫兰迪色系)
# count 图使用深蓝灰渐变，profit 图使用翡翠绿渐变
color_count = sns.color_palette("ch:s=.25,rot=-.25", 10)[::-1]
color_profit = sns.color_palette("crest", 10)[::-1]

# 3. 自定义坐标轴格式化函数：将百亿级别的数字转换为以 'B' (Billion) 结尾的格式
def billon_fmt(x, pos):
    return f'${x*1e-9:.1f}B'

# ------------------------------------------
# 图 1 (左上): 演出数量 Top 10 演员
# ------------------------------------------
sns.barplot(ax=axes[0, 0], x=top_actors_count.values, y=top_actors_count.index, 
            palette=color_count, edgecolor="black", linewidth=0.5)
axes[0, 0].set_title('Top 10 Actors by Movie Count', fontsize=18, fontweight='bold', pad=15)
axes[0, 0].set_xlabel('Number of Movies', fontsize=14, fontweight='medium')
axes[0, 0].set_ylabel('')
axes[0, 0].tick_params(axis='both', which='major', labelsize=13)
# 添加具体的数值文本到柱状图末尾
for i, v in enumerate(top_actors_count.values):
    axes[0, 0].text(v + 0.5, i, f" {v}", va='center', fontsize=12, fontweight='bold', color='#333333')

# ------------------------------------------
# 图 2 (右上): 累计利润 Top 10 演员
# ------------------------------------------
sns.barplot(ax=axes[0, 1], x=top_actors_profit.values, y=top_actors_profit.index, 
            palette=color_profit, edgecolor="black", linewidth=0.5)
axes[0, 1].set_title('Top 10 Actors by Total Profit', fontsize=18, fontweight='bold', pad=15)
axes[0, 1].set_xlabel('Total Profit (USD)', fontsize=14, fontweight='medium')
axes[0, 1].set_ylabel('')
axes[0, 1].tick_params(axis='both', which='major', labelsize=13)
axes[0, 1].xaxis.set_major_formatter(FuncFormatter(billon_fmt))
for i, v in enumerate(top_actors_profit.values):
    axes[0, 1].text(v + 1e8, i, f" ${v*1e-9:.1f}B", va='center', fontsize=12, fontweight='bold', color='#333333')

# ------------------------------------------
# 图 3 (左下): 执导数量 Top 10 导演
# ------------------------------------------
sns.barplot(ax=axes[1, 0], x=top_directors_count.values, y=top_directors_count.index, 
            palette=color_count, edgecolor="black", linewidth=0.5)
axes[1, 0].set_title('Top 10 Directors by Movie Count', fontsize=18, fontweight='bold', pad=15)
axes[1, 0].set_xlabel('Number of Movies', fontsize=14, fontweight='medium')
axes[1, 0].set_ylabel('')
axes[1, 0].tick_params(axis='both', which='major', labelsize=13)
for i, v in enumerate(top_directors_count.values):
    axes[1, 0].text(v + 0.3, i, f" {v}", va='center', fontsize=12, fontweight='bold', color='#333333')

# ------------------------------------------
# 图 4 (右下): 累计利润 Top 10 导演
# ------------------------------------------
sns.barplot(ax=axes[1, 1], x=top_directors_profit.values, y=top_directors_profit.index, 
            palette=color_profit, edgecolor="black", linewidth=0.5)
axes[1, 1].set_title('Top 10 Directors by Total Profit', fontsize=18, fontweight='bold', pad=15)
axes[1, 1].set_xlabel('Total Profit (USD)', fontsize=14, fontweight='medium')
axes[1, 1].set_ylabel('')
axes[1, 1].tick_params(axis='both', which='major', labelsize=13)
axes[1, 1].xaxis.set_major_formatter(FuncFormatter(billon_fmt))
for i, v in enumerate(top_directors_profit.values):
    axes[1, 1].text(v + 1e8, i, f" ${v*1e-9:.1f}B", va='center', fontsize=12, fontweight='bold', color='#333333')


# ==========================================
# 第四部分：全局排版修饰与保存 (Final Polish & Save)
# ==========================================
# 移除所有子图的顶部和右侧外边框线 (学术期刊常见要求)
sns.despine(fig)

# 为每个子图的 X 轴增加浅灰色虚线网格，辅助视觉对齐
for ax in axes.flat:
    ax.grid(axis='x', linestyle='--', alpha=0.6, color='#E0E0E0')
    ax.set_axisbelow(True) # 确保网格线位于彩色柱子下方

# 自动调整子图间距，防止文字重叠
plt.tight_layout(pad=3.0)

# 导出超高清图片 (400 DPI，无多余白边)，可直接插入 Word 报告或 PPT 中
output_filename = 'academic_hollywood_analysis.png'
plt.savefig(output_filename, dpi=400, bbox_inches='tight')
print(f"✅ 图表已成功生成并保存为 '{output_filename}'！")

# 强制在 Jupyter Notebook 环境中显示图表
plt.show()