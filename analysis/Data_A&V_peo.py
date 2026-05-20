import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# ==========================================
# Data Loading & Preprocessing
# ==========================================
print("Loading and merging data...")
# 1. Load the two core datasets
df_credits = pd.read_csv("cleaned_archive/credits_cleaned.csv")
df_box_office = pd.read_csv("cleaned_archive/movies_metadata_box_office.csv")

# 2. Merge datasets on movie ID (id)
df = pd.merge(df_box_office, df_credits, on='id', how='inner')

# 3. Calculate core financial metric: Profit = Revenue - Budget
df['profit'] = df['revenue'] - df['budget']

# 4. Define data cleaning function: Convert string-formatted lists to actual Python lists
def clean_list_string(s):
    try:
        lst = ast.literal_eval(s)
        # Ensure the result is a list and not empty
        if isinstance(lst, list) and len(lst) > 0:
            return lst
        return []
    except:
        return []

print("Cleaning and expanding data structures...")
# Apply cleaning function to cast and director columns
df['cast_list_clean'] = df['cast_list'].apply(clean_list_string)
df['directors_clean'] = df['directors'].apply(clean_list_string)


# ==========================================
# Part 2: Data Extraction & Aggregation
# ==========================================
print("Calculating Top 10 charts...")
# Explode lists to isolate each actor and director for independent counting
df_cast = df.explode('cast_list_clean')
df_directors = df.explode('directors_clean')

# --- Actor Statistics ---
# 1. Top 10 actors by movie count
top_actors_count = df_cast['cast_list_clean'].value_counts().head(10)
# 2. Top 10 actors by total profit
top_actors_profit = df_cast.groupby('cast_list_clean')['profit'].sum().sort_values(ascending=False).head(10)

# --- Director Statistics ---
# 3. Top 10 directors by movie count
top_directors_count = df_directors['directors_clean'].value_counts().head(10)
# 4. Top 10 directors by total profit
top_directors_profit = df_directors.groupby('directors_clean')['profit'].sum().sort_values(ascending=False).head(10)


# ==========================================
# Part 3: Academic Data Visualization
# ==========================================
print("Generating academic-style high-resolution charts...")

# 1. Reset defaults and enable advanced style
plt.style.use('default')
# Set white background close to academic standards
sns.set_theme(style="ticks", rc={"axes.facecolor": "#FFFFFF", "figure.facecolor": "#FFFFFF"})
# Set universal sans-serif font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# Create a 2x2 subplot canvas (width=20, height=14)
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# 2. Define advanced color palettes
# Blue-gray gradient for count charts, emerald green gradient for profit charts
color_count = sns.color_palette("ch:s=.25,rot=-.25", 10)[::-1]
color_profit = sns.color_palette("crest", 10)[::-1]

# 3. Format axis labels into Billions ('B')
def billon_fmt(x, pos):
    return f'${x*1e-9:.1f}B'

# ------------------------------------------
# Chart 1: Top 10 Actors by Movie Count
# ------------------------------------------
sns.barplot(ax=axes[0, 0], x=top_actors_count.values, y=top_actors_count.index,
            hue=top_actors_count.index, legend=False,
            palette=color_count, edgecolor="black", linewidth=0.5)
axes[0, 0].set_title('Top 10 Actors by Movie Count', fontsize=18, fontweight='bold', pad=15)
axes[0, 0].set_xlabel('Number of Movies', fontsize=14, fontweight='medium')
axes[0, 0].set_ylabel('')
axes[0, 0].tick_params(axis='both', which='major', labelsize=13)
# Add text labels to the end of bars
for i, v in enumerate(top_actors_count.values):
    axes[0, 0].text(v + 0.5, i, f" {v}", va='center', fontsize=12, fontweight='bold', color='#333333')

# ------------------------------------------
# Chart 2: Top 10 Actors by Total Profit
# ------------------------------------------
sns.barplot(ax=axes[0, 1], x=top_actors_profit.values, y=top_actors_profit.index,
            hue=top_actors_profit.index, legend=False,
            palette=color_profit, edgecolor="black", linewidth=0.5)
axes[0, 1].set_title('Top 10 Actors by Total Profit', fontsize=18, fontweight='bold', pad=15)
axes[0, 1].set_xlabel('Total Profit (USD)', fontsize=14, fontweight='medium')
axes[0, 1].set_ylabel('')
axes[0, 1].tick_params(axis='both', which='major', labelsize=13)
axes[0, 1].xaxis.set_major_formatter(FuncFormatter(billon_fmt))
for i, v in enumerate(top_actors_profit.values):
    axes[0, 1].text(v + 1e8, i, f" ${v*1e-9:.1f}B", va='center', fontsize=12, fontweight='bold', color='#333333')

# ------------------------------------------
# Chart 3: Top 10 Directors by Movie Count
# ------------------------------------------
sns.barplot(ax=axes[1, 0], x=top_directors_count.values, y=top_directors_count.index,
            hue=top_directors_count.index, legend=False,
            palette=color_count, edgecolor="black", linewidth=0.5)
axes[1, 0].set_title('Top 10 Directors by Movie Count', fontsize=18, fontweight='bold', pad=15)
axes[1, 0].set_xlabel('Number of Movies', fontsize=14, fontweight='medium')
axes[1, 0].set_ylabel('')
axes[1, 0].tick_params(axis='both', which='major', labelsize=13)
for i, v in enumerate(top_directors_count.values):
    axes[1, 0].text(v + 0.3, i, f" {v}", va='center', fontsize=12, fontweight='bold', color='#333333')

# ------------------------------------------
# Chart 4: Top 10 Directors by Total Profit
# ------------------------------------------
sns.barplot(ax=axes[1, 1], x=top_directors_profit.values, y=top_directors_profit.index,
            hue=top_directors_profit.index, legend=False,
            palette=color_profit, edgecolor="black", linewidth=0.5)
axes[1, 1].set_title('Top 10 Directors by Total Profit', fontsize=18, fontweight='bold', pad=15)
axes[1, 1].set_xlabel('Total Profit (USD)', fontsize=14, fontweight='medium')
axes[1, 1].set_ylabel('')
axes[1, 1].tick_params(axis='both', which='major', labelsize=13)
axes[1, 1].xaxis.set_major_formatter(FuncFormatter(billon_fmt))
for i, v in enumerate(top_directors_profit.values):
    axes[1, 1].text(v + 1e8, i, f" ${v*1e-9:.1f}B", va='center', fontsize=12, fontweight='bold', color='#333333')


# ==========================================
# Part 4: Final Polish & Save
# ==========================================
# Remove top and right borders (academic standard)
sns.despine(fig)

# Add light gray gridlines to the X-axis
for ax in axes.flat:
    ax.grid(axis='x', linestyle='--', alpha=0.6, color='#E0E0E0')
    ax.set_axisbelow(True)

# Automatically adjust subplot spacing to prevent text overlap
plt.tight_layout(pad=3.0)

# Export ultra-high-definition image (400 DPI, no extra margins), can be directly inserted into Word reports or PPT
output_filename = 'outputs/academic_hollywood_analysis.png'
plt.savefig(output_filename, dpi=400, bbox_inches='tight')
print(f"Chart successfully generated and saved as '{output_filename}'!")

# Force display of the chart in Jupyter Notebook environment
plt.show()