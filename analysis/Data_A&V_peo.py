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
# Apply the cleaning function to the cast and directors columns
df['cast_list_clean'] = df['cast_list'].apply(clean_list_string)
df['directors_clean'] = df['directors'].apply(clean_list_string)


# ==========================================
# Data Extraction & Aggregation
# ==========================================
print("Calculating Top 10 lists...")
# Use explode to "unpack" lists with multiple elements, so each actor/director occupies a separate row for independent statistics
df_cast = df.explode('cast_list_clean')
df_directors = df.explode('directors_clean')

# --- Actor Statistics ---
# 1. Top 10 actors by movie count
top_actors_count = df_cast['cast_list_clean'].value_counts().head(10)
# 2. Top 10 actors by total profit (filtering out NaN or 0 profit values)
top_actors_profit = df_cast.groupby('cast_list_clean')['profit'].sum().sort_values(ascending=False).head(10)

# --- Director Statistics ---
# 3. Top 10 directors by movie count
top_directors_count = df_directors['directors_clean'].value_counts().head(10)
# 4. Top 10 directors by total profit
top_directors_profit = df_directors.groupby('directors_clean')['profit'].sum().sort_values(ascending=False).head(10)


# ==========================================
# Academic Data Visualization
# ==========================================
print("Generating academic-grade high-resolution charts...")

# 1. Restore default settings and enable advanced chart styles
plt.style.use('default')
# Set pure white background, remove unnecessary grid lines, close to academic journal standards
sns.set_theme(style="ticks", rc={"axes.facecolor": "#FFFFFF", "figure.facecolor": "#FFFFFF"})
# Set universal sans-serif font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# Create a 2x2 canvas, size width 20, height 14
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# 2. Define advanced gradient color schemes (Morandi color palette)
# count charts use deep blue-gray gradient, profit charts use emerald green gradient
color_count = sns.color_palette("ch:s=.25,rot=-.25", 10)[::-1]
color_profit = sns.color_palette("crest", 10)[::-1]

# 3. Custom axis formatting function: Convert billion-level numbers to 'B' (Billion) format
def billon_fmt(x, pos):
    return f'${x*1e-9:.1f}B'

# ------------------------------------------
# Chart 1 (Top Left): Top 10 Actors by Movie Count
# ------------------------------------------
sns.barplot(ax=axes[0, 0], x=top_actors_count.values, y=top_actors_count.index, 
            palette=color_count, edgecolor="black", linewidth=0.5)
axes[0, 0].set_title('Top 10 Actors by Movie Count', fontsize=18, fontweight='bold', pad=15)
axes[0, 0].set_xlabel('Number of Movies', fontsize=14, fontweight='medium')
axes[0, 0].set_ylabel('')
axes[0, 0].tick_params(axis='both', which='major', labelsize=13)
# Add value labels to the end of the bars
for i, v in enumerate(top_actors_count.values):
    axes[0, 0].text(v + 0.5, i, f" {v}", va='center', fontsize=12, fontweight='bold', color='#333333')

# ------------------------------------------
# Chart 2 (Top Right): Top 10 Actors by Total Profit
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
# Chart 3 (Bottom Left): Top 10 Directors by Movie Count
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
# Chart 4 (Bottom Right): Top 10 Directors by Total Profit
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
# Part 4: Global Layout and Save (Final Polish & Save)
# ==========================================
# Remove top and right spines from all subplots (common requirement in academic journals)
sns.despine(fig)

# Add light gray dashed grid lines to the X axis of each subplot for visual alignment
for ax in axes.flat:
    ax.grid(axis='x', linestyle='--', alpha=0.6, color='#E0E0E0')
    ax.set_axisbelow(True) # Ensure grid lines are below the colored bars

# Automatically adjust subplot spacing to prevent text overlap
plt.tight_layout(pad=3.0)

# Export ultra-high-definition image (400 DPI, no extra margins), can be directly inserted into Word reports or PPT
output_filename = 'outputs/academic_hollywood_analysis.png'
plt.savefig(output_filename, dpi=400, bbox_inches='tight')
print(f"Chart successfully generated and saved as '{output_filename}'!")

# Force display of the chart in Jupyter Notebook environment
plt.show()