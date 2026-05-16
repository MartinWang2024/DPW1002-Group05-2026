import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_beautiful_visuals(movies_path, ratings_path):
    # --- 1. Data Cleaning & Preparation ---
    movies = pd.read_csv(movies_path, low_memory=False)
    ratings = pd.read_csv(ratings_path)
    
    movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
    movies = movies.dropna(subset=['id'])
    movies['id'] = movies['id'].astype(int)
    
    avg_ratings = ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    avg_ratings = avg_ratings.rename(columns={'mean': 'avg_rating', 'count': 'rating_count'})
    
    merged_df = pd.merge(movies, avg_ratings, left_on='id', right_on='movieId', how='inner')
    merged_df['revenue'] = pd.to_numeric(merged_df['revenue'], errors='coerce')
    valid_data = merged_df[(merged_df['revenue'] > 0) & (merged_df['revenue'].notnull())].copy()

    # --- 2. Core Feature Engineering ---
    bins = [0, 2.999, 3.999, 5.01]
    labels = ['Low (<3.0)', 'Medium (3.0-4.0)', 'High (>4.0)']
    valid_data['rating_tier'] = pd.cut(valid_data['avg_rating'], bins=bins, labels=labels)
    valid_data['revenue_m'] = valid_data['revenue'] / 1_000_000

    # --- 3. Visualization ---
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=120)

    # Left Plot: Bar chart for Mean & Median
    tier_stats = valid_data.groupby('rating_tier', observed=False)['revenue_m'].agg(
        Mean=('mean'),
        Median=('median')
    ).reset_index()
    
    melted_stats = pd.melt(tier_stats, id_vars="rating_tier", var_name="Metric", value_name="Revenue ($M)")
    
    sns.barplot(
        x="rating_tier", 
        y="Revenue ($M)", 
        hue="Metric", 
        data=melted_stats, 
        palette=["#4C72B0", "#55A868"], 
        ax=axes[0]
    )
    axes[0].set_title('Average & Median Revenue by Rating Tier', pad=15)
    axes[0].set_xlabel('Rating Tier')
    axes[0].set_ylabel('Box Office Revenue (Millions USD)')
    
    for p in axes[0].patches:
        height = p.get_height()
        if pd.notnull(height) and height > 0:
            axes[0].annotate(f'${height:.0f}M', 
                             (p.get_x() + p.get_width() / 2., height), 
                             ha='center', va='center', 
                             xytext=(0, 9), textcoords='offset points',
                             fontsize=12)

    # Right Plot: Boxplot + Stripplot for distributions
    sns.boxplot(
        x="rating_tier", 
        y="revenue_m", 
        data=valid_data, 
        palette="pastel",
        showfliers=False, 
        ax=axes[1]
    )
    sns.stripplot(
        x="rating_tier", 
        y="revenue_m", 
        data=valid_data, 
        color="#C44E52", 
        alpha=0.5, 
        jitter=True, 
        size=4,
        ax=axes[1]
    )
    axes[1].set_title('Revenue Distribution & Outliers (Blockbusters)', pad=15)
    axes[1].set_xlabel('Rating Tier')
    axes[1].set_ylabel('Box Office Revenue (Millions USD)')
    axes[1].set_ylim(0, 2000)

    plt.tight_layout()
    plt.show()

create_beautiful_visuals('cleaned_archive/movies_metadata_cleaned.csv', 'cleaned_archive/ratings_small_cleaned.csv')