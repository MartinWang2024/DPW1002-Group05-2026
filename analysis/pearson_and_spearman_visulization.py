import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def visualize_correlations(movies_path, ratings_path):
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

    valid_data['revenue_m'] = valid_data['revenue'] / 1_000_000

    corr_pearson, p_pearson = stats.pearsonr(valid_data['avg_rating'], valid_data['revenue_m'])
    corr_spearman, p_spearman = stats.spearmanr(valid_data['avg_rating'], valid_data['revenue_m'])

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=120)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # ==========================================
    # Left Plot: Pearson
    # ==========================================
    sns.regplot(
        x="avg_rating", 
        y="revenue_m", 
        data=valid_data,
        ax=axes[0],
        scatter_kws={'alpha':0.4, 'color': '#4C72B0', 's': 30},
        line_kws={'color': '#C44E52', 'linewidth': 2}
    )
    
    axes[0].set_title('Linear Correlation (Pearson)', fontsize=16, pad=15)
    axes[0].set_xlabel('Average Rating (1-5 Stars)', fontsize=14)
    axes[0].set_ylabel('Box Office Revenue (Millions USD)', fontsize=14)
    
    textstr_pearson = '\n'.join((
        f'Pearson r = {corr_pearson:.4f}',
        f'p-value = {p_pearson:.4f}',
        'Measures linear relationship.',
        'Near 0 means no straight line fits well.'
    ))
    axes[0].text(0.05, 0.95, textstr_pearson, transform=axes[0].transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    # ==========================================
    # Right Plot: Spearman
    # ==========================================
    valid_data['rating_quantile'] = pd.qcut(valid_data['avg_rating'], q=10, duplicates='drop')
    quantile_stats = valid_data.groupby('rating_quantile', observed=False)['revenue_m'].median().reset_index()
    quantile_stats['rating_mid'] = quantile_stats['rating_quantile'].apply(lambda x: x.mid)

    sns.scatterplot(
        x="avg_rating", 
        y="revenue_m", 
        data=valid_data,
        ax=axes[1],
        alpha=0.2, color='gray'
    )
    
    sns.lineplot(
        x="rating_mid",
        y="revenue_m",
        data=quantile_stats,
        ax=axes[1],
        color='#55A868',
        linewidth=3,
        marker='o',
        markersize=10
    )

    axes[1].set_title('Rank Correlation (Spearman)', fontsize=16, pad=15)
    axes[1].set_xlabel('Average Rating (1-5 Stars)', fontsize=14)
    axes[1].set_ylabel('Box Office Revenue (Millions USD)', fontsize=14)
    axes[1].set_ylim(axes[0].get_ylim()) 

    textstr_spearman = '\n'.join((
        f'Spearman \u03C1 = {corr_spearman:.4f}',
        f'p-value = {p_spearman:.4f}',
        'Measures monotonic (rank) relationship.',
        'Green line shows median revenue trend.',
        'Near 0 means rank doesn\'t predict revenue.'
    ))
    axes[1].text(0.05, 0.95, textstr_spearman, transform=axes[1].transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()

visualize_correlations('cleaned_archive/movies_metadata_cleaned.csv', 'cleaned_archive/ratings_small_cleaned.csv')