import pandas as pd

def analyze_rating_revenue_correlation(movies_path, ratings_path):
    # 1. Load Data
    df_movies = pd.read_csv(movies_path, low_memory=False)
    df_ratings = pd.read_csv(ratings_path)
    
    # 2. Data Cleaning
    df_movies['id'] = pd.to_numeric(df_movies['id'], errors='coerce')
    df_movies = df_movies.dropna(subset=['id'])
    df_movies['id'] = df_movies['id'].astype(int)
    
    # 3. Data Aggregation
    avg_ratings = df_ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    avg_ratings = avg_ratings.rename(columns={'mean': 'avg_rating', 'count': 'rating_count'})
    
    # 4. Merge Data
    merged_df = pd.merge(df_movies, avg_ratings, left_on='id', right_on='movieId', how='inner')
    
    # 5. Revenue Data Cleaning
    merged_df['revenue'] = pd.to_numeric(merged_df['revenue'], errors='coerce')
    valid_data = merged_df[(merged_df['revenue'] > 0) & (merged_df['revenue'].notnull())].copy()
    
    # 6. Calculate Correlations
    corr_pearson = valid_data['avg_rating'].corr(valid_data['revenue'], method='pearson')
    corr_spearman = valid_data['avg_rating'].corr(valid_data['revenue'], method='spearman')
    
    # 7. Extract Descriptive Statistics
    stats_data = {
        "valid_movies_count": len(valid_data),
        "pearson_correlation": corr_pearson,
        "spearman_correlation": corr_spearman,
        "revenue_stats": valid_data['revenue'].describe(),
        "rating_stats": valid_data['avg_rating'].describe()
    }
    
    return valid_data, stats_data

clean_df, results = analyze_rating_revenue_correlation(
    'cleaned_archive/movies_metadata_cleaned.csv', 
    'cleaned_archive/ratings_small_cleaned.csv'
)

print("\n=== Analysis Results ===")
print(f"Valid Movie Samples: {results['valid_movies_count']}")
print(f"Pearson Correlation (Linear): {results['pearson_correlation']:.4f}")
print(f"Spearman Correlation (Rank): {results['spearman_correlation']:.4f}")
print("=======================\n")