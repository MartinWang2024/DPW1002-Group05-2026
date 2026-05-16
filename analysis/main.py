import pandas as pd

def advanced_movie_analysis(movies_path, ratings_path):
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

    # ====================================
    # Module 1: Rating Tiers
    # ====================================
    bins = [0, 2.999, 3.999, 5.01]
    labels = ['Low (<3.0)', 'Medium (3.0-4.0)', 'High (>4.0)']
    valid_data['rating_tier'] = pd.cut(valid_data['avg_rating'], bins=bins, labels=labels)
    
    tier_stats = valid_data.groupby('rating_tier')['revenue'].agg(
        avg_revenue=('mean'),
        median_revenue=('median'),
        movie_count=('count')
    ).reset_index()
    
    print("=== 1. Rating Tiers Analysis ===")
    for _, row in tier_stats.iterrows():
        print(f"[{row['rating_tier']}] Movies: {row['movie_count']} | Avg Revenue: ${row['avg_revenue']:,.0f} | Median: ${row['median_revenue']:,.0f}")

    # ====================================
    # Module 2: Outliers
    # ====================================
    Q1 = valid_data['revenue'].quantile(0.25)
    Q3 = valid_data['revenue'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR 
    
    outliers = valid_data[valid_data['revenue'] > upper_bound]
    normal_movies = valid_data[valid_data['revenue'] <= upper_bound]
    
    print("\n=== 2. Box Office Outliers Analysis ===")
    print(f"Defining blockbusters as revenue > ${upper_bound:,.0f}")
    print(f"Blockbusters: {len(outliers)}, Avg Rating: {outliers['avg_rating'].mean():.2f}")
    print(f"Normal Movies: {len(normal_movies)}, Avg Rating: {normal_movies['avg_rating'].mean():.2f}")

    # ====================================
    # Module 3: Top 5 Records
    # ====================================
    print("\n=== 3. Top 5 Box Office Movies ===")
    top5 = valid_data.nlargest(5, 'revenue')[['title', 'revenue', 'avg_rating']]
    for _, row in top5.iterrows():
        print(f"'{row['title']}' - Revenue: ${row['revenue']:,.0f} - Rating: {row['avg_rating']:.2f}")
        
    return valid_data

final_data = advanced_movie_analysis('cleaned_archive/movies_metadata_cleaned.csv', 'cleaned_archive/ratings_small_cleaned.csv')