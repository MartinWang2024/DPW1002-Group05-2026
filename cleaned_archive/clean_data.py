import numpy as np
import pandas as pd

try:
    from .clean_credits import clean_credits
    from .clean_keywords import clean_keywords
    from .clean_links_small import clean_links_small
    from .clean_movies_metadata import clean_movies_metadata
    from .clean_ratings_small import clean_ratings_small
except ImportError:
    from clean_credits import clean_credits
    from clean_keywords import clean_keywords
    from clean_links_small import clean_links_small
    from clean_movies_metadata import clean_movies_metadata
    from clean_ratings_small import clean_ratings_small


# ─────────────────────────────────────────────
# main process
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Movie Box Office Factor Analysis - Data Cleaning")
    print("  Using pandas", pd.__version__, "/ numpy", np.__version__)
    print("=" * 60)

    movies = clean_movies_metadata()
    ratings = clean_ratings_small()
    keywords = clean_keywords()
    credits = clean_credits()
    links = clean_links_small()

    # Summary
    print("\n" + "=" * 60)
    print("  Cleaning completed. Output files:")
    print("=" * 60)
    print("  movies_metadata_cleaned.csv")
    print("  movies_metadata_box_office.csv")
    print("  ratings_small_cleaned.csv")
    print("  keywords_cleaned.csv")
    print("  credits_cleaned.csv")
    print("  links_small_cleaned.csv")

    print("\n  Key statistics:")
    print(f"  movies_metadata after full cleaning: {len(movies):,}")
    print(f"  movies_metadata box office subset:   {((movies['revenue'] > 0) & (movies['budget'] > 0)).sum():,}")
    print(f"  ratings_small after cleaning:        {len(ratings):,}")
    print(f"  keywords after cleaning:             {len(keywords):,}")
    print(f"  credits after cleaning:              {len(credits):,}")
    print(f"  links_small after cleaning:          {len(links):,}")

    print("\n  runtime / budget / revenue outlier flag columns are kept in")
    print("  movies_metadata_cleaned.csv for optional filtering in later analysis.")


if __name__ == "__main__":
    main()
