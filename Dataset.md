# About Dataset

## Context

These files contain metadata for all 45,000 movies listed in the Full MovieLens Dataset. The dataset consists of movies released on or before July 2017.

Data points include cast, crew, plot keywords, budget, revenue, posters, release dates, languages, production companies, countries, TMDB vote counts, and vote averages.

This dataset also includes files containing 26 million ratings from 270,000 users for all 45,000 movies. Ratings are on a scale of 1-5 and have been obtained from the official GroupLens website.

## Content

This dataset consists of the following files:

- **`movies_metadata.csv`**: The main Movies Metadata file. Contains information on 45,000 movies featured in the Full MovieLens dataset, including posters, backdrops, budget, revenue, release dates, languages, production countries, and companies.
- **`keywords.csv`**: Contains movie plot keywords for MovieLens movies, available in the form of a stringified JSON object.
- **`credits.csv`**: Consists of cast and crew information for all movies, available in the form of a stringified JSON object.
- **`links.csv`**: Contains the TMDB and IMDB IDs of all movies featured in the Full MovieLens dataset.
- **`links_small.csv`**: Contains the TMDB and IMDB IDs of a small subset of 9,000 movies from the full dataset.
- **`ratings_small.csv`**: A subset of 100,000 ratings from 700 users on 9,000 movies.

The Full MovieLens Dataset (26 million ratings and 750,000 tag applications from 270,000 users on all 45,000 movies) can be accessed via the official GroupLens source.

## Acknowledgements

This dataset is an ensemble of data collected from TMDB and GroupLens.

The Movie Details, Credits, and Keywords have been collected from the TMDB Open API. This product uses the TMDb API but is not endorsed or certified by TMDb.

Their API also provides access to data on many additional movies, actors, actresses, crew members, and TV shows.

The Movie Links and Ratings have been obtained from the official GroupLens website. These files are part of the dataset available there.

## Inspiration

This dataset was assembled as part of my second Capstone Project for Springboard's Data Science Career Track.

I wanted to perform an extensive EDA on movie data to narrate the history and story of cinema, and use this metadata in combination with MovieLens ratings to build various types of recommender systems.

Both notebooks are available as kernels with this dataset:

- The Story of Film
- Movie Recommender Systems

Some of the things you can do with this dataset:

- Predict movie revenue and/or movie success based on specific metrics.
- Analyze which movies tend to get higher vote counts and vote averages on TMDB.
- Build content-based and collaborative filtering recommendation engines.


