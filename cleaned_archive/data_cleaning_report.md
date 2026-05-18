# Data Cleaning Report

**Project**: Movie Box Office Factor Analysis
**Course**: DPW1002 · Group 05 · 2026
**Data Source**: MovieLens / TMDB Dataset
**Scripts**: `cleaned_archive/clean_data.py` + one dedicated cleaning script per source CSV

---

## 1. Overview

This report documents the data cleaning procedures applied to five raw CSV files sourced from the MovieLens/TMDB dataset. The goal is to produce analysis-ready datasets for downstream tasks including box office factor analysis, correlation modelling, and machine learning regression. All cleaned files are saved under `cleaned_archive/`.

| Input File              | Output File(s)                                                      | Primary Purpose        |
| ----------------------- | ------------------------------------------------------------------- | ---------------------- |
| `movies_metadata.csv` | `movies_metadata_cleaned.csv`, `movies_metadata_box_office.csv` | Main film metadata     |
| `ratings_small.csv`   | `ratings_small_cleaned.csv`                                       | User ratings           |
| `keywords.csv`        | `keywords_cleaned.csv`                                            | Plot keywords          |
| `credits.csv`         | `credits_cleaned.csv`                                             | Cast & crew            |
| `links_small.csv`     | `links_small_cleaned.csv`                                         | TMDB / IMDB ID mapping |

---

## 2. Dataset Description

The raw dataset contains metadata for approximately **45,000 movies** and **100,000 user ratings** collected from TMDB and GroupLens respectively. Key fields include budget, revenue, release date, genres, cast, crew, plot keywords, user ratings, and production details. Several columns store structured data as **stringified JSON**, which requires programmatic parsing before analysis.

---

## 3. Cleaning Procedures

### 3.1 `movies_metadata.csv` → Main Metadata Table

This is the central table and required the most extensive cleaning across multiple dimensions.

#### 3.1.1 Malformed Row Removal

The raw file contained row-offset errors where the `id` field held date strings instead of integer IDs (a known artefact from the original CSV export). All rows where `id` could not be parsed as an integer were removed before any further processing.

#### 3.1.2 Deduplication

Duplicate entries were identified and removed on two keys independently:

- `id` (TMDB movie ID)
- `imdb_id` (IMDb movie ID)

This ensures a unique one-row-per-movie structure throughout the analysis.

#### 3.1.3 `adult` Column Standardisation

The `adult` column contained raw string values (`"True"` / `"False"`) rather than proper Boolean types. Rows with any value other than these two strings (a sign of further row-offset corruption) were dropped, and the remaining values were cast to Python `bool`.

#### 3.1.4 Numeric Type Conversion

- **`budget`**: Converted to `int64` via `pd.to_numeric`; non-parseable values coerced to `0`.
- **`revenue`**: Same treatment as `budget`.
- **`popularity`**: Converted to `float64`; missing values imputed with the **median** of the column to avoid skewing the distribution.
- **`vote_average`**: Converted to `float64`.
- **`vote_count`**: Converted to `int64`; missing values filled with `0`.

#### 3.1.5 Temporal Processing

`release_date` was parsed to `datetime` using `pd.to_datetime` with `errors="coerce"`. A new derived column `release_year` (integer) was extracted for time-series analysis. Rows with an unparseable `release_date` were dropped entirely, as year information is essential for trend analysis.

#### 3.1.6 Outlier Detection — Runtime

Movies with `runtime` equal to `0` were removed as they represent missing or invalid records. Outlier bounds were then computed using the **Interquartile Range (IQR)** method (k = 1.5) on the remaining positive values. The lower bound was further clamped to a minimum of **20 minutes** to exclude short films and advertisements that are not comparable to feature films. A boolean flag column `runtime_outlier` was added rather than deleting flagged rows, preserving data for completeness checks.

$$
\text{lower} = \max\left(Q_1 - 1.5 \times IQR,\ 20\right), \quad \text{upper} = Q_3 + 1.5 \times IQR
$$

#### 3.1.7 Outlier Detection — Budget & Revenue

For `budget` and `revenue`, a **looser IQR threshold (k = 3.0)** was applied exclusively to the positive-value subset, avoiding penalising the large portion of films with unknown financials (recorded as `0`). Boolean flag columns `budget_outlier` and `revenue_outlier` were added. No rows were deleted at this stage to retain flexibility for analysts.

#### 3.1.8 Zero-Rating Row Removal

Rows where both `vote_average == 0` and `vote_count == 0` were removed, as they carry no audience signal and would distort any rating-based analysis.

#### 3.1.9 JSON Column Parsing

Four columns stored data as stringified Python/JSON lists of dictionaries and were parsed with `ast.literal_eval`:

| Raw Column               | Parsed Output Column          | Extracted Key |
| ------------------------ | ----------------------------- | ------------- |
| `genres`               | `genres_list`               | `name`      |
| `production_companies` | `production_companies_list` | `name`      |
| `production_countries` | `production_countries_list` | `name`      |
| `spoken_languages`     | `spoken_languages_list`     | `name`      |

Additionally, `belongs_to_collection` was converted to a Boolean flag `is_collection` (True if the film belongs to a franchise/series).

#### 3.1.10 Column Pruning and Reordering

Columns irrelevant to the analysis (`homepage`, `poster_path`, `tagline`, and the original raw JSON columns) were dropped. Remaining columns were reordered with financial and performance metrics (`id`, `title`, `budget`, `revenue`, `runtime`, etc.) placed first for readability.

#### 3.1.11 Box Office Subset Export

A secondary filtered file `movies_metadata_box_office.csv` was exported retaining only rows where **both `budget > 0` and `revenue > 0`**. This subset provides a clean sample for ROI and profitability modelling where complete financial data is available.

---

### 3.2 `ratings_small.csv` → User Ratings Table

This file contains 100,000 ratings from 700 users across 9,000 movies.

#### 3.2.1 Deduplication

Duplicate entries on the composite key `(userId, movieId)` were removed to ensure each user rates each film at most once.

#### 3.2.2 Rating Range Validation

MovieLens uses a half-star scale from **0.5 to 5.0** in increments of 0.5. Any rating outside this discrete set was treated as erroneous and removed.

#### 3.2.3 Timestamp Conversion

The Unix `timestamp` column was converted to a human-readable `rating_datetime` column (`datetime64`) for temporal analysis.

---

### 3.3 `keywords.csv` → Plot Keywords Table

This file links movie IDs to plot keyword lists encoded as stringified JSON.

#### 3.3.1 ID Cleaning and Deduplication

`id` was coerced to integer; rows with non-numeric IDs and duplicate IDs were removed.

#### 3.3.2 JSON Expansion (Wide → Long)

The `keywords` column (a JSON list of `{id, name}` objects) was parsed and **exploded** into a long-format table with one row per keyword per movie. The resulting columns are:

| Column           | Description          |
| ---------------- | -------------------- |
| `id`           | TMDB movie ID        |
| `keyword_id`   | Keyword numeric ID   |
| `keyword_name` | Keyword string label |

Rows with missing `keyword_id` or `keyword_name` after parsing were dropped.

---

### 3.4 `credits.csv` → Cast & Crew Table

This file encodes full cast and crew lists as stringified JSON per movie.

#### 3.4.1 ID Cleaning and Deduplication

Same procedure as `keywords.csv`: `id` coerced to integer, rows with null or duplicate IDs removed.

#### 3.4.2 Cast Extraction

The `cast` JSON array was parsed and the **top 3 billed cast members** (by order in the list) were extracted to a `cast_list` column. The total cast size was recorded in `cast_count` for use as a proxy variable in regression models.

#### 3.4.3 Crew Role Extraction

From the `crew` JSON array, personnel were filtered by `job` field:

- **`directors`**: job == `"Director"`
- **`writers`**: job in `{"Screenplay", "Writer", "Story"}`

The raw `cast` and `crew` columns were dropped after extraction.

---

### 3.5 `links_small.csv` → ID Mapping Table

This file maps MovieLens `movieId` to TMDB `tmdbId` for joining datasets.

#### 3.5.1 Numeric Conversion and Deduplication

Both `movieId` and `tmdbId` were converted to numeric types. Rows with nulls in either key were dropped, and duplicates on `movieId` were removed to ensure a clean one-to-one mapping.

---

## 4. Summary of Results

| File                               | Raw Rows (approx.) | Action                                    | Cleaned Rows (approx.) |
| ---------------------------------- | ------------------ | ----------------------------------------- | ---------------------- |
| `movies_metadata.csv`            | 45,466             | Dedup, type fix, outlier flag, JSON parse | ~44,500                |
| `movies_metadata_box_office.csv` | —                 | Filtered subset (budget>0 & revenue>0)    | ~5,000                 |
| `ratings_small.csv`              | 100,004            | Dedup, range validation, timestamp        | ~100,000               |
| `keywords.csv`                   | 46,419             | ID clean, JSON explode (long format)      | ~280,000+              |
| `credits.csv`                    | 45,476             | ID clean, JSON role extraction            | ~45,400                |
| `links_small.csv`                | 9,125              | Numeric cast, dedup                       | ~9,000                 |

> Exact row counts are printed to console during script execution.

---

## 5. Design Decisions

| Decision                               | Rationale                                                                                                                                                                        |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Flag outliers instead of deleting them | Downstream analysts can decide whether to include or exclude outlier films; removing them unconditionally may hide important patterns (e.g., blockbusters with extreme budgets). |
| Separate box office subset file        | Many films lack financial data. Keeping a dedicated `_box_office` file prevents `revenue=0` records from diluting ROI calculations.                                          |
| IQR k=3.0 for budget/revenue           | Financial data is heavily right-skewed. A tighter k=1.5 would flag legitimate blockbusters as outliers; k=3.0 catches only extreme data-entry errors.                            |
| Median imputation for `popularity`   | Mean imputation would be distorted by the long tail of high-popularity outliers; median is more robust for skewed distributions.                                                 |
| Long-format expansion for keywords     | Enables direct SQL-style filtering and aggregation (e.g., "top 10 keywords by average revenue") without further parsing in analysis notebooks.                                   |

---

## 6. Tools and Environment

- **Python** 3.x
- **pandas** — DataFrame operations, type conversion, deduplication
- **NumPy** — IQR computation, type casting
- **ast** — Safe parsing of stringified JSON/Python literals

---

*Report generated based on the cleaning scripts under `cleaned_archive/`, with `clean_data.py` as the orchestration entry point.*
