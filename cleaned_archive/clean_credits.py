import pandas as pd

try:
    from .clean_utils import load_csv, safe_parse_list, save_csv
except ImportError:
    from clean_utils import load_csv, safe_parse_list, save_csv


def clean_credits() -> pd.DataFrame:
    df = load_csv("credits.csv")
    original_len = len(df)

    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype(int)
    df = df.drop_duplicates(subset=["id"])
    print(f"  ID cleaning/deduplication: removed {before - len(df)}")

    def top_cast(val, n: int = 3) -> list[str]:
        items = safe_parse_list(val)
        return [item["name"] for item in items[:n] if isinstance(item, dict) and "name" in item]

    def extract_role(val, job_filter: set[str]) -> list[str]:
        items = safe_parse_list(val)
        return [
            item["name"]
            for item in items
            if isinstance(item, dict) and item.get("job") in job_filter
        ]

    df["cast_list"] = df["cast"].apply(top_cast)
    df["cast_count"] = df["cast"].apply(lambda v: len(safe_parse_list(v)))
    df["directors"] = df["crew"].apply(lambda v: extract_role(v, {"Director"}))
    df["writers"] = df["crew"].apply(
        lambda v: extract_role(v, {"Screenplay", "Writer", "Story"})
    )

    df = df.drop(columns=["cast", "crew"])

    print(f"\n  Final row count: {len(df)}  (original: {original_len})")
    save_csv(df, "credits_cleaned.csv")
    return df


def main() -> None:
    clean_credits()


if __name__ == "__main__":
    main()