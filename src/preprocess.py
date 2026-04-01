# src/preprocess.py

import re
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

INPUT_FILE = DATA_DIR / "apple_iphone_11_reviews.csv"
OUTPUT_FILE = OUTPUT_DIR / "cleaned_reviews.csv"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_rating(value):
    match = re.search(r"(\d+(\.\d+)?)", str(value))
    if match:
        return float(match.group(1))
    return None


def map_sentiment(rating):
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    required_cols = ["review_text", "review_title", "review_rating"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["review_text"] = df["review_text"].fillna("").astype(str)
    df["review_title"] = df["review_title"].fillna("").astype(str)

    df["combined_text"] = (df["review_title"] + " " + df["review_text"]).str.strip()

    df["rating_num"] = df["review_rating"].apply(extract_rating)
    df = df.dropna(subset=["rating_num", "combined_text"]).copy()

    df["rating_num"] = df["rating_num"].astype(float)
    df["combined_text"] = df["combined_text"].str.strip()
    df = df[df["combined_text"] != ""].copy()

    df["clean_text"] = df["combined_text"].apply(clean_text)
    df = df[df["clean_text"] != ""].copy()

    df["sentiment"] = df["rating_num"].apply(map_sentiment)

    final_df = df[
        [
            "combined_text",
            "clean_text",
            "rating_num",
            "sentiment",
            "review_title",
            "review_text",
            "product",
            "review_country",
            "reviewed_at",
        ]
    ].copy()

    final_df.to_csv(OUTPUT_FILE, index=False)

    print("Saved:", OUTPUT_FILE)
    print("\nSentiment distribution:")
    print(final_df["sentiment"].value_counts())


if __name__ == "__main__":
    main()