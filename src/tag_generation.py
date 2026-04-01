# src/tag_generation.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"


def generate_tags(text_series, top_n=30):
    vectorizer = CountVectorizer(max_features=500, ngram_range=(1, 2), stop_words="english")
    X = vectorizer.fit_transform(text_series)
    terms = np.array(vectorizer.get_feature_names_out())
    counts = np.asarray(X.sum(axis=0)).flatten()

    tag_df = pd.DataFrame({
        "tag": terms,
        "score": counts
    }).sort_values(by="score", ascending=False)

    return tag_df.head(top_n)


def main():
    df = pd.read_csv(OUTPUT_DIR / "cleaned_reviews.csv")

    # Overall tags
    overall_tags = generate_tags(df["clean_text"], top_n=50)
    overall_tags["rank"] = range(1, len(overall_tags) + 1)
    overall_tags.to_csv(OUTPUT_DIR / "tag_rankings.csv", index=False)

    # Sentiment-wise tags
    for label in ["positive", "neutral", "negative"]:
        subset = df[df["sentiment"] == label]
        if len(subset) > 0:
            tags = generate_tags(subset["clean_text"], top_n=20)
            tags["rank"] = range(1, len(tags) + 1)
            tags.to_csv(OUTPUT_DIR / f"{label}_tags.csv", index=False)

    print("Saved tag files in outputs/")


if __name__ == "__main__":
    main()