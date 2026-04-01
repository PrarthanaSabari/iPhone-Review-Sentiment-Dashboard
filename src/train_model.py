# src/train_model.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

EXPECTED_CLASSES = ["negative", "neutral", "positive"]


def main():
    input_file = OUTPUT_DIR / "cleaned_reviews.csv"

    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}. Run preprocess.py first.")

    if input_file.stat().st_size == 0:
        raise ValueError(f"File is empty: {input_file}")

    df = pd.read_csv(input_file)

    required_cols = ["clean_text", "sentiment", "combined_text", "rating_num"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["clean_text"] = df["clean_text"].fillna("").astype(str).str.strip()
    df["sentiment"] = df["sentiment"].fillna("").astype(str).str.strip().str.lower()

    df = df[df["clean_text"] != ""].copy()
    df = df[df["sentiment"].isin(EXPECTED_CLASSES)].copy()

    if df.empty:
        raise ValueError("No valid data available after cleaning.")

    print("Class distribution:")
    print(df["sentiment"].value_counts())

    if df["sentiment"].nunique() < 2:
        raise ValueError(
            f"Need at least 2 classes for training, found only: {df['sentiment'].unique().tolist()}"
        )

    bow_vectorizer = CountVectorizer(max_features=1000, stop_words="english")
    X_bow = bow_vectorizer.fit_transform(df["clean_text"])

    bow_words = np.array(bow_vectorizer.get_feature_names_out())
    bow_counts = np.asarray(X_bow.sum(axis=0)).flatten()

    bow_df = pd.DataFrame({
        "word": bow_words,
        "count": bow_counts
    }).sort_values(by="count", ascending=False)

    bow_df.to_csv(OUTPUT_DIR / "bow_terms.csv", index=False)

    tfidf_vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words="english")
    X_tfidf = tfidf_vectorizer.fit_transform(df["clean_text"])
    y = df["sentiment"]

    class_counts = y.value_counts()
    use_stratify = class_counts.min() >= 2

    split_kwargs = {"test_size": 0.2, "random_state": 42}
    if use_stratify:
        split_kwargs["stratify"] = y

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X_tfidf, y, df.index, **split_kwargs
    )

    if pd.Series(y_train).nunique() < 2:
        raise ValueError("Training split has less than 2 classes. Need more balanced data.")

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, labels=EXPECTED_CLASSES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=EXPECTED_CLASSES)

    pd.DataFrame(report).transpose().to_csv(OUTPUT_DIR / "classification_report.csv")
    pd.DataFrame(
        cm,
        index=EXPECTED_CLASSES,
        columns=EXPECTED_CLASSES
    ).to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    pred_df = df.loc[test_idx, ["combined_text", "clean_text", "rating_num", "sentiment"]].copy()
    pred_df["predicted_sentiment"] = y_pred
    pred_df.to_csv(OUTPUT_DIR / "sentiment_predictions.csv", index=False)

    svd = TruncatedSVD(n_components=2, random_state=42)
    X_2d = svd.fit_transform(X_tfidf)

    reduced_df = pd.DataFrame({
        "x": X_2d[:, 0],
        "y": X_2d[:, 1],
        "sentiment": y.values
    })
    reduced_df.to_csv(OUTPUT_DIR / "reduced_features.csv", index=False)

    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())

    if len(model.classes_) == 2:
        coef = model.coef_[0]

        top_pos_idx = np.argsort(coef)[-20:]
        top_neg_idx = np.argsort(coef)[:20]

        pd.DataFrame({
            "term": feature_names[top_pos_idx],
            "weight": coef[top_pos_idx]
        }).sort_values("weight", ascending=False).to_csv(
            OUTPUT_DIR / "top_tfidf_positive.csv", index=False
        )

        pd.DataFrame({
            "term": feature_names[top_neg_idx],
            "weight": coef[top_neg_idx]
        }).sort_values("weight", ascending=True).to_csv(
            OUTPUT_DIR / "top_tfidf_negative.csv", index=False
        )

    else:
        for i, cls in enumerate(model.classes_):
            top_idx = np.argsort(model.coef_[i])[-20:]
            top_terms = pd.DataFrame({
                "term": feature_names[top_idx],
                "weight": model.coef_[i][top_idx]
            }).sort_values("weight", ascending=False)
            top_terms.to_csv(OUTPUT_DIR / f"top_tfidf_{cls}.csv", index=False)

    joblib.dump(model, OUTPUT_DIR / "sentiment_model.pkl")
    joblib.dump(tfidf_vectorizer, OUTPUT_DIR / "tfidf_vectorizer.pkl")

    print(f"\nAccuracy: {accuracy:.4f}")
    print("Saved training outputs in outputs/")


if __name__ == "__main__":
    main()