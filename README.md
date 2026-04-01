# iPhone Review Sentiment Analysis Dashboard

This project performs **sentiment analysis on product reviews** and presents the complete NLP workflow in a **Streamlit dashboard**.

The system includes:
- Text preprocessing
- Bag of Words (BoW)
- TF-IDF feature extraction
- Dimensionality reduction
- Sentiment classification
- Tag ranking, refinement, and enrichment
- Interactive dashboard visualization

---

## Project Objective

The goal of this project is to collect user reviews and analyze their opinions as:

- Positive
- Negative
- Neutral

The dashboard visually presents each stage of the NLP pipeline so that the user can understand how raw review text is converted into features, classified, and summarized using ranked tags.

---

## Dataset

This project uses a CSV file containing iPhone reviews.

**Input file name:**
`apple_iphone_11_reviews.csv`

**Important columns used from the dataset:**
- `review_text`
- `review_title`
- `review_rating`

The `review_rating` column is parsed from values like:

`5.0 out of 5 stars`

and converted into numeric ratings for sentiment labeling.

---

## Sentiment Labeling Logic

The sentiment classes are assigned from review ratings as follows:

- Rating <= 2  → Negative
- Rating = 3   → Neutral
- Rating >= 4  → Positive

---

## Folder Structure

```bash
Iphone review sentiment analysis/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── apple_iphone_11_reviews.csv
│
├── src/
│   ├── preprocess.py
│   └── train_model.py
│
└── outputs/
    ├── cleaned_reviews.csv
    ├── bow_terms.csv
    ├── reduced_features.csv
    ├── sentiment_predictions.csv
    ├── confusion_matrix.csv
    ├── classification_report.csv
    ├── sentiment_model.pkl
    ├── tfidf_vectorizer.pkl
    ├── top_tfidf_positive.csv
    ├── top_tfidf_negative.csv
    └── ...
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Plotly
- streamlit-option-menu

These libraries are commonly used for machine learning, text processing, model persistence, and interactive dashboards in Streamlit apps. [web:94][web:100]

---

## Features

### 1. Text Preprocessing
- Combines review title and review text
- Converts text to lowercase
- Removes special characters, symbols, and URLs
- Removes extra spaces
- Creates cleaned review text

### 2. Bag of Words
- Uses `CountVectorizer`
- Extracts frequent terms from cleaned reviews
- Saves top terms into `bow_terms.csv`

### 3. TF-IDF Feature Extraction
- Uses `TfidfVectorizer`
- Extracts important weighted words and phrases
- Saves top class-related terms into output CSV files

### 4. Dimensionality Reduction
- Uses `TruncatedSVD`
- Converts high-dimensional TF-IDF vectors into 2D points
- Helps visualize review clusters by sentiment

### 5. Sentiment Classification
- Uses `LogisticRegression`
- Trains on TF-IDF features
- Predicts sentiment classes
- Generates confusion matrix and classification report

### 6. Tag Ranking, Refinement, and Enrichment
- Uses the most frequent extracted words as tags
- Ranks tags by frequency
- Refines tag names for clean presentation
- Displays primary and secondary tags in the dashboard

### 7. Dashboard
The Streamlit app includes top navigation for:
- Overview
- Preprocessing
- BoW
- TF-IDF
- Reduction
- Classification
- Tags

---

## Installation

Install all required packages using:

```bash
pip install -r requirements.txt
```

If you do not have a `requirements.txt` file, install manually using:

```bash
pip install pandas numpy scikit-learn joblib streamlit plotly streamlit-option-menu
```

---

## How to Run the Project

Run the files in this exact order.

### Step 1: Preprocess the data

```bash
python src/preprocess.py
```

This creates:
- `outputs/cleaned_reviews.csv`

### Step 2: Train the model and generate outputs

```bash
python src/train_model.py
```

### Step 3: Launch the dashboard

```bash
streamlit run app.py
```

Streamlit apps are typically started with `streamlit run <file>` so the app opens in a local browser interface. [web:98][web:92]

---

## Dashboard Sections

### Overview
Displays:
- Total reviews
- Average rating
- Number of sentiment classes
- Sentiment distribution chart
- Sample reviews

### Preprocessing
Displays:
- Original and cleaned review text
- Rating and sentiment label
- Text-length distribution

### BoW
Displays:
- Top frequent words
- BoW term table

### TF-IDF
Displays:
- Important weighted terms
- Class-related feature importance

### Reduction
Displays:
- 2D reduced TF-IDF feature map using SVD

### Classification
Displays:
- Confusion matrix
- Classification report
- Predicted sentiment samples

### Tags
Displays:
- Ranked tags
- Refined tags
- Primary and secondary tag categories

---

## Output Files

| File Name | Description |
|---|---|
| `cleaned_reviews.csv` | Preprocessed review dataset |
| `bow_terms.csv` | Top Bag of Words terms |
| `reduced_features.csv` | 2D reduced TF-IDF representation |
| `sentiment_predictions.csv` | Predicted sentiments for test samples |
| `confusion_matrix.csv` | Confusion matrix of classifier |
| `classification_report.csv` | Precision, recall, F1-score report |
| `sentiment_model.pkl` | Trained Logistic Regression model |
| `tfidf_vectorizer.pkl` | Saved TF-IDF vectorizer |
| `top_tfidf_*.csv` | Top TF-IDF features per class |

---