# app.py

import pandas as pd
import streamlit as st
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

st.set_page_config(
    page_title="iPhone Review Sentiment Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(59,130,246,0.20), transparent 28%),
        radial-gradient(circle at top right, rgba(16,185,129,0.18), transparent 25%),
        radial-gradient(circle at bottom left, rgba(139,92,246,0.18), transparent 30%),
        linear-gradient(135deg, #0b1220 0%, #111827 40%, #0f172a 100%);
    color: #f3f4f6;
}

[data-testid="stAppViewContainer"] {
    background: transparent;
}

[data-testid="block-container"] {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
    max-width: 1400px;
}

header[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

section[data-testid="stSidebar"] {
    display: none;
}

.hero {
    padding: 2rem 2rem 1.6rem 2rem;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(255,255,255,0.09), rgba(255,255,255,0.03));
    border: 1px solid rgba(255,255,255,0.10);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    margin-bottom: 1.3rem;
}

.hero h1 {
    font-size: 2.2rem;
    margin-bottom: 0.35rem;
    color: white;
    letter-spacing: -0.5px;
}

.hero p {
    font-size: 1rem;
    color: #d1d5db;
    margin-bottom: 0;
}

.glass-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.04));
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 20px;
    padding: 1rem 1rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.20);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    margin-bottom: 1rem;
}

.metric-box {
    background: linear-gradient(180deg, rgba(30,41,59,0.85), rgba(15,23,42,0.75));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 6px 18px rgba(0,0,0,0.18);
}

.metric-box h3 {
    margin: 0;
    font-size: 0.95rem;
    color: #cbd5e1;
    font-weight: 500;
}

.metric-box h2 {
    margin-top: 0.35rem;
    margin-bottom: 0;
    color: white;
    font-size: 1.7rem;
}

h2, h3 {
    color: #f9fafb !important;
}

p, label, div {
    color: #e5e7eb;
}

.stDataFrame, .stTable {
    border-radius: 16px;
    overflow: hidden;
}

div[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
}

.stAlert {
    border-radius: 16px;
}

hr {
    border-color: rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)

cleaned_file = OUTPUT_DIR / "cleaned_reviews.csv"
bow_file = OUTPUT_DIR / "bow_terms.csv"
reduced_file = OUTPUT_DIR / "reduced_features.csv"
pred_file = OUTPUT_DIR / "sentiment_predictions.csv"
cm_file = OUTPUT_DIR / "confusion_matrix.csv"
report_file = OUTPUT_DIR / "classification_report.csv"

required_files = [cleaned_file, bow_file, reduced_file, pred_file, cm_file, report_file]
missing = [f.name for f in required_files if not f.exists()]

if missing:
    st.error("Missing files: " + ", ".join(missing))
    st.info("Run:\n1. python src/preprocess.py\n2. python src/train_model.py\n3. streamlit run app.py")
    st.stop()

df = pd.read_csv(cleaned_file)
bow_df = pd.read_csv(bow_file)
reduced_df = pd.read_csv(reduced_file)
pred_df = pd.read_csv(pred_file)
cm_df = pd.read_csv(cm_file, index_col=0)
report_df = pd.read_csv(report_file, index_col=0)
top_tfidf_files = list(OUTPUT_DIR.glob("top_tfidf_*.csv"))

selected = option_menu(
    menu_title=None,
    options=["Overview", "Preprocessing", "BoW", "TF-IDF", "Reduction", "Classification", "Tags"],
    icons=["house", "sliders", "list-task", "file-earmark-text", "scatter-chart", "check2-circle", "tags"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {
            "padding": "0!important",
            "background-color": "rgba(255,255,255,0.04)",
            "border": "1px solid rgba(255,255,255,0.08)",
            "border-radius": "16px",
            "margin-bottom": "1rem"
        },
        "icon": {"color": "#93c5fd", "font-size": "18px"},
        "nav-link": {
            "font-size": "15px",
            "font-weight": "500",
            "color": "#e5e7eb",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "rgba(255,255,255,0.06)",
        },
        "nav-link-selected": {
            "background": "linear-gradient(90deg, #2563eb, #06b6d4)",
            "color": "white",
            "border-radius": "12px",
        },
    }
)

st.markdown("""
<div class="hero">
    <h1>iPhone Review Sentiment Analysis Dashboard</h1>
    <p>
        A polished NLP dashboard for review preprocessing, BoW extraction, TF-IDF feature engineering,
        dimensionality reduction, sentiment classification, and tag refinement.
    </p>
</div>
""", unsafe_allow_html=True)

def plot_template(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

if selected == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-box"><h3>Total Reviews</h3><h2>{len(df)}</h2></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><h3>Average Rating</h3><h2>{df["rating_num"].mean():.2f}</h2></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-box"><h3>Sentiment Classes</h3><h2>{df["sentiment"].nunique()}</h2></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-box"><h3>BoW Terms</h3><h2>{len(bow_df)}</h2></div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Sentiment Distribution")
    sentiment_counts = df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["sentiment", "count"]
    fig = px.pie(
        sentiment_counts,
        names="sentiment",
        values="count",
        hole=0.5,
        color="sentiment",
        color_discrete_map={
            "positive": "#22c55e",
            "neutral": "#f59e0b",
            "negative": "#ef4444"
        }
    )
    st.plotly_chart(plot_template(fig), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Sample Reviews")
    st.dataframe(df[["combined_text", "rating_num", "sentiment"]].head(12), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif selected == "Preprocessing":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Cleaned Text Output")
    st.dataframe(df[["combined_text", "clean_text", "rating_num", "sentiment"]].head(15), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    temp_df = df.copy()
    temp_df["text_len"] = temp_df["clean_text"].astype(str).apply(lambda x: len(x.split()))
    fig = px.histogram(temp_df, x="text_len", nbins=30, color_discrete_sequence=["#60a5fa"])
    fig.update_layout(xaxis_title="Words per review", yaxis_title="Count")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Text Length Distribution")
    st.plotly_chart(plot_template(fig), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif selected == "BoW":
    top_bow = bow_df.head(20).sort_values("count", ascending=True)
    fig = px.bar(top_bow, x="count", y="word", orientation="h", color="count", color_continuous_scale="Blues")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Top Bag of Words Terms")
    st.plotly_chart(plot_template(fig), use_container_width=True)
    st.dataframe(bow_df.head(30), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif selected == "TF-IDF":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Important TF-IDF Features")
    if top_tfidf_files:
        selected_file = st.selectbox("Select TF-IDF file", [f.name for f in top_tfidf_files])
        tfidf_df = pd.read_csv(OUTPUT_DIR / selected_file)
        if "term" in tfidf_df.columns and "weight" in tfidf_df.columns:
            fig = px.bar(
                tfidf_df.head(20).sort_values("weight", ascending=True),
                x="weight",
                y="term",
                orientation="h",
                color="weight",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(plot_template(fig), use_container_width=True)
        st.dataframe(tfidf_df.head(25), use_container_width=True)
    else:
        st.warning("No TF-IDF files found.")
    st.markdown('</div>', unsafe_allow_html=True)

elif selected == "Reduction":
    fig = px.scatter(
        reduced_df,
        x="x",
        y="y",
        color="sentiment",
        opacity=0.75,
        color_discrete_map={
            "positive": "#22c55e",
            "neutral": "#f59e0b",
            "negative": "#ef4444"
        }
    )
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("2D Reduced Feature Map")
    st.plotly_chart(plot_template(fig), use_container_width=True)
    st.dataframe(reduced_df.head(20), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif selected == "Classification":
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Confusion Matrix")
        heatmap = go.Figure(data=go.Heatmap(
            z=cm_df.values,
            x=list(cm_df.columns),
            y=list(cm_df.index),
            colorscale="Blues",
            text=cm_df.values,
            texttemplate="%{text}"
        ))
        st.plotly_chart(plot_template(heatmap), use_container_width=True)
        st.dataframe(cm_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Classification Report")
        st.dataframe(report_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Prediction Samples")
    st.dataframe(pred_df.head(20), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif selected == "Tags":
    tags_df = bow_df.head(20).copy()
    tags_df["rank"] = range(1, len(tags_df) + 1)
    tags_df["refined_tag"] = tags_df["word"].str.replace("_", " ").str.title()
    tags_df["tag_type"] = tags_df["count"].apply(
        lambda x: "Primary" if x >= tags_df["count"].quantile(0.75) else "Secondary"
    )

    fig = px.bar(
        tags_df.sort_values("count", ascending=True),
        x="count",
        y="refined_tag",
        orientation="h",
        color="tag_type",
        color_discrete_map={"Primary": "#38bdf8", "Secondary": "#64748b"}
    )

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Ranked and Refined Tags")
    st.plotly_chart(plot_template(fig), use_container_width=True)
    st.dataframe(tags_df[["rank", "word", "refined_tag", "count", "tag_type"]], use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)