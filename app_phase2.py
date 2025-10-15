from googletrans import Translator
from fpdf import FPDF
from datetime import datetime
import io
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"  
import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from transformers import pipeline
import matplotlib.pyplot as plt
from collections import Counter
import textwrap
from PIL import Image
# Put at the very top (after imports), before any st.* calls
st.set_page_config(
    page_title="Brand Pulse",
    page_icon="assets/brand_pulse_logo.png",
    layout="wide"
)

@st.cache_resource
def get_model():
    """Load Hugging Face sentiment analysis model once per session"""
    from transformers import pipeline
    return pipeline("sentiment-analysis")
# -------------------------
# Session-state initialization
# -------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# -------------------- Translation Helper --------------------
@st.cache_resource
def get_translator():
    return Translator()

def translate_to_english(text):
    try:
        translator = get_translator()
        result = translator.translate(text, dest='en')
        return result.text
    except Exception as e:
        return text  # fallback: return original if translation fails
    
@st.cache_resource
def get_summarizer():
    # much faster summarizer
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",   # ~300 MB
        framework="pt"                           # ensures PyTorch backend
    )

# -------------------------
# Prep / Downloads
# -------------------------
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
# ---------- HEADER: Brand Pulse Logo + Title ----------

# Load logo
logo = Image.open("assets/brand_pulse_logo.png")

# Create two columns — logo (left) and title (right)
# ---- HEADER ----
# ---- HEADER ----
col1, col2 = st.columns([0.15, 0.85])
with col1:
    st.image("assets/brand_pulse_logo.png", use_container_width=True)
with col2:
    st.markdown("""
        <div class="header-text">
            <h1>💡 Brand Pulse</h1>
            <h4>AI-powered LinkedIn Sentiment & Engagement Analyzer</h4>
        </div>
    """, unsafe_allow_html=True)

# ---- FINAL HEADER FIX ----
st.markdown("""
    <style>
        /* Adjust main layout spacing */
        div.block-container {
            padding-top: 1.5rem !important;   /* Slight gap from very top */
        }

        /* Center header vertically with logo */
        .header-text {
            padding-top: 1.1rem;              /* ✅ pushes the text slightly lower */
        }

        /* Title style */
        .header-text h1 {
            margin: 0;
            color: #0077B5;
            font-weight: 700;
            font-size: 2.2rem;
        }

        /* Subtitle style */
        .header-text h4 {
            margin-top: 0.4rem;
            color: grey;
            font-weight: 500;
        }
    </style>
""", unsafe_allow_html=True)


# -------------------------
# Analysis button
# -------------------------
st.write("---")
st.write("###  Run sentiment analysis")
if st.session_state.df is None:
    st.info("Upload a CSV or load the sample dataset from the sidebar, then click Run.")
else:
    if st.button("Run Sentiment Analysis", key="run_analysis"):
        df = st.session_state.df.copy()

        # require 'post' column
        if "post" not in df.columns:
            st.error("Your dataset must contain a 'post' column with text to analyze.")
        else:
            with st.spinner("Loading sentiment model and analyzing posts..."):
                model = get_model()
                # Run predictions (batch)
                # Step 1: Translate posts to English before sentiment analysis
                df["translated_post"] = df["post"].apply(lambda x: translate_to_english(str(x)))

                # Step 2: Run sentiment model on translated text
                results = [model(str(t))[0] for t in df["translated_post"].astype(str)]

                df["sentiment"] = [r["label"] for r in results]
                df["score"] = [r["score"] for r in results]
                # ---- Emotion Detection (on translated text) ----
                @st.cache_resource
                def get_emotion_model():
                    from transformers import pipeline
                    return pipeline(
                        "text-classification",
                        model="j-hartmann/emotion-english-distilroberta-base",
                        return_all_scores=False
                    )

                emotion_model = get_emotion_model()
                emotion_results = [emotion_model(str(t))[0] for t in df["translated_post"].astype(str)]
                df["emotion"] = [r["label"] for r in emotion_results]


                # convert date if exists
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")

                # numeric sentiment value and weighted score
                df["sentiment_value"] = df["sentiment"].apply(lambda x: 1 if str(x).upper() == "POSITIVE" else -1)
                df["weighted_score"] = df["sentiment_value"] * df["score"]

                # store
                st.session_state.df = df
                st.session_state.analysis_done = True

            st.success("✅ Sentiment analysis complete!")

# -------------------------
# Page header / styling
# -------------------------

# ====== Global UI CSS (cards, padding, palette) ======
st.markdown(
    """
    <style>
    :root{
        --brand-blue: #0077B5;
        --soft-blue: #E6F2FA;
        --card-bg: #ffffff;
        --muted: #6E6E6E;
    }

    /* page padding */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }

    /* main title style (if using markdown title) */
    .main-title {
        font-size: 34px;
        font-weight: 700;
        color: var(--brand-blue);
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 15px;
        color: var(--muted);
        margin-top: 0;
        margin-bottom: 12px;
    }

    /* Card style used for panels */
    .bp-card {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 6px 18px rgba(14, 30, 37, 0.06);
        margin-bottom: 18px;
    }

    /* Smaller card (for small metric groups) */
    .bp-small-card {
        background: var(--card-bg);
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 4px 10px rgba(14, 30, 37, 0.04);
    }

    /* Sidebar tweaks */
    .css-1d391kg { /* container class can vary across versions - this is best-effort */
        padding-top: 20px;
    }

    /* Footer text style */
    .footer { font-size: 13px; color: #777; margin-top: 18px; text-align:center; }
    /* Set card background for dark theme */
[data-theme="dark"] .bp-card {
    background-color: #1E1E1E !important;
    color: #EAEAEA !important;
}


    /* Responsive tweaks for small screens */
    @media (max-width: 768px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .main-title { font-size: 28px; }
    }

    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
    <style>
        :root {
    --brand-blue: #0077B5;
    --brand-light: #E6F2FA;
    --brand-dark: #1E1E1E;
}   
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar: data source
# -------------------------
st.sidebar.header("🧭 Controls (Data)")
data_choice = st.sidebar.radio(
    "Choose your action",
    ["Use sample dataset (example)", "Upload CSV"],
    index=0,
    key="data_choice"
)

# Load sample data helper
@st.cache_data
def load_sample_data():
    data = {
        "date": [
            "2025-10-01", "2025-10-03", "2025-10-08",
            "2025-10-10", "2025-10-11", "2025-10-12"
        ],
        "post": [
            "Proud to launch our new EV model!",
            "Facing delays in delivery due to supply issues.",
            "Honored to receive the Innovation Award 2025!",
            "Customer feedback has been amazing so far!",
            "We are sorry for the inconvenience caused by the delay.",
            "Excited about our partnership with Tesla!"
        ],
        "likes": [320, 150, 420, 380, 90, 500],
        "comments": [54, 78, 97, 66, 22, 134]
    }
    return pd.DataFrame(data)
# --- Force rerun logic ---
if st.session_state.get("sample_loaded", False) and st.session_state.df is not None:
    st.session_state.analysis_done = False

# Handle data input
uploaded_file = None
if data_choice == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your LinkedIn CSV (must contain a 'post' column, optional: date/likes/comments)", type=["csv"], key="uploader")
    if uploaded_file is not None:
        try:
            df_temp = pd.read_csv(uploaded_file)
            st.session_state.df = df_temp
            st.success(f"Uploaded {len(df_temp)} rows.")
            st.session_state.analysis_done = False
        except Exception as e:
            st.sidebar.error(f"Failed to read CSV: {e}")
else:
    # sample dataset
    if st.sidebar.button("Load sample dataset", key="load_sample_btn"):
        st.session_state.df = load_sample_data()
        st.session_state.analysis_done = False
        st.sidebar.success("Sample dataset loaded.")

# If session has data, show a small preview in sidebar
if st.session_state.df is not None:
    st.sidebar.markdown(f"**Rows:** {len(st.session_state.df)}")
    if "post" in st.session_state.df.columns:
        st.sidebar.markdown("`post` column: ✅")
    else:
        st.sidebar.markdown("`post` column: ❌ (required)")


# ---------- SHOW DASHBOARD ONLY AFTER ANALYSIS ----------
if "df" in st.session_state and st.session_state.df is not None:
    df = st.session_state.df

    st.markdown("---")
    st.markdown("## 📊 Dashboard & Interactive Insights")

# -------------------------
# Dashboard & Filters
# -------------------------
st.write("---")

# Only proceed if analysis has been done (or df present with sentiment)
if st.session_state.df is None or not st.session_state.analysis_done:
    st.info("Run sentiment analysis first (or load sample dataset and click 'Run Sentiment Analysis').")
    st.stop()

# Use the analyzed dataframe
df = st.session_state.df.copy()

# Sidebar Filters (these are outside the Run button and will re-render)
st.sidebar.header("🔍 Filters & Options")

# Sentiment multiselect - key unique
sentiment_options = sorted(df["sentiment"].astype(str).str.upper().unique().tolist())
sentiment_selected = st.sidebar.multiselect("Sentiment", options=["ALL"] + sentiment_options, default=["ALL"], key="filter_sentiment")

# Date range
if "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["date"]):
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="filter_date")
else:
    date_range = None

# Sort option
sort_option = st.sidebar.selectbox("Sort by", options=["Default", "Most Likes", "Most Comments", "Highest Score"], key="filter_sort")

# Apply filters to produce filtered_df
filtered_df = df.copy()

# Sentiment filter
if sentiment_selected and "ALL" not in sentiment_selected:
    # normalize and filter
    filtered_df["sentiment"] = filtered_df["sentiment"].astype(str).str.upper()
    filtered_df = filtered_df[filtered_df["sentiment"].isin(sentiment_selected)]

# Date filter
if date_range is not None and isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_df = filtered_df[(filtered_df["date"] >= start_date) & (filtered_df["date"] <= end_date)]

# Sorting
if sort_option == "Most Likes" and "likes" in filtered_df.columns:
    filtered_df = filtered_df.sort_values(by="likes", ascending=False)
elif sort_option == "Most Comments" and "comments" in filtered_df.columns:
    filtered_df = filtered_df.sort_values(by="comments", ascending=False)
elif sort_option == "Highest Score" and "score" in filtered_df.columns:
    filtered_df = filtered_df.sort_values(by="score", ascending=False)

# Safety: if empty, show info
if filtered_df.empty:
    st.warning("No posts match your filter criteria. Try changing filters.")
    # still show a small empty table and stop further visuals
    st.dataframe(filtered_df.head(0))
    st.stop()


# -------------------------
# Summary metrics (based on filtered_df)
# -------------------------
st.subheader("Summary Metrics")
avg_confidence = filtered_df["score"].mean() * 100 if "score" in filtered_df.columns else 0
pos_pct = (filtered_df["sentiment"].str.upper() == "POSITIVE").sum() / len(filtered_df) * 100
neg_pct = (filtered_df["sentiment"].str.upper() == "NEGATIVE").sum() / len(filtered_df) * 100

c1, c2, c3 = st.columns(3)
c1.metric("Avg Sentiment Confidence", f"{avg_confidence:.1f}%")
c2.metric("Positive", f"{pos_pct:.1f}%")
c3.metric("Negative", f"{neg_pct:.1f}%")

# -------------------- PDF Export Helper --------------------
def generate_pdf_report(filtered_df, avg_sent, pos_count, neg_count, ai_summary_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Header
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(0, 102, 204)
    pdf.cell(0, 12, "Brand Pulse: AI Sentiment Report", ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(10)

    # Metrics
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, " Summary Metrics", ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Total Posts: {len(filtered_df)}", ln=True)
    pdf.cell(0, 8, f"Positive: {pos_count}", ln=True)
    pdf.cell(0, 8, f"Negative: {neg_count}", ln=True)
    pdf.cell(0, 8, f"Average Sentiment Score: {avg_sent:.2f}", ln=True)
    pdf.ln(8)

    # Add Sentiment Chart
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, " Sentiment Distribution", ln=True)
    pdf.ln(4)
    buf = io.BytesIO()
    fig = px.pie(filtered_df, names="sentiment", title="Sentiment Distribution")
    fig.write_image(buf, format="png")
    buf.seek(0)
    pdf.image(buf, x=20, w=170)
    pdf.ln(10)

    # Add Emotion Chart if available
    if "emotion" in filtered_df.columns:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, " Emotion Distribution", ln=True)
        pdf.ln(4)
        buf2 = io.BytesIO()
        fig2 = px.bar(
            filtered_df,
            x="emotion",
            color="emotion",
            title="Emotion Distribution"
        )
        fig2.write_image(buf2, format="png")
        buf2.seek(0)
        pdf.image(buf2, x=20, w=170)
        pdf.ln(10)

    # Top Keywords
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, " Top Keywords", ln=True)
    pdf.set_font("Helvetica", "", 12)
    text = " ".join(filtered_df["post"].astype(str)).lower()
    words = text.split()
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    top10 = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
    for word, count in top10:
        pdf.cell(0, 8, f"{word} ({count})", ln=True)
    pdf.ln(10)

    # AI Summary
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, " AI Summary Insights", ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(0, 8, ai_summary_text if ai_summary_text else "No AI Summary available.")
    pdf.ln(8)

    # Footer
    pdf.set_y(-25)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 8, "© 2025 Brand Pulse Analytics | Built with Streamlit & Hugging Face", ln=True, align="C")

    # Save file
    filename = f"BrandPulse_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(filename)
    return filename

# ---------- TABS LAYOUT ----------
tab_overview, tab_emotion, tab_wordcloud, tab_top, tab_trend, tab_compare, tab_ai = st.tabs(
    ["📊 Overview", "🎭 Emotion", "☁️ WordCloud", "🔝 Top Posts", "📈 Trends", "🏷️ Brand Comparison", "🧠 Executive Summary"]
)

# ---------------- Tab: Overview ----------------
with tab_overview:
    
    st.subheader("Overview Dashboard")

    # (Then keep the rest of your overview visuals below this)

    # metrics row
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Sentiment Confidence", f"{avg_confidence:.1f}%")
    c2.metric("Positive", f"{pos_pct:.1f}%")
    c3.metric("Negative", f"{neg_pct:.1f}%")

    st.subheader("Sentiment Distribution")
    fig = px.pie(filtered_df, names="sentiment", title="Sentiment Split")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Emotion distribution
# Emotion distribution
with tab_emotion:
    st.subheader("🎭 Emotion Detection")

    # Count emotion occurrences
    emotion_counts = df['emotion'].value_counts().reset_index()
    emotion_counts.columns = ['emotion', 'count']

    # Plot emotion distribution
    fig_emotion = px.bar(
        emotion_counts,
        x='emotion',
        y='count',
        color='emotion',
        title="🎭 Emotion Distribution",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig_emotion, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Tab: WordCloud ----------------
with tab_wordcloud:
    st.subheader("Topic Keywords Wordcloud")
    all_text = " ".join(filtered_df["post"].astype(str)).lower()
    stop_words = set(stopwords.words("english"))
    filtered_words = " ".join([w for w in all_text.split() if w not in stop_words and len(w) > 1])
    wc = WordCloud(width=900, height=300, background_color="white").generate(filtered_words)
    fig_wc, ax = plt.subplots(figsize=(10, 3))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig_wc)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Tab: Top Posts ----------------
with tab_top:
    st.subheader("Top Positive & Negative Posts")
    top_pos = filtered_df.sort_values(by="score", ascending=False).head(3)
    top_neg = filtered_df.sort_values(by="score", ascending=True).head(3)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 😊 Top Positive")
        for _, r in top_pos.iterrows():
            st.write(r["post"])
            st.caption(f"Score: {r['score']:.2f} | Likes: {r.get('likes', 'N/A')} | Comments: {r.get('comments', 'N/A')}")
    with col2:
        st.markdown("### 😞 Top Negative")
        for _, r in top_neg.iterrows():
            st.markdown(f"**Original:** {r['post']}")
            st.caption(f"**Translated:** {r['translated_post']}")

            st.caption(f"Score: {r['score']:.2f} | Likes: {r.get('likes', 'N/A')} | Comments: {r.get('comments', 'N/A')}")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Tab: Trends ----------------
with tab_trend:
    st.subheader("📈 Sentiment & Emotion Trends")

    if "date" in filtered_df.columns:
        # --- Convert to proper datetime format ---
        filtered_df["date"] = pd.to_datetime(filtered_df["date"], errors="coerce")
        filtered_df["date_only"] = filtered_df["date"].dt.date

        # --- Dropdown for filtering sentiment type ---
        trend_filter = st.selectbox(
            "Select Sentiment View",
            options=["All", "Positive", "Negative"],
            index=0,
            key="trend_filter"
        )

        df_trend = filtered_df.copy()
        if trend_filter == "Positive":
            df_trend = df_trend[df_trend["sentiment"].str.upper() == "POSITIVE"]
        elif trend_filter == "Negative":
            df_trend = df_trend[df_trend["sentiment"].str.upper() == "NEGATIVE"]

        # --- Sentiment Trend over Time ---
        trend = (
            df_trend.groupby("date_only")["weighted_score"]
            .mean()
            .reset_index()
            .rename(columns={"weighted_score": "avg_sentiment"})
        )

        fig = px.line(
            trend,
            x="date_only",
            y="avg_sentiment",
            title=f"{trend_filter} Sentiment Trend Over Time",
            markers=True
        )
        fig.update_traces(line=dict(width=3))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Average Sentiment (-1 to +1)",
            template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white",
            hovermode="x unified",
            showlegend=False
        )

        # --- Add average line ---
        avg_line = trend["avg_sentiment"].mean()
        fig.add_hline(
            y=avg_line,
            line_dash="dot",
            line_color="lightblue",
            annotation_text=f"Avg: {avg_line:.2f}",
            annotation_position="bottom right"
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- Optional Emotion Trend ---
        if "emotion" in df_trend.columns:
            st.markdown("#### 🎭 Emotion Over Time")
            emotion_trend = (
                df_trend.groupby(["date_only", "emotion"])
                .size()
                .reset_index(name="count")
            )
            fig_emotion = px.line(
                emotion_trend,
                x="date_only",
                y="count",
                color="emotion",
                markers=True,
                title="Emotion Frequency Over Time"
            )
            fig_emotion.update_layout(
                xaxis_title="Date",
                yaxis_title="Emotion Count",
                template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
            )
            st.plotly_chart(fig_emotion, use_container_width=True)

    else:
        st.info("No date column available to show trends.")
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- Tab: Brand Comparison ----------------
with tab_compare:
    st.subheader("🏷️ Brand Comparison Dashboard")

    st.write("Compare sentiment and emotion between two different brand datasets.")

    colA, colB = st.columns(2)
    with colA:
        brandA_file = st.file_uploader("Upload Brand A CSV", type=["csv"], key="brandA_file")
        brandA_name = st.text_input("Brand A Name", "Brand A")
    with colB:
        brandB_file = st.file_uploader("Upload Brand B CSV", type=["csv"], key="brandB_file")
        brandB_name = st.text_input("Brand B Name", "Brand B")

    # Check uploads
    if brandA_file is not None and brandB_file is not None:
        try:
            dfA = pd.read_csv(brandA_file)
            dfB = pd.read_csv(brandB_file)

            # Ensure 'post' exists
            if "post" not in dfA.columns or "post" not in dfB.columns:
                st.error("Both CSVs must contain a 'post' column.")
            else:
                with st.spinner("Running sentiment analysis on both brands..."):

                    # ✅ Use the global cached model instead of redefining it
                    model = get_model()

                    # ✅ Translate both datasets before sentiment analysis
                    dfA["translated_post"] = dfA["post"].apply(lambda x: translate_to_english(str(x)))
                    dfB["translated_post"] = dfB["post"].apply(lambda x: translate_to_english(str(x)))

                    # ✅ Run sentiment model on translated text
                    dfA["sentiment"] = [model(str(t))[0]["label"] for t in dfA["translated_post"].astype(str)]
                    dfB["sentiment"] = [model(str(t))[0]["label"] for t in dfB["translated_post"].astype(str)]

                    dfA["score"] = [model(str(t))[0]["score"] for t in dfA["translated_post"].astype(str)]
                    dfB["score"] = [model(str(t))[0]["score"] for t in dfB["translated_post"].astype(str)]

                    # Compute stats
                    posA = (dfA["sentiment"].str.upper() == "POSITIVE").sum()
                    negA = (dfA["sentiment"].str.upper() == "NEGATIVE").sum()
                    posB = (dfB["sentiment"].str.upper() == "POSITIVE").sum()
                    negB = (dfB["sentiment"].str.upper() == "NEGATIVE").sum()

                    avgA = dfA["score"].mean() * 100
                    avgB = dfB["score"].mean() * 100

                st.success("✅ Sentiment comparison complete!")

                # Display metrics
                c1, c2 = st.columns(2)
                c1.metric(f"{brandA_name} - Positive Posts", f"{posA}")
                c1.metric(f"{brandA_name} - Negative Posts", f"{negA}")
                c1.metric(f"{brandA_name} - Avg Confidence", f"{avgA:.1f}%")

                c2.metric(f"{brandB_name} - Positive Posts", f"{posB}")
                c2.metric(f"{brandB_name} - Negative Posts", f"{negB}")
                c2.metric(f"{brandB_name} - Avg Confidence", f"{avgB:.1f}%")

                # Combine and visualize
                dfA["brand"] = brandA_name
                dfB["brand"] = brandB_name
                df_compare = pd.concat([dfA, dfB])

                fig = px.bar(
                    df_compare,
                    x="brand",
                    color="sentiment",
                    title=f"Brand Sentiment Comparison: {brandA_name} vs {brandB_name}",
                    barmode="group"
                )
                st.plotly_chart(fig, use_container_width=True)


                # Optional: Emotion comparison if column exists
                if "emotion" in dfA.columns and "emotion" in dfB.columns:
                    df_emotion = pd.concat([dfA[["brand", "emotion"]], dfB[["brand", "emotion"]]])
                    fig_emotion = px.histogram(
                        df_emotion,
                        x="emotion",
                        color="brand",
                        barmode="group",
                        title="Emotion Distribution Across Brands"
                    )
                    st.plotly_chart(fig_emotion, use_container_width=True)

        except Exception as e:
            st.error(f"Failed to compare brands: {e}")
    else:
        st.info("Upload two brand CSVs to compare sentiment and emotion.")
    

# ---------- AI Summary Helper Functions ----------
from collections import Counter
import textwrap

def _top_keywords(texts, top_k=12):
    words = (" ".join(texts)).lower().split()
    cnt = Counter(words)
    return [w for w, _ in cnt.most_common(top_k)]

def build_summary_context(filtered_df: pd.DataFrame) -> str:
    n_posts = len(filtered_df)
    pos = (filtered_df["sentiment"].astype(str).str.upper() == "POSITIVE").sum()
    neg = (filtered_df["sentiment"].astype(str).str.upper() == "NEGATIVE").sum()
    neu = n_posts - pos - neg

    avg_score = filtered_df.get("weighted_score", pd.Series([0]*n_posts)).mean()
    likes_avg = filtered_df["likes"].mean() if "likes" in filtered_df.columns else None
    comments_avg = filtered_df["comments"].mean() if "comments" in filtered_df.columns else None

    emo_counts = (
        filtered_df["emotion"].astype(str).value_counts().to_dict()
        if "emotion" in filtered_df.columns else {}
    )

    try:
        sw = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        sw = set(stopwords.words("english"))

    tokens = []
    for p in filtered_df["post"].astype(str).tolist():
        tokens.extend([w for w in p.lower().split() if w.isalpha() and w not in sw and len(w) > 2])
    keywords = _top_keywords([" ".join(tokens)], top_k=12)

    tmp = filtered_df.copy()
    if "score" not in tmp.columns:
        tmp["score"] = 0.5
    top_pos_posts = tmp.sort_values("score", ascending=False)["post"].head(2).tolist()
    top_neg_posts = tmp.sort_values("score", ascending=True)["post"].head(2).tolist()

    def clip(txt, n=220):
        return (txt[:n] + "…") if len(txt) > n else txt

    pos_examples = [clip(x) for x in top_pos_posts]
    neg_examples = [clip(x) for x in top_neg_posts]

    lines = [
        f"Total posts: {n_posts}",
        f"Sentiment counts → positive: {pos}, neutral: {neu}, negative: {neg}",
        f"Average weighted sentiment (−1..+1): {avg_score:.3f}",
    ]
    if likes_avg is not None:    lines.append(f"Average likes: {likes_avg:.1f}")
    if comments_avg is not None: lines.append(f"Average comments: {comments_avg:.1f}")
    if emo_counts:
        lines.append("Emotion counts: " + ", ".join(f"{k}:{v}" for k,v in emo_counts.items()))
    if keywords:
        lines.append("Top keywords: " + ", ".join(keywords))
    if pos_examples:
        lines.append("Sample positive posts: " + " | ".join(pos_examples))
    if neg_examples:
        lines.append("Sample negative posts: " + " | ".join(neg_examples))

    context = "\n".join(lines)
    return textwrap.shorten(context, width=2000, placeholder=" ...")

def build_insight_prompt(context: str) -> str:
    return f"""
You are an insights analyst. Given the dataset stats below, write an executive summary for a brand manager.
Be concise, action-oriented, and avoid fluff.

DATA:
{context}

Write:
1) 4-6 bullet insights (what happened, why it matters).
2) 2-3 risks/concerns detected.
3) 3 clear recommendations (what to do next week).
Keep it under 160 words total. No headings, just bullets.
"""

# ---------------- Tab: AI Summary ----------------
with tab_ai:
    st.subheader("🧠 AI-Powered Executive Summary")

    # show quick stats (kept from your original)
    avg_sent = filtered_df["weighted_score"].mean() if "weighted_score" in filtered_df.columns else 0
    pos_count = (filtered_df["sentiment"].astype(str).str.upper() == "POSITIVE").sum()
    neg_count = (filtered_df["sentiment"].astype(str).str.upper() == "NEGATIVE").sum()
    neu_count = len(filtered_df) - pos_count - neg_count

    cols = st.columns(3)
    cols[0].metric("Avg Weighted Sentiment", f"{avg_sent:.2f}")
    cols[1].metric("Positive Posts", f"{pos_count}")
    cols[2].metric("Negative Posts", f"{neg_count}")

    # simple guidance banner (kept, slightly condensed)
    if avg_sent > 0.3:
        st.success("Overall sentiment is broadly positive.")
    elif avg_sent < -0.3:
        st.error("Overall sentiment trends negative — investigate root causes.")
    else:
        st.info("Overall sentiment is mixed.")

    # Generate AI summary
    st.divider()
    st.write("Click to generate an **executive summary** with insights, risks, and recommendations.")
    if st.button("✨ Generate AI Summary", key="btn_ai_summary"):
        try:
            with st.spinner("Thinking through your data..."):
            # load only once per session, but only when this block runs
                if "summarizer" not in st.session_state:
                    st.session_state.summarizer = get_summarizer()

                summarizer = st.session_state.summarizer
                context = build_summary_context(filtered_df)
                prompt  = build_insight_prompt(context)
                summarizer = get_summarizer()
                out = summarizer(
                    prompt,
                    max_length=120,
                    min_length=50,
                    do_sample=False
                )
                text = out[0].get("summary_text", out[0].get("generated_text", "")).strip()
                # ✅ store it for later use
                st.session_state.ai_summary_text = text  
                st.markdown("#### Insight Summary")
                st.markdown(text)

        except Exception as e:
            st.error(f"Failed to generate summary: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

st.divider()
st.subheader("📄 Export Report")

if st.button("💾 Generate PDF Report"):
    try:
        summary_text = st.session_state.get("ai_summary_text", "AI summary not generated yet.")
        file_path = generate_pdf_report(filtered_df, avg_sent, pos_count, neg_count, summary_text)
        with open(file_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Report as PDF",
                data=f,
                file_name=file_path,
                mime="application/pdf"
            )
    except Exception as e:
        st.error(f"Error generating report: {e}")

 # ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        <p>🚀 <strong>Brand Pulse</strong> — Powered by AI Insights</p>
        <p style="font-size:13px; color:#777;">
            Built with ❤️ using Streamlit, Transformers, and Python<br>
            © 2025 Brand Pulse Analytics. All rights reserved.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
