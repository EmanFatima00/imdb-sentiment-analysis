"""
IMDB Movie Review Sentiment Analysis App
Author: Eman Fatima | BS-AI @ PAF-IAST | ML Intern @ ProSensia
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib, re, os, json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

st.set_page_config(page_title="Movie Sentiment Analyzer", page_icon="🎬", layout="wide")

st.markdown("""
<style>
    .main-header { background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        padding: 2rem; border-radius: 12px; text-align: center; color: white; margin-bottom: 2rem; }
    .main-header h1 { font-size: 2.2rem; margin: 0; }
    .main-header p  { font-size: 1rem; opacity: 0.85; margin: 0.5rem 0 0; }
    .metric-card { background: white; border-radius: 10px; padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;
        border-left: 4px solid #e94560; }
    .metric-card h3 { font-size: 1.8rem; color: #e94560; margin: 0; }
    .metric-card p  { color: #64748B; margin: 0; font-size: 0.9rem; }
    .result-positive { background: linear-gradient(135deg, #D1FAE5, #A7F3D0);
        border: 2px solid #059669; border-radius: 12px; padding: 2rem; text-align: center; margin: 1rem 0; }
    .result-negative { background: linear-gradient(135deg, #FEE2E2, #FECACA);
        border: 2px solid #DC2626; border-radius: 12px; padding: 2rem; text-align: center; margin: 1rem 0; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model   = joblib.load(os.path.join(OUTPUT_DIR, "sentiment_model.pkl"))
    tfidf   = joblib.load(os.path.join(OUTPUT_DIR, "tfidf.pkl"))
    mpath   = os.path.join(OUTPUT_DIR, "metrics.json")
    metrics = json.load(open(mpath)) if os.path.exists(mpath) else {}
    return model, tfidf, metrics

model, tfidf, metrics = load_model()

stop_words = set(stopwords.words('english'))
negation_words = {'no','not','nor','never','neither','nobody','nothing',
                  'nowhere','cannot',"can't","won't","don't","doesn't",
                  "didn't","isn't","aren't","wasn't","weren't"}
stop_words -= negation_words
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text   = re.sub(r'<.*?>', ' ', text)
    text   = re.sub(r'http\S+|www\S+', 'URL', text)
    text   = re.sub(r'[^a-zA-Z\s!?]', ' ', text)
    text   = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens
              if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

def predict(review):
    cleaned = preprocess(review)
    vec     = tfidf.transform([cleaned])
    pred    = model.predict(vec)[0]
    proba   = model.predict_proba(vec)[0]
    return pred, proba, cleaned

st.markdown("""
<div class="main-header">
    <h1>🎬 Movie Review Sentiment Analyzer</h1>
    <p>AI-powered sentiment detection — Logistic Regression + TF-IDF Bigrams | Trained on 49,581 IMDB reviews</p>
</div>""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown(f'<div class="metric-card"><h3>{metrics.get("accuracy","89.95")}%</h3><p>Accuracy</p></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="metric-card"><h3>{metrics.get("roc_auc","0.9655")}</h3><p>ROC-AUC</p></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="metric-card"><h3>{metrics.get("total_reviews","49581"):,}</h3><p>Reviews Trained</p></div>', unsafe_allow_html=True)
with c4: st.markdown('<div class="metric-card"><h3>4</h3><p>Models Compared</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["🎬 Analyze Review", "📊 Model Insights", "ℹ️ About"])

with tab1:
    col_main, col_examples = st.columns([2, 1])

    with col_examples:
        st.markdown("### 🎬 Try these examples")
        examples = {
            "⭐ Great movie":     "This film was absolutely incredible! The performances were outstanding, the story was deeply moving, and the direction was masterful. One of the best films I've ever seen.",
            "⭐ Another positive": "A wonderful cinematic experience. The characters felt real and the emotional depth was extraordinary. I laughed, I cried, and left the theatre feeling inspired.",
            "💔 Bad movie":       "This was a complete waste of time. The plot made no sense, the acting was terrible, and the special effects looked cheap. I walked out halfway through.",
            "💔 Another negative": "Possibly the worst film I've seen this year. Boring, predictable, and completely forgettable. Save your money and watch something else.",
            "😐 Mixed review":    "The film had some interesting ideas but failed to execute them properly. Good cinematography but the script needed a lot more work.",
        }
        for label, review in examples.items():
            if st.button(label, use_container_width=True):
                st.session_state['example_review'] = review

    with col_main:
        st.markdown("### ✍️ Enter a movie review")
        default = st.session_state.get('example_review', '')
        review  = st.text_area("Paste or type a movie review:", value=default, height=180,
                               placeholder="Write your movie review here...")
        col_b1, col_b2 = st.columns([1, 3])
        with col_b1:
            analyze_btn = st.button("🎬 Analyze", type="primary", use_container_width=True)
        with col_b2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        if clear_btn:
            st.session_state['example_review'] = ''
            st.rerun()

    if analyze_btn and review.strip():
        pred, proba, cleaned = predict(review)
        pos_pct = proba[1] * 100
        neg_pct = proba[0] * 100

        st.markdown("---")
        res_col, detail_col = st.columns([1, 1])

        with res_col:
            if pred == 1:
                st.markdown(f"""<div class="result-positive">
                    <h1>😍 POSITIVE</h1>
                    <h2 style="color:#059669;font-size:2.5rem;margin:0">{pos_pct:.1f}%</h2>
                    <p style="color:#064E3B;">Confidence Score</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="result-negative">
                    <h1>😞 NEGATIVE</h1>
                    <h2 style="color:#DC2626;font-size:2.5rem;margin:0">{neg_pct:.1f}%</h2>
                    <p style="color:#7F1D1D;">Confidence Score</p>
                </div>""", unsafe_allow_html=True)

            # Sentiment bar
            fig_b, ax_b = plt.subplots(figsize=(5, 1.8))
            ax_b.barh([''], [pos_pct], color='#059669', height=0.4, label='Positive')
            ax_b.barh([''], [neg_pct], left=[pos_pct], color='#DC2626', height=0.4, label='Negative')
            ax_b.set_xlim(0, 100)
            ax_b.set_title("Sentiment Split", fontweight='bold', fontsize=10)
            ax_b.legend(loc='upper right', fontsize=8)
            if pos_pct > 10:
                ax_b.text(pos_pct/2, 0, f"{pos_pct:.1f}%", ha='center', va='center',
                          color='white', fontweight='bold', fontsize=10)
            if neg_pct > 10:
                ax_b.text(pos_pct + neg_pct/2, 0, f"{neg_pct:.1f}%", ha='center', va='center',
                          color='white', fontweight='bold', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig_b)
            plt.close()

        with detail_col:
            st.markdown("#### 📝 Review Analysis")
            st.markdown(f"**Length:** {len(review)} characters")
            st.markdown(f"**Word count:** {len(review.split())} words")
            st.markdown(f"**Exclamation marks:** {review.count('!')}")
            st.markdown(f"**Question marks:** {review.count('?')}")
            st.markdown(f"**Uppercase ratio:** {sum(c.isupper() for c in review)/max(len(review),1)*100:.1f}%")
            st.markdown("**Cleaned tokens (model input):**")
            st.code(cleaned[:300] + "..." if len(cleaned) > 300 else cleaned)

    elif analyze_btn:
        st.warning("Please enter a movie review to analyze.")

with tab2:
    st.markdown("### 📊 Model Performance & Analysis")
    c1, c2 = st.columns(2)
    with c1:
        for img, cap in [("model_comparison.png","Model Comparison"),
                          ("confusion_matrix.png","Confusion Matrix")]:
            path = os.path.join(OUTPUT_DIR, img)
            if os.path.exists(path): st.image(path, caption=cap, use_container_width=True)
    with c2:
        for img, cap in [("roc_curves.png","ROC Curves"),
                          ("top_sentiment_words.png","Top Sentiment Words")]:
            path = os.path.join(OUTPUT_DIR, img)
            if os.path.exists(path): st.image(path, caption=cap, use_container_width=True)
    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        path = os.path.join(OUTPUT_DIR, "wordclouds.png")
        if os.path.exists(path): st.image(path, caption="Word Clouds — Positive vs Negative", use_container_width=True)
    with c4:
        path = os.path.join(OUTPUT_DIR, "length_distribution.png")
        if os.path.exists(path): st.image(path, caption="Review Length Distribution", use_container_width=True)
    st.markdown("---")
    st.markdown("### 📈 Model Results Summary")
    perf = {
        "Model":     ["Naive Bayes","Logistic Regression ✅","SVM","SGD Classifier"],
        "Accuracy":  ["87.49%","89.95%","89.55%","89.19%"],
        "F1 Score":  ["0.8768","0.9003","0.8962","0.8935"],
        "ROC-AUC":   ["0.9436","0.9655","0.9622","0.9600"],
        "CV Score":  ["86.84%±0.37","89.55%±0.40","89.03%±0.38","88.94%±0.31"],
    }
    st.dataframe(pd.DataFrame(perf), use_container_width=True, hide_index=True)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        ### 🔬 About This Project
        Advanced NLP sentiment analysis trained on 49,581 IMDB movie reviews.

        **Pipeline:**
        - ✅ 4 ML models trained and compared
        - ✅ TF-IDF with bigrams (10,000 features)
        - ✅ HTML tag removal from raw reviews
        - ✅ NLTK lemmatization — negation words preserved
        - ✅ 6 engineered text features
        - ✅ 5-fold stratified cross-validation
        - ✅ Final: Logistic Regression (Accuracy: 89.95%, ROC-AUC: 0.9655)

        **Dataset:** IMDB Movie Reviews (50,000 reviews, perfectly balanced)
        """)
    with c2:
        st.markdown("""
        ### 👩‍💻 Developer
        **Eman Fatima**
        BS Artificial Intelligence — Semester 6 | PAF-IAST

        - 🏢 ML Intern @ ProSensia
        - 🎓 Dean's List — SGPA 3.72
        - 🤖 HR Manager @ CtrlAltCrew

        **Tech Stack:**
        Python · Scikit-learn · NLTK · WordCloud · Streamlit · TF-IDF

        [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/eman-fatima-99962230b)
        [![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/EmanFatima00)
        """)
