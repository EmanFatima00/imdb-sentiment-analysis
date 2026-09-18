"""
IMDB Movie Review Sentiment Analysis — Advanced NLP Pipeline
Author: Eman Fatima | BS-AI @ PAF-IAST | ML Intern @ ProSensia
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re, os, json, joblib, warnings
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)

warnings.filterwarnings('ignore')
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)

# ── Paths ─────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "IMDB Dataset.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  IMDB SENTIMENT ANALYSIS — ADVANCED NLP PIPELINE")
print("=" * 60)

# ── 1. Load Data ──────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df.drop_duplicates(subset='review', inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
df['label'] = df['sentiment'].str.lower().map({'positive': 1, 'negative': 0})

print(f"\n✅ Dataset loaded: {len(df):,} reviews")
print(f"   Positive: {(df['label']==1).sum():,} ({(df['label']==1).mean()*100:.1f}%)")
print(f"   Negative: {(df['label']==0).sum():,} ({(df['label']==0).mean()*100:.1f}%)")

# ── 2. Feature Engineering ────────────────────────────────────
df['review_length']   = df['review'].apply(len)
df['word_count']      = df['review'].apply(lambda x: len(x.split()))
df['exclamation']     = df['review'].apply(lambda x: x.count('!'))
df['question_marks']  = df['review'].apply(lambda x: x.count('?'))
df['uppercase_ratio'] = df['review'].apply(lambda x: sum(c.isupper() for c in x) / max(len(x), 1))
df['has_url']         = df['review'].apply(lambda x: int(bool(re.search(r'http|www', x, re.I))))
print("\n✅ Feature engineering — 6 new features added")

# ── 3. Text Preprocessing ─────────────────────────────────────
stop_words = set(stopwords.words('english'))
# Keep negation words — important for sentiment!
negation_words = {'no', 'not', 'nor', 'never', 'neither', 'nobody', 'nothing',
                  'nowhere', 'cannot', "can't", "won't", "don't", "doesn't",
                  "didn't", "isn't", "aren't", "wasn't", "weren't"}
stop_words -= negation_words
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text   = re.sub(r'<.*?>', ' ', text)           # remove HTML tags
    text   = re.sub(r'http\S+|www\S+', 'URL', text)
    text   = re.sub(r'[^a-zA-Z\s!?]', ' ', text)
    text   = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens
              if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

print("🔄 Preprocessing reviews (this may take ~1 min for 50K reviews)...")
df['clean_review'] = df['review'].apply(preprocess)
print("✅ Preprocessing complete — HTML removed, lemmatized, negations kept")

# ── 4. Train/Test Split ───────────────────────────────────────
X_text = df['clean_review']
y      = df['label']
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, stratify=y, random_state=42)
print(f"\n✅ Split: {len(X_train_text):,} train / {len(X_test_text):,} test")

# ── 5. TF-IDF with Bigrams ────────────────────────────────────
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=3,
    max_df=0.95,
)
X_train = tfidf.fit_transform(X_train_text)
X_test  = tfidf.transform(X_test_text)
print(f"✅ TF-IDF vectorization — {X_train.shape[1]:,} features (unigrams + bigrams)")

# ── 6. Train Models ───────────────────────────────────────────
models = {
    'Naive Bayes':         MultinomialNB(alpha=0.5),
    'Logistic Regression': LogisticRegression(C=5, max_iter=1000, random_state=42),
    'SVM':                 CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000, random_state=42)),
    'SGD Classifier':      SGDClassifier(loss='log_loss', max_iter=100, random_state=42),
}

print("\n" + "=" * 60)
print("  MODEL COMPARISON")
print("=" * 60)

results = {}
cv_skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n⏳ Training {name}...")
    model.fit(X_train, y_train)
    preds     = model.predict(X_test)
    proba     = model.predict_proba(X_test)[:, 1]
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_skf, scoring='accuracy', n_jobs=-1)
    results[name] = {
        'model':     model,
        'accuracy':  accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds),
        'recall':    recall_score(y_test, preds),
        'f1':        f1_score(y_test, preds),
        'roc_auc':   roc_auc_score(y_test, proba),
        'cv_mean':   cv_scores.mean(),
        'cv_std':    cv_scores.std(),
        'preds':     preds,
        'proba':     proba,
    }
    print(f"   Accuracy : {results[name]['accuracy']*100:.2f}%")
    print(f"   F1       : {results[name]['f1']:.4f}")
    print(f"   ROC-AUC  : {results[name]['roc_auc']:.4f}")
    print(f"   CV Score : {results[name]['cv_mean']*100:.2f}% ± {results[name]['cv_std']*100:.2f}%")

# ── 7. Best Model ─────────────────────────────────────────────
best_name = max(results, key=lambda x: results[x]['roc_auc'])
best      = results[best_name]
print(f"\n🏆 Best Model: {best_name}")
print(f"   Accuracy : {best['accuracy']*100:.2f}%")
print(f"   F1 Score : {best['f1']:.4f}")
print(f"   ROC-AUC  : {best['roc_auc']:.4f}")
print(f"\n{classification_report(y_test, best['preds'], target_names=['Negative','Positive'])}")

# ── 8. Save ───────────────────────────────────────────────────
joblib.dump(best['model'], os.path.join(OUTPUT_DIR, "sentiment_model.pkl"))
joblib.dump(tfidf,         os.path.join(OUTPUT_DIR, "tfidf.pkl"))
metrics = {
    "accuracy":   round(best['accuracy']*100, 2),
    "f1":         round(best['f1'], 4),
    "roc_auc":    round(best['roc_auc'], 4),
    "precision":  round(best['precision'], 4),
    "recall":     round(best['recall'], 4),
    "best_model": best_name,
    "total_reviews": len(df),
}
json.dump(metrics, open(os.path.join(OUTPUT_DIR, "metrics.json"), "w"))
print("\n✅ Model, vectorizer and metrics saved to outputs/")

# ── 9. Visualizations ─────────────────────────────────────────
print("\n📊 Generating visualizations...")
COLORS = ['#DC2626', '#059669']

# A. Model Comparison
names_ = list(results.keys())
accs_  = [results[n]['accuracy']*100 for n in names_]
f1s_   = [results[n]['f1']           for n in names_]
aucs_  = [results[n]['roc_auc']      for n in names_]
cols_  = ['#6366F1','#059669','#D97706','#0891B2']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Model Performance Comparison", fontsize=14, fontweight='bold')
for ax, vals, title, xlim in zip(
        axes, [accs_, f1s_, aucs_],
        ["Accuracy (%)","F1 Score","ROC-AUC"],
        [(80, 100),(0.8, 1.0),(0.85, 1.0)]):
    ax.barh(names_, vals, color=cols_)
    ax.set_title(title); ax.set_xlim(*xlim)
    for i, v in enumerate(vals):
        lbl = f"{v:.1f}%" if "Accuracy" in title else f"{v:.4f}"
        ax.text(v+0.001 if "Accuracy" not in title else v+0.1, i, lbl, va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()

# B. ROC Curves
plt.figure(figsize=(8, 6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['proba'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.4f})", linewidth=2)
plt.plot([0,1],[0,1],'gray',linestyle=':')
plt.title("ROC Curves — All Models", fontsize=13, fontweight='bold')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.legend(loc='lower right', fontsize=9); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), dpi=150, bbox_inches='tight')
plt.close()

# C. Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, best['preds'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative','Positive'],
            yticklabels=['Negative','Positive'])
plt.title(f"Confusion Matrix — {best_name}", fontweight='bold')
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150, bbox_inches='tight')
plt.close()

# D. Word Clouds
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, label, title, cmap in zip(
        axes, [1, 0], ['Positive Reviews', 'Negative Reviews'],
        ['Greens', 'Reds']):
    text = ' '.join(df[df['label']==label]['clean_review'].sample(2000, random_state=42))
    wc   = WordCloud(width=600, height=300, background_color='white',
                     colormap=cmap, max_words=100).generate(text)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off'); ax.set_title(title, fontsize=13, fontweight='bold')
plt.suptitle("Most Common Words — Positive vs Negative Reviews", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "wordclouds.png"), dpi=150, bbox_inches='tight')
plt.close()

# E. Review Length Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for col, ax, title in zip(
        ['review_length','word_count'], axes,
        ['Review Length (chars)','Word Count']):
    for label, color, lname in zip([1, 0], COLORS, ['Positive','Negative']):
        data = df[df['label']==label][col]
        ax.hist(data, bins=50, alpha=0.6, color=color, label=lname, density=True)
    ax.set_title(f'{title} Distribution', fontweight='bold')
    ax.set_xlabel(title); ax.set_ylabel('Density'); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "length_distribution.png"), dpi=150, bbox_inches='tight')
plt.close()

# F. Top Sentiment Words (from Logistic Regression coefficients)
lr_model = results['Logistic Regression']['model']
feature_names = tfidf.get_feature_names_out()
coef = lr_model.coef_[0]
top_pos_idx = np.argsort(coef)[-20:]
top_neg_idx = np.argsort(coef)[:20]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].barh(feature_names[top_pos_idx], coef[top_pos_idx], color='#059669')
axes[0].set_title("Top 20 Positive Sentiment Words", fontweight='bold')
axes[1].barh(feature_names[top_neg_idx], coef[top_neg_idx], color='#DC2626')
axes[1].set_title("Top 20 Negative Sentiment Words", fontweight='bold')
plt.suptitle("Logistic Regression — Most Influential Words", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_sentiment_words.png"), dpi=150, bbox_inches='tight')
plt.close()

print("✅ All visualizations saved to outputs/")
print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print(f"  Best Model : {best_name}")
print(f"  Accuracy   : {best['accuracy']*100:.2f}%")
print(f"  F1 Score   : {best['f1']:.4f}")
print(f"  ROC-AUC    : {best['roc_auc']:.4f}")
print("=" * 60)
