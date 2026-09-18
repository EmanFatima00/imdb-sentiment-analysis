# 🎬 IMDB Movie Review Sentiment Analysis

> **Advanced NLP sentiment classifier — 89.9% accuracy on 49,581 real movie reviews**
> Built by Eman Fatima | BS-AI @ PAF-IAST 

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5-orange?style=flat-square)
![NLTK](https://img.shields.io/badge/NLTK-3.8-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 🚀 Live Demo
**[👉 Click here to try the live app](https://imdb-sentiment-analysis-qmtex2cbenfnvlf2m77fqf.streamlit.app/)**

---

## 📌 Project Overview

This project builds an advanced NLP pipeline to classify IMDB movie reviews as
Positive or Negative. The system compares 4 machine learning models using TF-IDF
bigram features, NLTK lemmatization with negation word preservation, and 6 engineered
text features — deployed as an interactive Streamlit web application.

**Key upgrade over basic sentiment analysis:** Negation words like "not", "never",
"don't" are deliberately preserved during preprocessing — critical for sentiment tasks
where "not good" means something very different from "good".

---

## 🏆 Results

| Model                       | Accuracy | F1 Score | ROC-AUC | CV Score         |
|-----------------------------|----------|----------|---------|------------------|
| Naive Bayes                 | 87.49%   | 0.8768   | 0.9436  | 86.84% ± 0.37%  |
| **Logistic Regression ✅**  | **89.95%** | **0.9003** | **0.9655** | **89.55% ± 0.40%** |
| SVM                         | 89.55%   | 0.8962   | 0.9622  | 89.03% ± 0.38%  |
| SGD Classifier              | 89.19%   | 0.8935   | 0.9600  | 88.94% ± 0.31%  |

**Final Model: Logistic Regression | Accuracy: 89.95% | ROC-AUC: 0.9655**

---

## 🔬 NLP Pipeline

```
Raw Reviews (49,581 IMDB reviews)
       ↓
HTML Tag Removal (<br />, <p>, etc.)
       ↓
URL Normalization
       ↓
NLTK Lemmatization
(negation words preserved: not, never, don't...)
       ↓
6 Engineered Text Features
(length, word count, exclamations, uppercase ratio...)
       ↓
TF-IDF Vectorization
(unigrams + bigrams, 10,000 features, sublinear TF)
       ↓
Train/Test Split (80/20, Stratified)
       ↓
4 Models + 5-Fold Cross Validation
       ↓
Best Model → Logistic Regression → Deployed
```

---

## ✨ Key Features

- **4 ML models** trained and compared with cross-validation
- **TF-IDF with bigrams** — captures two-word sentiment patterns
- **HTML tag removal** — IMDB reviews contain raw HTML, cleaned properly
- **Negation word preservation** — "not good", "never boring" handled correctly
- **6 engineered features** — review length, word count, punctuation signals
- **Interactive Streamlit app** with 5 example reviews, confidence bar, review analysis
- **6 publication-quality visualizations** — word clouds, ROC curves, sentiment word analysis

---

## 📊 Dataset

- **Source:** IMDB Dataset of 50K Movie Reviews (Kaggle)
- **Size:** 49,581 reviews (after deduplication)
- **Balance:** Perfectly balanced — 50% Positive / 50% Negative
- **Review length:** Average ~230 words per review

---

## ⚙️ Run Locally

```bash
git clone https://github.com/EmanFatima00/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
pip install -r requirements.txt
python model_training.py
streamlit run app.py
```

---

## 📁 Project Structure

```
imdb-sentiment-analysis/
│
├── app.py                    # Streamlit web application
├── model_training.py         # Full NLP + ML pipeline
├── IMDB Dataset.csv          # Dataset
├── requirements.txt          # Dependencies
├── README.md                 # This file
│
└── outputs/
    ├── sentiment_model.pkl       # Trained Logistic Regression model
    ├── tfidf.pkl                 # TF-IDF vectorizer
    ├── metrics.json              # Model performance metrics
    ├── model_comparison.png      # Model comparison chart
    ├── roc_curves.png            # ROC curves
    ├── confusion_matrix.png      # Confusion matrix
    ├── wordclouds.png            # Positive vs Negative word clouds
    ├── top_sentiment_words.png   # Most influential words
    └── length_distribution.png  # Review length analysis
```

---

## 👩‍💻 Author

**Eman Fatima**
BS Artificial Intelligence — Semester 6 | PAF-IAST, Pakistan

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/eman-fatima-99962230b)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/EmanFatima00)

---

*⭐ Star this repo if you found it useful!*
