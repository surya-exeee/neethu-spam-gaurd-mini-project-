"""
train_model.py
--------------
Loads the SMS Spam Collection dataset, trains two ML models,
picks the best one by spam F1-score, and saves everything to model/.

Run this AFTER you have placed your dataset in the data/ folder.

Supports three file formats automatically:
  1. Kaggle  : data/spam.csv          (columns v1, v2, latin-1 encoding)
  2. UCI     : data/SMSSpamCollection  (tab-separated, no header, utf-8)
  3. Practice: data/sample_sms.csv    (tab-separated, no header, utf-8)

Usage:
    python train_model.py
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR  = "data"
MODEL_DIR = "model"
RANDOM_STATE = 42
TEST_SIZE    = 0.20
TOP_N_WORDS  = 15

CANDIDATE_FILES = [
    ("data/spam.csv",           "kaggle"),
    ("data/SMSSpamCollection",  "uci"),
    ("data/sample_sms.csv",     "uci"),
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    for filepath, fmt in CANDIDATE_FILES:
        if not os.path.exists(filepath):
            continue

        print(f"[DATA] Found dataset: {filepath}  (format: {fmt})")

        if fmt == "kaggle":
            df = pd.read_csv(filepath, encoding="latin-1", usecols=["v1", "v2"])
            df.columns = ["label", "message"]
        else:
            df = pd.read_csv(
                filepath,
                sep="\t",
                header=None,
                names=["label", "message"],
                encoding="utf-8",
            )
        return df

    print("\n[ERROR] No dataset file found in the data/ folder.")
    print("  Option A (practice): python make_sample_data.py")
    print("  Option B (real UCI): https://archive.ics.uci.edu/dataset/228/sms+spam+collection")
    print("  Option C (Kaggle)  : https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset")
    sys.exit(1)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["label", "message"])
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["message"] != ""]
    df = df.drop_duplicates(subset=["label", "message"])
    after = len(df)
    print(f"[CLEAN] Removed {before - after} empty/duplicate rows  ({before} -> {after})")

    df["label"] = df["label"].str.lower().str.strip()

    counts   = df["label"].value_counts()
    total    = len(df)
    spam_pct = counts.get("spam", 0) / total * 100
    ham_pct  = counts.get("ham",  0) / total * 100
    print(f"\n[BALANCE]")
    print(f"  Ham  (legit) : {counts.get('ham',  0):5d}  ({ham_pct:.1f}%)")
    print(f"  Spam         : {counts.get('spam', 0):5d}  ({spam_pct:.1f}%)")
    print(f"  Total        : {total}")
    print(
        "\n  NOTE: Spam is a minority class (~13% in the real dataset).\n"
        "  This is CLASS IMBALANCE. A model that always says 'ham' gets ~87%\n"
        "  accuracy but catches zero spam. That is why we use F1-score, not accuracy."
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE EXTRACTION (TF-IDF)
# ─────────────────────────────────────────────────────────────────────────────
def build_vectorizer(n_docs: int = 9999) -> TfidfVectorizer:
    # min_df=2 is ideal for large datasets (real data has 5000+ messages).
    # For the small practice file we drop it to 1 so the vocabulary is not empty.
    min_df = 1 if n_docs < 500 else 2
    print(f"  min_df set to {min_df} (dataset has {n_docs} messages)")
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=min_df,
        sublinear_tf=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="spam", zero_division=0)
    rec  = recall_score(y_test, y_pred, pos_label="spam", zero_division=0)
    f1   = f1_score(y_test, y_pred, pos_label="spam", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred, labels=["ham", "spam"]).tolist()

    SEP = "-" * 55
    print(f"\n{SEP}")
    print(f"  {name}")
    print(f"{SEP}")
    print(f"  Accuracy          : {acc*100:.2f}%")
    print(f"  Spam Precision    : {prec*100:.2f}%")
    print(f"  Spam Recall       : {rec*100:.2f}%")
    print(f"  Spam F1-Score     : {f1*100:.2f}%")
    print(f"\n  Confusion Matrix  (rows=actual, cols=predicted)")
    print(f"               Pred Ham   Pred Spam")
    print(f"  Actual Ham :  {cm[0][0]:7d}   {cm[0][1]:9d}")
    print(f"  Actual Spam:  {cm[1][0]:7d}   {cm[1][1]:9d}")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"], zero_division=0))

    return {
        "name"     : name,
        "accuracy" : round(acc,  4),
        "precision": round(prec, 4),
        "recall"   : round(rec,  4),
        "f1"       : round(f1,   4),
        "cm"       : cm,
    }


def get_top_spam_words(vectorizer: TfidfVectorizer, model, n: int = TOP_N_WORDS) -> list:
    feature_names = vectorizer.get_feature_names_out()

    if hasattr(model, "coef_"):
        coefs = model.coef_
        if coefs.shape[0] == 1:
            spam_weights = coefs[0]
        else:
            spam_idx = list(model.classes_).index("spam")
            spam_weights = coefs[spam_idx]
    elif hasattr(model, "feature_log_prob_"):
        spam_idx = list(model.classes_).index("spam")
        ham_idx  = list(model.classes_).index("ham")
        spam_weights = model.feature_log_prob_[spam_idx] - model.feature_log_prob_[ham_idx]
    else:
        return []

    positive_mask    = spam_weights > 0
    positive_weights = spam_weights[positive_mask]
    positive_names   = feature_names[positive_mask]
    top_idx          = np.argsort(positive_weights)[::-1][:n]
    top_words        = positive_names[top_idx].tolist()
    return top_words


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  SpamGuard - Model Training Pipeline")
    print("=" * 55)

    df  = load_data()
    df  = clean_data(df)

    X = df["message"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\n[SPLIT] Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    print("\n[TF-IDF] Fitting vectoriser ...")
    vectorizer  = build_vectorizer(n_docs=len(X))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,} features")

    print("\n[MODEL] Training Multinomial Naive Bayes ...")
    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(X_train_vec, y_train)

    print("[MODEL] Training Logistic Regression ...")
    lr_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    lr_model.fit(X_train_vec, y_train)

    print("\n[RESULTS] Evaluation:")
    nb_metrics = evaluate_model("Multinomial Naive Bayes", nb_model, X_test_vec, y_test)
    lr_metrics = evaluate_model("Logistic Regression",    lr_model, X_test_vec, y_test)

    if lr_metrics["f1"] >= nb_metrics["f1"]:
        best_model    = lr_model
        best_name     = "Logistic Regression"
        best_metrics  = lr_metrics
        other_metrics = nb_metrics
        other_name    = "Multinomial Naive Bayes"
    else:
        best_model    = nb_model
        best_name     = "Multinomial Naive Bayes"
        best_metrics  = nb_metrics
        other_metrics = lr_metrics
        other_name    = "Logistic Regression"

    print(f"\n[WINNER] Best model (by Spam F1): {best_name}")
    print(
        "\n  WHY F1, NOT ACCURACY?\n"
        "  Accuracy counts correct predictions for BOTH ham and spam.\n"
        "  Because spam is rare (~13%), a model that labels every message\n"
        "  as ham is 87% accurate yet completely useless.\n"
        "  F1-score for spam rewards finding real spam AND not\n"
        "  mislabelling normal messages. That is what we actually care about."
    )

    top_spam_words = get_top_spam_words(vectorizer, best_model, n=TOP_N_WORDS)
    print(f"\n[TOP WORDS] Top {TOP_N_WORDS} spam-indicator words:")
    print("  " + ", ".join(top_spam_words))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.joblib"))
    joblib.dump(best_model, os.path.join(MODEL_DIR, "model.joblib"))

    metadata = {
        "dataset"         : "SMS Spam Collection",
        "total_messages"  : len(df),
        "train_size"      : len(X_train),
        "test_size"       : len(X_test),
        "best_model"      : best_name,
        "other_model"     : other_name,
        "best_metrics"    : best_metrics,
        "other_metrics"   : other_metrics,
        "top_spam_words"  : top_spam_words,
        "tfidf_vocab_size": len(vectorizer.vocabulary_),
        "class_counts"    : {
            "ham" : int((y == "ham").sum()),
            "spam": int((y == "spam").sum()),
        },
    }

    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[SAVED] Files written to {MODEL_DIR}/")
    print("  model.joblib       - trained " + best_name)
    print("  vectorizer.joblib  - TF-IDF vectoriser")
    print("  metadata.json      - metrics and word lists")
    print("\n[DONE] Training complete! Now run:  python app.py")
    print("  Then open http://127.0.0.1:5000 in your browser.")


if __name__ == "__main__":
    main()
