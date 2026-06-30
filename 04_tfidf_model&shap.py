# ==============================================
# 04_tfidf_model.py
# TF-IDF + Logistic Regression baseline model
# with SHAP explainability
# Run AFTER 01_data_preparation.py
# ==============================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)
import nltk
import stopwordsiso as swiso

try:
    import malaya
    MALAYA_AVAILABLE = True
except:
    MALAYA_AVAILABLE = False
    print("⚠️  malaya not installed — using fallback stopwords only")

from config import (
    MASTER_CSV, TFIDF_READY_CSV, SAVED_MODELS_FOLDER,
    RANDOM_SEED, TEST_SIZE, TFIDF_MAX_FEAT, TFIDF_NGRAM
)

os.makedirs(SAVED_MODELS_FOLDER, exist_ok=True)

# ==============================================
# STEP 1: BUILD STOPWORD LIST (for TF-IDF only)
# ==============================================

def build_stopwords():
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords

    sw = set(stopwords.words('indonesian'))
    sw |= set(stopwords.words('english'))
    sw |= set(swiso.stopwords("ms"))
    sw |= set(swiso.stopwords("en"))

    if MALAYA_AVAILABLE:
        sw |= set(malaya.text.function.STOPWORDS)

    # Malaysian fillers — low information value
    sw |= {
        "ya", "okey", "ok", "lah", "ke", "ni", "tu", "pun", "dah", "je",
        "nak", "apa", "kita", "saya", "kamu", "dia", "mereka", "anda",
        "encik", "puan", "tuan", "terima", "kasih", "hello", "hai",
        "mana", "macam", "tadi", "boleh", "ada", "cakap", "buat",
        "tahu", "tunggu", "betul", "eh", "actually", "like", "know",
        "then", "just", "get", "one", "so"
    }
    return sw


def apply_stopwords(text, stopwords_set):
    if not isinstance(text, str):
        return ""
    tokens   = text.split()
    filtered = [w for w in tokens
                if w not in stopwords_set and len(w) > 2]
    return " ".join(filtered)


# ==============================================
# STEP 2: PREPARE TF-IDF DATASET
# Stopwords removed ONLY for TF-IDF pipeline
# ==============================================

def prepare_tfidf_data():
    df         = pd.read_csv(MASTER_CSV)
    stopwords_ = build_stopwords()

    print("Applying stopword removal for TF-IDF pipeline...")
    df['content_tfidf'] = df['content'].apply(
        lambda x: apply_stopwords(x, stopwords_)
    )

    # Check for rows that became empty after stopword removal
    empty = (df['content_tfidf'].str.strip() == '').sum()
    if empty > 0:
        print(f"⚠️  {empty} rows became empty after stopword removal")
        df = df[df['content_tfidf'].str.strip() != ''].reset_index(drop=True)

    df.to_csv(TFIDF_READY_CSV, index=False)
    print(f"✅ TF-IDF ready dataset saved: {TFIDF_READY_CSV}")
    print(f"   Total: {len(df)} | Scam: {(df['label']==1).sum()} | Normal: {(df['label']==0).sum()}")
    return df


# ==============================================
# STEP 3: TRAIN MODEL
# ==============================================

def train_tfidf_model(df):
    gss = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED
)
    train_idx, test_idx = next(
    gss.split(
        df,
        groups=df['source_file']
    )
)

    train_df = df.iloc[train_idx]
    test_df  = df.iloc[test_idx]

    X_train = train_df['content_tfidf'].fillna('')
    X_test  = test_df['content_tfidf'].fillna('')

    y_train = train_df['label']
    y_test  = test_df['label']
    

    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    # Vectorize
    vectorizer = TfidfVectorizer(
        max_features = TFIDF_MAX_FEAT,
        ngram_range  = TFIDF_NGRAM
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    # Train
    model = LogisticRegression(
        class_weight = 'balanced',
        random_state = RANDOM_SEED,
        max_iter     = 1000
    )
    model.fit(X_train_tfidf, y_train)

    return model, vectorizer, X_train_tfidf, X_test_tfidf, X_test, y_test


# ==============================================
# STEP 4: EVALUATE
# ==============================================

def evaluate_model(model, X_test_tfidf, y_test):
    y_pred = model.predict(X_test_tfidf)

    print("\n" + "="*50)
    print("📋 CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(
        y_test, y_pred,
        target_names=['Normal', 'Scam'],
        digits=4
    ))
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Scam'],
                yticklabels=['Normal', 'Scam'])
    plt.title('Confusion Matrix — TF-IDF Logistic Regression')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVED_MODELS_FOLDER, 'tfidf_confusion_matrix.png'),
                dpi=150)
    plt.show()
    print(f"✅ Confusion matrix saved")

    return y_pred


# ==============================================
# STEP 5: SHAP EXPLAINABILITY
# Generates global + local waterfall plots
# from REAL test samples (not hardcoded text)
# ==============================================

def run_shap(model, vectorizer, X_train_tfidf, X_test_tfidf, X_test, y_test, y_pred):
    feature_names = vectorizer.get_feature_names_out()

    explainer   = shap.LinearExplainer(model, X_train_tfidf,
                                        feature_names=feature_names)
    shap_values = explainer(X_test_tfidf)

    # --- Global bar plot ---
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.title("Global SHAP — Mean Feature Importance (TF-IDF)")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVED_MODELS_FOLDER, 'tfidf_shap_global.png'),
                dpi=150, bbox_inches='tight')
    plt.show()

    
        # ===========================================
# Find representative scam sample
# ===========================================

    SCAM_KEYWORDS = [
            "akaun",
            "bank",
            "polis",
            "duit",
            "wang",
            "otp",
            "tac",
            "sprm",
            "mahkamah",
            "bekukan",
            "macau"
        ]

    
    candidates = []

    for i, (yt, yp) in enumerate(zip(y_test, y_pred)):

        if yt != 1 or yp != 1:
            continue

        text = X_test.iloc[i]

        score = sum(
            keyword in text.lower()
            for keyword in SCAM_KEYWORDS
        )

        candidates.append({

            "index": i,

            "score": score,

            "prob": model.predict_proba(
                X_test_tfidf[i]
            )[0][1],

            "text": text

        })
    

        # ===========================================
    # Step 2: SORT THE CANDIDATES
    # ===========================================

    candidates = sorted(

        candidates,

        key=lambda x: (
            x["score"],
            x["prob"]
        ),

        reverse=True
    )

    
        # ===========================================
    # Step 3: PRINT TOP 10
    # ===========================================

    print("\nTop 10 Representative Scam Candidates")
    print("=" * 80)

    for c in candidates[:10]:

        print(f"Index : {c['index']}")
        print(f"Keyword Score : {c['score']}")
        print(f"P(Scam) : {c['prob']:.4f}")
        print("-" * 80)
        print(c["text"])
        print("=" * 80)

    # ===========================================
    # Step 4: CHOOSE ONE
    # ===========================================

    idx = candidates[3]["index"]      # temporary
    
    
        
        # ===========================================
    # Print the selected representative sample
    # ===========================================
    print("\nRepresentative Scam Sample")
    print("-" * 60)
    print("\nRepresentative Scam Transcript")
    print("=" * 80)
    print(X_test.iloc[idx])
    print("=" * 80)

    plt.figure(figsize=(12, 6))
    shap.plots.waterfall(
        shap_values[idx],
        max_display=15,
        show=False
    )
    plt.title(f"Local SHAP — Scam Sample (Test idx {idx})")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            SAVED_MODELS_FOLDER,
            'tfidf_shap_scam.png'
        ),
        dpi=150,
        bbox_inches='tight'
    )
    plt.show()
    
        # ===========================================
    # Representative Normal Sample
    # ===========================================

    normal_indices = [
        i for i, (yt, yp) in enumerate(zip(y_test, y_pred))
        if yt == 0 and yp == 0
    ]

    # Choose the first correctly classified normal sample
    normal_idx = normal_indices[0]

    print("\nRepresentative Normal Transcript")
    print("=" * 80)
    print(X_test.iloc[normal_idx])
    print("=" * 80)

    plt.figure(figsize=(12, 6))

    shap.plots.waterfall(
        shap_values[normal_idx],
        max_display=15,
        show=False
    )

    plt.title(f"Local SHAP — Normal Sample (Test idx {normal_idx})")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            SAVED_MODELS_FOLDER,
            "tfidf_shap_normal.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

# ==============================================
    print("\n✅ All SHAP plots saved")


# ==============================================
# MAIN
# ==============================================

if __name__ == "__main__":
    print("="*50)
    print("STEP 1: Preparing TF-IDF dataset")
    print("="*50)
    df = prepare_tfidf_data()

    print("\n" + "="*50)
    print("STEP 2: Training model")
    print("="*50)
    model, vectorizer, X_train_tfidf, X_test_tfidf, X_test, y_test = \
        train_tfidf_model(df)

    print("\n" + "="*50)
    print("STEP 3: Evaluating model")
    print("="*50)
    y_pred = evaluate_model(model, X_test_tfidf, y_test)

    print("\n" + "="*50)
    print("STEP 4: SHAP Explainability")
    print("="*50)
    run_shap(model, vectorizer, X_train_tfidf, X_test_tfidf,
             X_test, y_test, y_pred)

    # Save model and vectorizer
    joblib.dump(model,      os.path.join(SAVED_MODELS_FOLDER, 'tfidf_model.pkl'))
    joblib.dump(vectorizer, os.path.join(SAVED_MODELS_FOLDER, 'tfidf_vectorizer.pkl'))
    print(f"\n✅ Model saved to: {SAVED_MODELS_FOLDER}")
    print("➡️  Next step: run 06_xlm_roberta_model.py")

    # Save classification report to file
    report = classification_report(
        y_test, y_pred,
        target_names=['Normal', 'Scam'],
        digits=4
    )
    report_path = os.path.join(SAVED_MODELS_FOLDER, 'tfidf_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}\n\n")
        f.write(report)
    print(f"✅ Classification report saved: {report_path}")