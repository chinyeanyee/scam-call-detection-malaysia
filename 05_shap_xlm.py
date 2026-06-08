# ==============================================
# 05_shap_xlm.py
# SHAP explainability for XLM-RoBERTa
# Uses REAL test samples — not hardcoded text
# Run AFTER 06_xlm_roberta_model.py
# ==============================================

import os
import torch
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import MASTER_CSV, SAVED_MODELS_FOLDER, RANDOM_SEED, XLM_MAX_LEN

MODEL_PATH = os.path.join(SAVED_MODELS_FOLDER, "xlm_roberta_scam")

# ==============================================
# STEP 1: LOAD MODEL
# ==============================================

def load_model():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    print("✅ Model loaded")
    return tokenizer, model


# ==============================================
# STEP 2: GET REAL TEST SAMPLES
# Uses same leakage-free split as training
# Picks clearly classified samples
# ==============================================

def get_test_samples():
    df = pd.read_csv(MASTER_CSV)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2,
                             random_state=RANDOM_SEED)
    _, val_idx = next(gss.split(df, groups=df['source_file']))
    val_df     = df.iloc[val_idx].reset_index(drop=True)

    tokenizer, model = load_model()
    predict_fn       = make_predict_fn(tokenizer, model)

    scam_samples   = val_df[
        (val_df['label'] == 1) &
        (val_df['content'].str.split().str.len() > 150)
    ]['content'].tolist()

    normal_samples = val_df[
        (val_df['label'] == 0) &
        (val_df['content'].str.split().str.len() > 100)
    ]['content'].tolist()

    # Scan ALL windows and pick highest scam probability window
    best_scam_text  = None
    best_scam_score = 0

    for text in scam_samples:
        words     = text.split()
        step      = 50
        window    = 100

        for start in range(0, max(1, len(words) - window), step):
            chunk = ' '.join(words[start:start + window])
            prob  = predict_fn([chunk])[0][1]
            if prob > best_scam_score:
                best_scam_score = prob
                best_scam_text  = chunk

    # Find most confident normal prediction
    best_normal_text  = None
    best_normal_score = 1

    for text in normal_samples:
        truncated = ' '.join(text.split()[:100])
        prob      = predict_fn([truncated])[0][1]
        if prob < best_normal_score:
            best_normal_score = prob
            best_normal_text  = truncated

    print(f"\nBest scam window   — P(Scam): {best_scam_score:.4f}")
    print(f"Best normal sample — P(Scam): {best_normal_score:.4f}")
    print(f"\nScam window (first 150 chars)  : {best_scam_text[:150]}")
    print(f"Normal text (first 150 chars)  : {best_normal_text[:150]}")

    return best_scam_text, best_normal_text, tokenizer, model
    df = pd.read_csv(MASTER_CSV)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2,
                             random_state=RANDOM_SEED)
    _, val_idx = next(gss.split(df, groups=df['source_file']))
    val_df     = df.iloc[val_idx].reset_index(drop=True)

    tokenizer, model = load_model()
    predict_fn       = make_predict_fn(tokenizer, model)

    # Only consider samples with enough words for meaningful SHAP
    scam_samples   = val_df[
        (val_df['label'] == 1) &
        (val_df['content'].str.split().str.len() > 100)
    ]['content'].tolist()

    normal_samples = val_df[
        (val_df['label'] == 0) &
        (val_df['content'].str.split().str.len() > 100)
    ]['content'].tolist()

    # Find most confident scam prediction
    best_scam_text  = None
    best_scam_score = 0

    for text in scam_samples:
        # Use middle 100 words — skip intro and outro
        words     = text.split()
        mid_start = max(0, len(words)//3)
        truncated = ' '.join(words[mid_start:mid_start+100])
        prob      = predict_fn([truncated])[0][1]
        if prob > best_scam_score:
            best_scam_score = prob
            best_scam_text  = truncated

    # Find most confident normal prediction
    best_normal_text  = None
    best_normal_score = 1

    for text in normal_samples:
        truncated = ' '.join(text.split()[:100])
        prob      = predict_fn([truncated])[0][1]
        if prob < best_normal_score:
            best_normal_score = prob
            best_normal_text  = truncated

    print(f"\nBest scam sample   — P(Scam): {best_scam_score:.4f}")
    print(f"Best normal sample — P(Scam): {best_normal_score:.4f}")
    print(f"\nScam text (first 100 chars)  : {best_scam_text[:100]}...")
    print(f"Normal text (first 100 chars): {best_normal_text[:100]}...")

    return best_scam_text, best_normal_text, tokenizer, model
# ==============================================
# STEP 3: PREDICTION FUNCTION FOR SHAP
# ==============================================

def make_predict_fn(tokenizer, model):
    def predict_proba(texts):
        texts  = [str(t) for t in texts]
        inputs = tokenizer(
            texts,
            padding        = True,
            truncation     = True,
            max_length     = XLM_MAX_LEN,
            return_tensors = "pt"
        )
        with torch.no_grad():
            outputs = model(**inputs)
            probs   = torch.softmax(outputs.logits, dim=1)
        return probs.cpu().numpy()
    return predict_proba


# ==============================================
# STEP 4: RUN SHAP + GENERATE WATERFALL PLOTS
# ==============================================

def run_shap_analysis(tokenizer, model, scam_text, normal_text):
    predict_fn = make_predict_fn(tokenizer, model)
    masker     = shap.maskers.Text(tokenizer=r"\W+")
    explainer  = shap.Explainer(predict_fn, masker)

    texts  = [scam_text, normal_text]
    labels = ["SCAM", "NORMAL"]

    # Print probabilities
    probs = predict_fn(texts)
    print("\n" + "="*50)
    print("MODEL PREDICTIONS")
    print("="*50)
    for i, (label, p) in enumerate(zip(labels, probs)):
        print(f"{label}")
        print(f"  P(Normal) = {p[0]:.4f}")
        print(f"  P(Scam)   = {p[1]:.4f}")
        print()

    # Compute SHAP values
    print("Computing SHAP values (this may take a few minutes)...")
    shap_values = explainer(texts)

    # Waterfall for SCAM text
    plt.figure(figsize=(12, 7))
    shap.plots.waterfall(
        shap_values[0, :, 1],
        max_display=15,
        show=False
    )
    plt.title("SHAP Explanation — SCAM Text (XLM-RoBERTa)")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVED_MODELS_FOLDER, 'xlm_shap_scam.png'),
                dpi=150, bbox_inches='tight')
    plt.show()

    # Waterfall for NORMAL text
    plt.figure(figsize=(12, 7))
    shap.plots.waterfall(
        shap_values[1, :, 1],
        max_display=15,
        show=False
    )
    plt.title("SHAP Explanation — NORMAL Text (XLM-RoBERTa)")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVED_MODELS_FOLDER, 'xlm_shap_normal.png'),
                dpi=150, bbox_inches='tight')
    plt.show()

    print("\n✅ SHAP waterfall plots saved")
    return shap_values, probs


# ==============================================
# STEP 5: HUMAN-READABLE EXPLANATION
# ==============================================

def explain_prediction(text, prob_scam):
    if prob_scam >= 0.8:
        verdict = "HIGH RISK — Likely Scam"
        reason  = ("Contains authority-related terms, urgency markers, or "
                   "financial directives commonly found in scam calls.")
    elif prob_scam >= 0.5:
        verdict = "MEDIUM RISK — Suspicious"
        reason  = ("Some patterns weakly resemble scam language. "
                   "Proceed with caution.")
    else:
        verdict = "LOW RISK — Likely Normal"
        reason  = ("Contains everyday conversational language without "
                   "scam indicators.")

    print(f"  Verdict : {verdict}")
    print(f"  Reason  : {reason}")
    print(f"  P(Scam) : {prob_scam:.4f}")


# ==============================================
# MAIN
# ==============================================

if __name__ == "__main__":
    print("="*50)
    print("Loading best scam and normal samples")
    print("="*50)
    scam_text, normal_text, tokenizer, model = get_test_samples()

    print("\n" + "="*50)
    print("Running SHAP analysis")
    print("="*50)
    shap_values, probs = run_shap_analysis(
        tokenizer, model, scam_text, normal_text
    )

    print("\n" + "="*50)
    print("HUMAN-READABLE EXPLANATIONS")
    print("="*50)
    print("SCAM sample:")
    explain_prediction(scam_text, probs[0][1])
    print("\nNORMAL sample:")
    explain_prediction(normal_text, probs[1][1])

    print("\n✅ SHAP analysis complete")
    print("➡️  Next step: run 07_dashboard.py")