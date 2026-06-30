import os
import torch
import shap
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from config import (
    MASTER_CSV,
    SAVED_MODELS_FOLDER,
    RANDOM_SEED,
    TEST_SIZE,
    XLM_MAX_LEN
)

MODEL_PATH = os.path.join(
    SAVED_MODELS_FOLDER,
    "xlm_roberta_scam"
)

def segment_text(text, tokenizer, max_len=50, overlap=10):

    if not isinstance(text, str) or not text.strip():
        return []

    tokens = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    if len(tokens) <= max_len:
        return [tokenizer.decode(tokens)]

    segments = []

    step = max_len - overlap

    for i in range(0, len(tokens), step):

        chunk = tokenizer.decode(tokens[i:i+max_len])

        segments.append(chunk)

        if i + max_len >= len(tokens):
            break

    return segments

# ==============================================
# STEP 2
# Load trained XLM-RoBERTa model
# ==============================================

def load_model():

    print("=" * 60)
    print("Loading trained XLM-RoBERTa model...")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.eval()

    print("✅ Model loaded successfully.")

    return tokenizer, model

# ==============================================
# STEP 3
# Prediction function for SHAP
# ==============================================

def make_predict_function(tokenizer, model):

    def predict(texts):

        # Ensure input is always a list of strings
        texts = [str(t) for t in texts]

        # Tokenize input
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=XLM_MAX_LEN,
            return_tensors="pt"
        )

        # Model inference
        with torch.no_grad():
            outputs = model(**encoded)
            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )

        return probabilities.cpu().numpy()

    return predict

# ==============================================
# STEP 4
# Prepare validation segments
# ==============================================

def prepare_validation_segments(tokenizer):

    print("=" * 60)
    print("Preparing validation segments...")
    print("=" * 60)

    # Load dataset
    df = pd.read_csv(MASTER_CSV)

    # Recreate the SAME transcript split
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )

    _, val_idx = next(
        splitter.split(
            df,
            groups=df["source_file"]
        )
    )

    val_df = df.iloc[val_idx].reset_index(drop=True)

    print(f"Validation transcripts : {len(val_df)}")

    segments = []

    # Segment every validation transcript
    for _, row in val_df.iterrows():

        chunk_list = segment_text(
            row["content"],
            tokenizer
        )

        for chunk in chunk_list:

            segments.append({

                "text": chunk,

                "label": row["label"],

                "source_file": row["source_file"]

            })

    segment_df = pd.DataFrame(segments)

    print(f"Validation segments : {len(segment_df)}")

    return segment_df



# ==============================================
# STEP 5
# Select representative validation segments
# ==============================================


# ==============================================
# Scam keyword scoring
# ==============================================

SCAM_KEYWORDS = [
    "akaun",
    "bank",
    "duit",
    "wang",
    "polis",
    "mahkamah",
    "otp",
    "tac",
    "transfer",
    "bayar",
    "pin",
    "kad",
    "pengenalan",
    "ic",
    "siasatan",
    "bekukan",
    "transaksi"
]

def keyword_score(text):

    text = str(text).lower()

    score = 0

    for keyword in SCAM_KEYWORDS:
        if keyword in text:
            score += 1

    return score






def select_representative_segments(segment_df,
                                   tokenizer,
                                   model):

    print("=" * 60)
    print("Selecting representative validation segments...")
    print("=" * 60)

    predict = make_predict_function(
        tokenizer,
        model
    )

    
    probabilities = []
    predictions = []

    # Predict all validation segments at once
    texts = segment_df["text"].tolist()

    all_probs = predict(texts)

    for prob in all_probs:

     probabilities.append(prob[1])      # Scam probability

     predictions.append(
        int(prob[1] >= 0.5)
    )    

    segment_df = segment_df.copy()

    segment_df["prob_scam"] = probabilities
    segment_df["prediction"] = predictions

    # Keep only correctly classified samples
    correct_df = segment_df[
        segment_df["label"] ==
        segment_df["prediction"]
    ].copy()

    print(f"Correctly classified segments : {len(correct_df)}")
    
    # Calculate scam keyword score
    correct_df["keyword_score"] = correct_df["text"].apply(keyword_score)

    #
    # Representative Scam
    #

    scam_df = correct_df[
        (correct_df["label"] == 1) &
        (correct_df["prob_scam"] >= 0.80) 
    ]

    # Show the best scam candidates
    print("\nTop 10 Scam Candidates")
    print("=" * 60)
    
    top_candidates = scam_df.sort_values(
    by=["keyword_score", "prob_scam"],
    ascending=[False, False]
    ).head(10)

    print(
    top_candidates[
        ["keyword_score", "prob_scam", "text"]
    ].to_string(index=True)
    )


    #
    # Representative Normal
    #

    normal_df = correct_df[
        (correct_df["label"] == 0) &
        (correct_df["prob_scam"] <= 0.15)
    ]

    if scam_df.empty:
        raise ValueError(
            "No representative scam segment found."
        )

    if normal_df.empty:
        raise ValueError(
            "No representative normal segment found."
        )
    selected_index = 687     # <-- change this after viewing the Top 10

    scam_segment = scam_df.loc[selected_index]

    normal_segment = normal_df.sort_values(
        "prob_scam",
        ascending=True
    ).iloc[0]

    print("\nRepresentative Scam Segment")
    print("----------------------------")
    print(f"P(Scam) : {scam_segment['prob_scam']:.4f}")
    print(scam_segment["text"][:250])

    print("\nRepresentative Normal Segment")
    print("-----------------------------")
    print(f"P(Scam) : {normal_segment['prob_scam']:.4f}")
    print(normal_segment["text"][:250])

    return (
        scam_segment,
        normal_segment
    )

# ==============================================
# STEP 6
# Run SHAP Explainability
# ==============================================

def run_shap(tokenizer,
             model,
             scam_segment,
             normal_segment):

    print("=" * 60)
    print("Running SHAP Explainability...")
    print("=" * 60)

    predict = make_predict_function(
        tokenizer,
        model
    )

    # Use the SAME tokenizer as XLM-RoBERTa
    masker = shap.maskers.Text(tokenizer)

    explainer = shap.Explainer(
        predict,
        masker
    )

    texts = [

        scam_segment["text"],

        normal_segment["text"]

    ]

    print("\nRepresentative Scam Segment")
    print(scam_segment["text"])

    print("\nRepresentative Normal Segment")
    print(normal_segment["text"])
    
    print("Computing SHAP values...")
    print("This may take several minutes.\n")

    shap_values = explainer(texts)

    probabilities = predict(texts)

    return (
        shap_values,
        probabilities
    )

# ==============================================
# STEP 7
# Save SHAP Waterfall Plots
# ==============================================

def save_waterfall_plots(shap_values):

    print("Saving waterfall plots...")

    #
    # Scam
    #

    plt.figure(figsize=(12,7))

    shap.plots.waterfall(

        shap_values[0,:,1],

        max_display=15,

        show=False

    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            SAVED_MODELS_FOLDER,

            "xlm_shap_scam_waterfall.png"

        ),

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    #
    # Normal
    #

    plt.figure(figsize=(12,7))

    shap.plots.waterfall(

        shap_values[1,:,1],

        max_display=15,

        show=False

    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            SAVED_MODELS_FOLDER,

            "xlm_shap_normal_waterfall.png"

        ),

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    print("✅ Waterfall plots saved.")

if __name__ == "__main__":

    print("=" * 60)
    print("XLM-RoBERTa SHAP Explainability")
    print("=" * 60)

    # Load model
    tokenizer, model = load_model()

    # Prepare validation segments
    segment_df = prepare_validation_segments(tokenizer)

    # Select representative samples
    scam_segment, normal_segment = select_representative_segments(
        segment_df,
        tokenizer,
        model
    )

    # Run SHAP
    shap_values, probabilities = run_shap(
        tokenizer,
        model,
        scam_segment,
        normal_segment
    )

    # Save figures
    save_waterfall_plots(shap_values)

    print("\n" + "=" * 60)
    print("SHAP probabilities")
    print("=" * 60)

    print(f"Scam segment   : {probabilities[0][1]:.4f}")
    print(f"Normal segment : {probabilities[1][1]:.4f}")

    print("\n✅ SHAP analysis complete.")
    print(f"Results saved to:\n{SAVED_MODELS_FOLDER}")
  

