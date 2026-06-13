# ==============================================
# config.py
# Central configuration — update BASE_PATH only
# ==============================================

import os

# ── Google Drive root (update if your drive letter is different) ──
BASE_PATH = r"G:\My Drive\Scam_Source"

# ── Input folders ──
SCAM_TRANSCRIPT_FOLDER    = os.path.join(BASE_PATH, "Scam_Transcript")
SCAM_SYNTHETIC_FOLDER     = os.path.join(BASE_PATH, "Synthetic_Scam_Transcript")
RAW_NORMAL_FOLDER         = os.path.join(BASE_PATH, "Raw_Normal_MagicHub")
CLEANED_CONVO_FOLDER      = os.path.join(BASE_PATH, "Cleaned_Convo_Transcript")
CLEANED_CALLCENTER_FOLDER = os.path.join(BASE_PATH, "Cleaned_CallCenter_Transcript")
SAVED_MODELS_FOLDER       = os.path.join(BASE_PATH, "Saved_Models")

# ── Output CSV files ──
MASTER_CSV      = os.path.join(BASE_PATH, "master_scam_data.csv")
TFIDF_READY_CSV = os.path.join(BASE_PATH, "tfidf_ready.csv")
XLM_READY_CSV   = os.path.join(BASE_PATH, "xlm_ready.csv")

# ── Model settings ──
RANDOM_SEED    = 42
TEST_SIZE      = 0.2
TFIDF_MAX_FEAT = 5000
TFIDF_NGRAM    = (1, 2)
XLM_MODEL_ID   = "xlm-roberta-base"
XLM_MAX_LEN    = 128
XLM_BATCH_SIZE = 8
XLM_EPOCHS       = 3
XLM_WEIGHT_DECAY = 0.05
XLM_LR           = 1e-5