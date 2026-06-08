# ==============================================
# api.py
# FastAPI backend for scam detection dashboard
# Endpoints:
#   GET  /              — health check
#   GET  /health        — health check
#   POST /predict       — predict from text
#   POST /analyze-audio — upload audio, transcribe, predict
# Run: uvicorn api:app --reload
# ==============================================

import os
import re
import time
import tempfile
import subprocess
import joblib
import shap
import torch
import numpy as np
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import azure.cognitiveservices.speech as speechsdk

load_dotenv()

from config import SAVED_MODELS_FOLDER, XLM_MAX_LEN

# ==============================================
# AZURE CREDENTIALS
# ==============================================
AZURE_KEY    = os.getenv("AZURE_SPEECH_KEY")
AZURE_REGION = os.getenv("AZURE_SPEECH_REGION", "southeastasia")

# ==============================================
# APP SETUP
# ==============================================
app = FastAPI(title="Scam Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ==============================================
# LOAD MODELS ON STARTUP
# ==============================================
print("Loading models...")

tfidf_model      = joblib.load(os.path.join(SAVED_MODELS_FOLDER, 'tfidf_model.pkl'))
tfidf_vectorizer = joblib.load(os.path.join(SAVED_MODELS_FOLDER, 'tfidf_vectorizer.pkl'))

# ==============================================
# TF-IDF SHAP BACKGROUND DATA
# ==============================================

import pandas as pd
from config import TFIDF_READY_CSV

try:
    tfidf_background_df = pd.read_csv(TFIDF_READY_CSV)

    background_texts = (
        tfidf_background_df['content_tfidf']
        .fillna('')
        .sample(
            n=min(100, len(tfidf_background_df)),
            random_state=42
        )
    )

    background_vec = tfidf_vectorizer.transform(background_texts)

    print(f"✅ Loaded TF-IDF SHAP background: {background_vec.shape}")

except Exception as e:
    print(f"⚠️ Could not load TF-IDF background data: {e}")
    background_vec = None

xlm_model_path = os.path.join(SAVED_MODELS_FOLDER, "xlm_roberta_scam")
xlm_tokenizer  = AutoTokenizer.from_pretrained(xlm_model_path)
xlm_model      = AutoModelForSequenceClassification.from_pretrained(xlm_model_path)
xlm_model.eval()

print("✅ All models loaded")


# ==============================================
# TEXT CLEANING
# ==============================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'scammer\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ==============================================
# AUDIO CONVERSION
# ==============================================
def convert_to_wav(input_path: str) -> str:
    """Convert audio file to WAV 16kHz mono using ffmpeg."""
    wav_path = input_path.rsplit('.', 1)[0] + '_converted.wav'
    subprocess.run([
        'ffmpeg',
        '-i', input_path,
        '-ar', '16000',
        '-ac', '1',
        '-f', 'wav',
        wav_path,
        '-y'
    ], check=True, capture_output=True)
    return wav_path


# ==============================================
# AZURE TRANSCRIPTION WITH RETRY
# ==============================================
def transcribe_audio(file_path: str, max_retries: int = 3) -> str:
    """Transcribe with retry logic for Azure connection instability."""

    for attempt in range(max_retries):
        print(f"Transcription attempt {attempt + 1}/{max_retries}")

        if not AZURE_KEY:
            raise ValueError("AZURE_SPEECH_KEY not set in .env file")

        speech_config = speechsdk.SpeechConfig(
            subscription = AZURE_KEY,
            region       = AZURE_REGION
        )
        speech_config.speech_recognition_language = "ms-MY"

        speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
        "10000"
        )

        speech_config.set_property(
        speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs,
        "3000"
        )

        

        audio_config = speechsdk.audio.AudioConfig(filename=file_path)
        recognizer   = speechsdk.SpeechRecognizer(
                           speech_config=speech_config,
                           audio_config=audio_config
                       )
        done            = False
        full_transcript = []

        def stop_cb(evt):
            nonlocal done
            done = True

        def on_recognized(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                if evt.result.text.strip():
                    full_transcript.append(evt.result.text.strip())

        recognizer.recognized.connect(on_recognized)
        recognizer.session_stopped.connect(stop_cb)
        recognizer.canceled.connect(stop_cb)

        recognizer.start_continuous_recognition()

        timeout = 60  # seconds max wait
        elapsed = 0
        while not done and elapsed < timeout:
            time.sleep(0.5)
            elapsed += 0.5

        if not done:
            print("⚠️ Session did not end cleanly, forcing stop...")

        recognizer.stop_continuous_recognition()
        time.sleep(1)  # give Azure 1 extra second to flush final segments

        transcript = " ".join(full_transcript).strip()

        if transcript and len(transcript.split()) >= 5:
            print(f"✅ Transcription successful on attempt {attempt + 1}")
            return transcript

        print(f"⚠️  Empty transcript on attempt {attempt + 1}, retrying...")
        wait_time = (attempt + 1) * 5
        print(f"Retrying in {wait_time} seconds...")
        time.sleep(wait_time)

    print("❌ All transcription attempts failed")
    return ""


# ==============================================
# PREDICTION FUNCTIONS
# ==============================================
def predict_tfidf(text):
    cleaned = clean_text(text)
    vec     = tfidf_vectorizer.transform([cleaned])
    prob    = tfidf_model.predict_proba(vec)[0]
    label   = int(tfidf_model.predict(vec)[0])
    return {
        "label"      : label,
        "probability": round(float(prob[1]), 4),
        "verdict"    : "SCAM" if label == 1 else "NORMAL"
    }


def predict_xlm(text):
    cleaned = clean_text(text)
    inputs  = xlm_tokenizer(
        cleaned,
        return_tensors = "pt",
        truncation     = True,
        max_length     = XLM_MAX_LEN,
        padding        = True
    )
    with torch.no_grad():
        outputs = xlm_model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1)[0]

    label = int(torch.argmax(probs).item())
    return {
        "label"      : label,
        "probability": round(float(probs[1].item()), 4),
        "verdict"    : "SCAM" if label == 1 else "NORMAL"
    }


# ==============================================
# SHAP FUNCTIONS
# ==============================================

# ==============================================
# RISK FACTOR DETECTION
# ==============================================

def detect_risk_factors(text):

    text = text.lower()

    factors = []

    # Authority impersonation
    if any(word in text for word in [
        "polis",
        "pdrm",
        "pegawai",
        "sprm",
        "lhdn",
        "mahkamah",
        "jabatan siasatan",
        "bank negara"
    ]):
        factors.append({
            "title": "Authority Impersonation",
            "description": "Claims to represent police, government agencies, or official authorities."
        })

    # Financial crime allegation
    if any(word in text for word in [
        "pengubahan wang haram",
        "jenayah",
        "penipuan",
        "sindiket",
        "aktiviti haram"
    ]):
        factors.append({
            "title": "Financial Crime Allegation",
            "description": "Accuses the victim of involvement in criminal or financial offences."
        })

    # Money transfer request
    if any(word in text for word in [
        "akaun",
        "pindahkan",
        "transfer",
        "pemindahan",
        "akaun selamat",
        "akaun perlindungan"
    ]):
        factors.append({
            "title": "Money Transfer Request",
            "description": "Requests funds to be transferred to another account."
        })

    # OTP / TAC request
    if any(word in text for word in [
        "otp",
        "tac",
        "verification code",
        "kod pengesahan"
    ]):
        factors.append({
            "title": "OTP / TAC Request",
            "description": "Requests banking authentication credentials."
        })

    # Urgency / threat tactics
    if any(word in text for word in [
        "waran",
        "tangkap",
        "ditahan",
        "24 jam",
        "segera",
        "mahkamah"
    ]):
        factors.append({
            "title": "Urgency / Threat Tactics",
            "description": "Uses threats, deadlines, or legal consequences to pressure action."
        })

    return factors


def generate_recommended_actions(risk_factors: list) -> list:
    actions = []
    factor_titles = [f["title"] for f in risk_factors]

    if "OTP / TAC Request" in factor_titles:
        actions.append("Do not share any OTP, TAC, or verification codes with the caller.")

    if "Money Transfer Request" in factor_titles:
        actions.append("Do not transfer money to any account provided during this call.")

    if "Authority Impersonation" in factor_titles:
        actions.append("Verify the caller's identity using official government or bank contact channels — not numbers given by the caller.")

    if "Financial Crime Allegation" in factor_titles:
        actions.append("Do not panic. Contact the relevant authority independently to verify the claim before taking any action.")

    if "Urgency / Threat Tactics" in factor_titles:
        actions.append("Ignore pressure tactics. Legitimate authorities will never demand immediate transfers or threaten arrest by phone.")

    if not actions:
        actions.append("No specific actions required. Stay alert and hang up if anything feels suspicious.")

    return actions


def get_tfidf_shap(text):

    cleaned = clean_text(text)

    vec = tfidf_vectorizer.transform([cleaned])
    print("\n========== TF-IDF DEBUG ==========")
    print("Non-zero features:", vec.nnz)

    feature_names = tfidf_vectorizer.get_feature_names_out()

    print("\n========== TF-IDF SHAP DEBUG ==========")
    print(f"Non-zero TF-IDF features: {vec.nnz}")

    if vec.nnz == 0:
        print("❌ Transcript produced no TF-IDF features")
        return []

    try:

        if background_vec is not None:
            explainer = shap.LinearExplainer(
                tfidf_model,
                background_vec,
                feature_names=feature_names
            )
        else:
            explainer = shap.LinearExplainer(
                tfidf_model,
                vec,
                feature_names=feature_names
            )

        shap_values = explainer(vec)

        vals = shap_values.values[0]
        print("Max SHAP:", vals.max())
        print("Min SHAP:", vals.min())
        print("SHAP > 0.001:", np.sum(np.abs(vals) > 0.001))

        print(f"Max SHAP: {vals.max():.6f}")
        print(f"Min SHAP: {vals.min():.6f}")

        nonzero_indices = np.where(np.abs(vals) > 0.00001)[0]

        print(f"Features with SHAP > 0.00001: {len(nonzero_indices)}")

        if len(nonzero_indices) == 0:
            print("❌ No significant SHAP features found")
            return []

        sorted_indices = nonzero_indices[
            np.argsort(np.abs(vals[nonzero_indices]))[::-1]
        ]

        top_indices = sorted_indices[:10]

        top_features = []

        for idx in top_indices:

            top_features.append({
                "word": str(feature_names[idx]),
                "shap_value": round(float(vals[idx]), 6),
                "direction": "scam" if vals[idx] > 0 else "normal"
            })

            print(
                f"{feature_names[idx]} : "
                f"{vals[idx]:.6f}"
            )

        return top_features

    except Exception as e:
        print(f"❌ TF-IDF SHAP Error: {e}")
        return []


def get_xlm_shap(text):
    cleaned = clean_text(text)

    def predict_fn(texts):
        texts  = [str(t) for t in texts]
        inputs = xlm_tokenizer(
            texts,
            padding        = True,
            truncation     = True,
            max_length     = XLM_MAX_LEN,
            return_tensors = "pt"
        )
        with torch.no_grad():
            outputs = xlm_model(**inputs)
            probs   = torch.softmax(outputs.logits, dim=1)
        return probs.cpu().numpy()

    masker      = shap.maskers.Text(tokenizer=r"\W+")
    explainer   = shap.Explainer(predict_fn, masker)
    shap_values = explainer([cleaned])

    tokens  = shap_values.data[0]
    values  = shap_values.values[0, :, 1]
    indices = np.argsort(np.abs(values))[-10:][::-1]

    top_features = []
    for idx in indices:
        if idx < len(tokens):
            top_features.append({
                "word"      : str(tokens[idx]),
                "shap_value": round(float(values[idx]), 4),
                "direction" : "scam" if values[idx] > 0 else "normal"
            })
    return top_features


# ==============================================
# SHARED VERDICT LOGIC
# ==============================================
def build_verdict(tfidf_result, xlm_result):
    scam_probs = []
    if tfidf_result:
        scam_probs.append(tfidf_result["probability"])
    if xlm_result:
        scam_probs.append(xlm_result["probability"])

    avg_prob = sum(scam_probs) / len(scam_probs) if scam_probs else 0

    if avg_prob >= 0.8:
        return {
            "final_verdict": "SCAM",
            "risk_level"   : "high",
            "explanation"  : (
                "This call contains strong indicators of fraud including "
                "authority impersonation, urgency markers, or financial "
                "directives commonly used in Malaysian phone scams."
            )
        }
    elif avg_prob >= 0.5:
        return {
            "final_verdict": "SUSPICIOUS",
            "risk_level"   : "medium",
            "explanation"  : (
                "This call contains some patterns that resemble scam "
                "communications. Proceed with caution and verify the "
                "caller's identity independently."
            )
        }
    else:
        return {
            "final_verdict": "NORMAL",
            "risk_level"   : "low",
            "explanation"  : (
                "This call appears to be a legitimate conversation "
                "without significant scam indicators."
            )
        }


# ==============================================
# REQUEST / RESPONSE MODELS
# ==============================================
class PredictRequest(BaseModel):
    text      : str
    use_model : str = "both"


class PredictResponse(BaseModel):
    transcript          : Optional[str] = None
    tfidf_result        : Optional[Dict[str, Any]] = None
    xlm_result          : Optional[Dict[str, Any]] = None
    tfidf_shap          : Optional[List[Dict[str, Any]]] = None
    xlm_shap            : Optional[List[Dict[str, Any]]] = None
    risk_factors        : Optional[List[Dict[str, str]]] = None
    recommended_actions : Optional[List[str]] = None
    final_verdict       : str
    risk_level          : str
    explanation         : str


# ==============================================
# ENDPOINTS
# ==============================================
@app.get("/")
def root():
    return {"message": "Scam Detection API is running"}


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": True}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    text = request.text.strip()

    if not text:
        return PredictResponse(
            final_verdict       = "UNKNOWN",
            risk_level          = "unknown",
            explanation         = "No text provided.",
            risk_factors        = [],
            recommended_actions = []
        )

    tfidf_result = xlm_result = tfidf_shap = xlm_shap = None

    if request.use_model in ["tfidf", "both"]:
        tfidf_result = predict_tfidf(text)
        tfidf_shap   = get_tfidf_shap(text)

    if request.use_model in ["xlm", "both"]:
        xlm_result = predict_xlm(text)
        xlm_shap   = get_xlm_shap(text)

    verdict      = build_verdict(tfidf_result, xlm_result)
    risk_factors = detect_risk_factors(text)
    actions      = generate_recommended_actions(risk_factors)

    print("========== RISK FACTORS ==========")
    print(risk_factors)
    print("========== RECOMMENDED ACTIONS ==========")
    print(actions)

    return PredictResponse(
        tfidf_result        = tfidf_result,
        xlm_result          = xlm_result,
        tfidf_shap          = tfidf_shap,
        xlm_shap            = xlm_shap,
        risk_factors        = risk_factors,
        recommended_actions = actions,
        **verdict
    )


@app.post("/analyze-audio", response_model=PredictResponse)
async def analyze_audio(file: UploadFile = File(...)):
    """
    Upload WAV, MP3 or M4A audio file.
    Converts to WAV, transcribes with Azure Speech,
    then runs scam detection.
    """
    if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a')):
        return PredictResponse(
            final_verdict       = "ERROR",
            risk_level          = "unknown",
            explanation         = "Please upload an audio file in WAV, MP3 or M4A format.",
            risk_factors        = [],
            recommended_actions = []
        )

    tmp_path = None
    wav_path = None

    try:
        # Save uploaded file to temp
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as tmp:
            content  = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Always convert to ensure correct WAV format
        print(f"Converting {file.filename} to standard WAV format...")
        wav_path = convert_to_wav(tmp_path)

        # Transcribe with retry
        print(f"Transcribing: {file.filename}...")
        transcript = transcribe_audio(wav_path)

        if not transcript or len(transcript.split()) < 5:
            return PredictResponse(
                transcript          = transcript,
                risk_factors        = [],
                recommended_actions = [],
                final_verdict       = "UNKNOWN",
                risk_level          = "unknown",
                explanation         = (
                    "Could not analyze — transcript too short or empty. "
                    "Please ensure audio is clear and in Bahasa Malaysia."
                )
            )

        print(f"Transcript: {transcript[:100]}...")

        # Predict
        tfidf_result = predict_tfidf(transcript)
        xlm_result   = predict_xlm(transcript)
        tfidf_shap   = get_tfidf_shap(transcript)
        xlm_shap     = get_xlm_shap(transcript)
        verdict      = build_verdict(tfidf_result, xlm_result)
        risk_factors = detect_risk_factors(transcript)
        actions      = generate_recommended_actions(risk_factors)

        print("========== RISK FACTORS ==========")
        print(risk_factors)
        print("========== RECOMMENDED ACTIONS ==========")
        print(actions)

        return PredictResponse(
            transcript          = transcript,
            tfidf_result        = tfidf_result,
            xlm_result          = xlm_result,
            tfidf_shap          = tfidf_shap,
            xlm_shap            = xlm_shap,
            risk_factors        = risk_factors,
            recommended_actions = actions,
            final_verdict       = verdict["final_verdict"],
            risk_level          = verdict["risk_level"],
            explanation         = verdict["explanation"]
        )

    except subprocess.CalledProcessError as e:
        return PredictResponse(
            final_verdict       = "ERROR",
            risk_level          = "unknown",
            explanation         = f"Audio conversion failed. Please ensure ffmpeg is installed. Error: {str(e)}",
            risk_factors        = [],
            recommended_actions = []
        )

    except Exception as e:
        return PredictResponse(
            final_verdict       = "ERROR",
            risk_level          = "unknown",
            explanation         = f"Error processing audio: {str(e)}",
            risk_factors        = [],
            recommended_actions = []
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        if wav_path and wav_path != tmp_path and os.path.exists(wav_path):
            os.remove(wav_path)
