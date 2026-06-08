# Scam or Not? — AI-Powered Phone Fraud Detection System for Malaysia

## Project Overview
This project develops an AI-based scam call detection system tailored to the Malaysian 
linguistic context. It combines a TF-IDF baseline model and XLM-RoBERTa transformer 
with SHAP explainability, risk factor detection, recommended actions, and a web dashboard 
for real-world demonstration.

**Author:** Chin Yean Yee  
**Institution:** Universiti Malaya — Faculty of Computer Science & Information Technology  
**Year:** 2026

---

## Run Order

| Step | File | Purpose |
|---|---|---|
| 1 | `01_data_preparation.py` | Clean transcripts, build master CSV |
| 2 | `02_generate_synthetic.py` | Generate synthetic normal transcripts |
| 3 | `03_generate_synthetic_scam.py` | Generate synthetic scam transcripts |
| 4 | `04_tfidf_model.py` | TF-IDF + Logistic Regression + SHAP |
| 5 | `06_xlm_roberta_model.py` | XLM-RoBERTa fine-tuning |
| 6 | `05_shap_xlm.py` | SHAP explainability for XLM-RoBERTa |
| 7 | `uvicorn api:app --reload` | Start FastAPI backend |

---

## Setup

### Step 1 — Prerequisites
- Python 3.11.9 (required — does NOT work on Python 3.12+)
- Google Drive for Desktop installed and mounted
- NVIDIA GPU recommended for XLM-RoBERTa training
- ffmpeg (required for audio conversion)

**Install ffmpeg:**

Windows:
```bash
winget install ffmpeg
```
Then restart your terminal and verify:
```bash
ffmpeg -version
```

Mac:
```bash
brew install ffmpeg
```

If `winget` is not available on Windows, download manually from https://ffmpeg.org/download.html and add the `bin` folder to your system PATH.


### Step 2 — Update config.py
Change `BASE_PATH` to match your Google Drive path:
```python
BASE_PATH = r"G:\My Drive\Scam_Source"   # Windows
# BASE_PATH = "/Users/yourname/Google Drive/My Drive/Scam_Source"  # Mac
```

### Step 3 — Create virtual environment
```bash
"C:\Program Files\Python311\python.exe" -m venv venv311
venv311\Scripts\activate
```

### Step 4 — Install dependencies
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 5 — Create .env file
Create a file named `.env` in the project root — never commit this to GitHub:
```
AZURE_SPEECH_KEY=your_azure_speech_key_here
AZURE_SPEECH_REGION=southeastasia
OPENAI_API_KEY=your_openai_api_key_here
```

| Variable | Description | Where to get |
|---|---|---|
| `AZURE_SPEECH_KEY` | Azure Speech to Text API key | https://portal.azure.com |
| `AZURE_SPEECH_REGION` | Azure region e.g. southeastasia | Azure Portal |
| `OPENAI_API_KEY` | OpenAI key for synthetic data generation | https://platform.openai.com/api-keys |

### Step 6 — Verify GPU
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Step 7 — Start API backend
```bash
uvicorn api:app --reload
```
- API runs at: http://127.0.0.1:8000  
- API docs at: http://127.0.0.1:8000/docs

### Step 8 — Expose API with ngrok
Open a NEW terminal (keep API terminal running) and run:
```bash
ngrok http 8000
```
Copy the https forwarding URL (e.g. https://abc123.ngrok-free.app)  
Update this URL in your Lovable frontend code wherever `API_URL` is defined.

Note: ngrok URL changes every time you restart — update Lovable code each session.

### Step 9 — Open dashboard
Open your Lovable dashboard URL in browser.  
Ensure both API server and ngrok are running before analyzing calls.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | Health check with model status |
| POST | `/predict` | Predict from raw text input |
| POST | `/analyze-audio` | Upload audio, transcribe, then predict |

### Response Fields

| Field | Type | Description |
|---|---|---|
| `transcript` | string | Azure STT transcript (audio endpoint only) |
| `tfidf_result` | object | TF-IDF model label, probability, verdict |
| `xlm_result` | object | XLM-RoBERTa label, probability, verdict |
| `tfidf_shap` | array | Top 10 SHAP features from TF-IDF |
| `xlm_shap` | array | Top 10 SHAP features from XLM-RoBERTa |
| `risk_factors` | array | Detected scam risk factors with descriptions |
| `recommended_actions` | array | Actionable advice based on detected risk factors |
| `final_verdict` | string | SCAM / SUSPICIOUS / NORMAL |
| `risk_level` | string | high / medium / low |
| `explanation` | string | Human-readable explanation of verdict |

---

## Risk Factor Detection

The system detects five scam risk factor categories from the transcript:

| Risk Factor | Trigger Keywords |
|---|---|
| Authority Impersonation | polis, pdrm, pegawai, sprm, lhdn, mahkamah, bank negara |
| Financial Crime Allegation | jenayah, penipuan, sindiket, aktiviti haram |
| Money Transfer Request | akaun, pindahkan, transfer, pemindahan |
| OTP / TAC Request | otp, tac, verification code, kod pengesahan |
| Urgency / Threat Tactics | waran, tangkap, ditahan, 24 jam, segera |

Each detected risk factor generates a corresponding recommended action in the response.

---

## Azure Speech Reliability Improvements

The `transcribe_audio()` function includes several hardening measures to reduce Azure
Speech intermittency from ~85-90% success rate to ~98-99%:

**1. Silence timeout configuration** — prevents Azure from stopping early on calls that
begin with ringing or silence before speech:
```python
speech_config.set_property(
    speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "10000"
)
speech_config.set_property(
    speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "3000"
)
```

**2. Session timeout guard** — prevents infinite hang if `session_stopped` never fires:
```python
timeout = 60  # seconds max wait
elapsed = 0
while not done and elapsed < timeout:
    time.sleep(0.5)
    elapsed += 0.5
```

**3. Post-stop flush delay** — `stop_continuous_recognition()` is non-blocking; a 1-second
sleep ensures Azure flushes the final recognized segment before the transcript is read:
```python
recognizer.stop_continuous_recognition()
time.sleep(1)
```

**4. Exponential backoff retry** — on empty transcript, waits longer between each attempt
rather than a fixed 2 seconds:
```python
wait_time = (attempt + 1) * 5  # 5s, 10s, 15s
```

---

## Notes

- **API backend** — `api.py` must be running before using the dashboard
- **ngrok** — required to connect Lovable hosted frontend to local API; free tier URL changes on every restart
- **Audio format** — WAV recommended for best Azure ASR accuracy
- **Two terminals needed** — one for uvicorn, one for ngrok
- **Synthetic data disclosure** — synthetic transcripts must be disclosed in Section 3.2 of the research report as required by academic ethics
- **Python version** — XLM-RoBERTa training requires Python 3.11; PyTorch CUDA does not support Python 3.12+
- **API keys** — never hardcode keys in source files; always use `.env`
- **SHAP** — explanations use real validation samples, not hardcoded text
- **GPU** — XLM-RoBERTa training tested on NVIDIA RTX 4060 Laptop GPU (8GB VRAM)

---

## Data Folder Structure
All data files stay in Google Drive — do not move them to local storage.
```
G:\My Drive\Scam_Source
├── Scam_Transcript\               — 140 real scam transcripts
├── Synthetic_Scam_Transcript\     — 30 synthetic scam transcripts
├── Raw_Normal_MagicHub\           — Raw MagicHub conversational audio transcripts
├── Cleaned_Convo_Transcript\      — Cleaned MagicHub + synthetic casual (50 files)
├── Cleaned_CallCenter_Transcript\ — Synthetic call centre transcripts (50 files)
├── Scam_Audio\                    — Raw scam audio files (WAV)
├── Saved_Models\                  — Trained TF-IDF and XLM-RoBERTa model files
├── master_scam_data.csv           — Final dataset (270 samples)
└── scam_video_tracker.csv         — YouTube download tracker
```

---

## Dataset Summary

| Source | Type | Label | Count |
|---|---|---|---|
| YouTube recordings | Real scam | 1 | 140 |
| GPT-4o-mini generated | Synthetic scam | 1 | 30 |
| MagicHub corpus | Real normal | 0 | 20 |
| GPT-4o-mini generated | Synthetic casual | 0 | 30 |
| GPT-4o-mini generated | Synthetic call centre | 0 | 50 |
| **Total** | | | **270** |

---

## Model Results

| Metric | TF-IDF + LR | XLM-RoBERTa |
|---|---|---|
| Accuracy | 98.15% | 97.97% |
| Scam Precision | 0.9714 | 0.9884 |
| Scam Recall | 1.0000 | 0.9835 |
| Scam F1 | 0.9855 | 0.9859 |

---

## To Run
1. `venv311\Scripts\activate`
2. Terminal 1: `uvicorn api:app --reload`
3. Terminal 2: `ngrok http 8000`
4. Open Lovable dashboard URL in browser
