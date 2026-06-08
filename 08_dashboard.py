# ==============================================
# 08_dashboard.py
# Streamlit dashboard for scam detection
# Run: streamlit run 08_dashboard.py
# ==============================================

import streamlit as st
import requests
import tempfile
import os
import plotly.graph_objects as go

# ==============================================
# CONFIG
# ==============================================
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title = "Scam or Not?",
    page_icon  = "🔍",
    layout     = "wide"
)

# ==============================================
# CUSTOM CSS
# ==============================================
st.markdown("""
<style>
    .main { background-color: #0d0d0d; }
    .stApp { background-color: #0d0d0d; color: white; }

    .header-title {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 0;
    }
    .header-scam { color: #e63946; }
    .header-normal { color: white; }

    .subtitle {
        text-align: center;
        color: #aaaaaa;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }

    .badge {
        background-color: #1a1a2e;
        border: 1px solid #e63946;
        color: #e63946;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        text-align: center;
        width: fit-content;
        margin: 0 auto;
        display: block;
    }

    .verdict-scam {
        background-color: #e63946;
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        font-size: 1.8rem;
        font-weight: bold;
        text-align: center;
    }

    .verdict-normal {
        background-color: #2d6a4f;
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        font-size: 1.8rem;
        font-weight: bold;
        text-align: center;
    }

    .verdict-suspicious {
        background-color: #e9c46a;
        color: black;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        font-size: 1.8rem;
        font-weight: bold;
        text-align: center;
    }

    .step-card {
        background-color: #1a1a1a;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        height: 150px;
    }

    .footer {
        text-align: center;
        color: #666;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid #222;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================
# SESSION STATE INIT
# Clears results when new file is uploaded
# ==============================================
if "last_filename" not in st.session_state:
    st.session_state.last_filename = None
if "result_data" not in st.session_state:
    st.session_state.result_data = None

# ==============================================
# HEADER
# ==============================================
st.markdown("""
<div style='text-align:center; padding: 2rem 0 1rem 0;'>
    <div class='header-title'>
        <span class='header-scam'>Scam</span>
        <span class='header-normal'> or Not?</span>
    </div>
    <div class='subtitle'>AI-Powered Phone Fraud Detection System for Malaysia</div>
    <div class='badge'>🔬 Powered by XLM-RoBERTa + TF-IDF</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==============================================
# HOW IT WORKS
# ==============================================
st.markdown("### How It Works")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='step-card'>
        <div style='font-size:2rem'>📞</div>
        <strong>1. Record</strong>
        <p style='color:#aaa; font-size:0.9rem'>Record your suspicious call using any call recorder app</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='step-card'>
        <div style='font-size:2rem'>⬆️</div>
        <strong>2. Upload</strong>
        <p style='color:#aaa; font-size:0.9rem'>Upload the recording here for AI analysis</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='step-card'>
        <div style='font-size:2rem'>⚡</div>
        <strong>3. Verdict</strong>
        <p style='color:#aaa; font-size:0.9rem'>Get instant verdict with explanation</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==============================================
# INPUT SECTION
# ==============================================
st.markdown("### Upload Call Recording")

uploaded_file = st.file_uploader(
    "Upload your call recording",
    type = ["wav", "mp3", "m4a"],
    help = "Supported formats: WAV, MP3, M4A • Max 50MB"
)

# Clear results when new file is uploaded
if uploaded_file:
    if uploaded_file.name != st.session_state.last_filename:
        st.session_state.result_data   = None
        st.session_state.last_filename = uploaded_file.name
    st.success(f"✅ File uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

analyze_btn = st.button("🔍 Analyze Now", type="primary", use_container_width=True)

# ==============================================
# ANALYSIS
# ==============================================
if analyze_btn and uploaded_file:
    # Clear previous results
    st.session_state.result_data = None

    with st.spinner("🎤 Transcribing audio... this may take 1-2 minutes"):
        try:
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            with open(tmp_path, 'rb') as f:
                response = requests.post(
                    f"{API_URL}/analyze-audio",
                    files   = {"file": (uploaded_file.name, f, "audio/wav")},
                    timeout = 300
                )

            os.remove(tmp_path)

            if response.status_code == 200:
                st.session_state.result_data = response.json()
            else:
                st.error(f"API error: {response.status_code}")
                st.stop()

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Please make sure `uvicorn api:app --reload` is running.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.stop()

elif analyze_btn and not uploaded_file:
    st.warning("⚠️ Please upload an audio file first.")

# ==============================================
# RESULTS — only shown when data exists
# ==============================================
if st.session_state.result_data:
    data = st.session_state.result_data

    st.markdown("---")
    st.markdown("### Results")

    # Transcript
    if data.get("transcript"):
        with st.expander("📄 Auto-Generated Transcript", expanded=False):
            st.write(data["transcript"])

    # Verdict banner
    verdict = data.get("final_verdict", "UNKNOWN")
    risk    = data.get("risk_level", "unknown")

    if verdict == "SCAM":
        st.markdown("""
        <div class='verdict-scam'>⚠️ SCAM DETECTED</div>
        """, unsafe_allow_html=True)
    elif verdict == "NORMAL":
        st.markdown("""
        <div class='verdict-normal'>✅ NORMAL CALL</div>
        """, unsafe_allow_html=True)
    elif verdict == "SUSPICIOUS":
        st.markdown("""
        <div class='verdict-suspicious'>⚠️ SUSPICIOUS CALL</div>
        """, unsafe_allow_html=True)
    else:
        st.warning(f"Result: {verdict}")

    # Risk badge
    risk_colors = {"high": "🔴", "medium": "🟡", "low": "🟢"}
    st.markdown(f"**Risk Level:** {risk_colors.get(risk, '⚪')} {risk.upper()}")

    # Explanation
    st.info(data.get("explanation", ""))

    st.markdown("---")

    # Model cards
    st.markdown("### Model Results")
    col1, col2 = st.columns(2)

    with col1:
        tfidf = data.get("tfidf_result")
        if tfidf:
            verdict_color = "🔴" if tfidf["verdict"] == "SCAM" else "🟢"
            st.markdown("**TF-IDF Model**")
            st.markdown(f"{verdict_color} **{tfidf['verdict']}**")
            confidence = tfidf["probability"] * 100
            st.markdown(f"Confidence: **{confidence:.1f}%**")
            st.progress(tfidf["probability"])

    with col2:
        xlm = data.get("xlm_result")
        if xlm:
            verdict_color = "🔴" if xlm["verdict"] == "SCAM" else "🟢"
            st.markdown("**XLM-RoBERTa Model**")
            st.markdown(f"{verdict_color} **{xlm['verdict']}**")
            confidence = xlm["probability"] * 100
            st.markdown(f"Confidence: **{confidence:.1f}%**")
            st.progress(xlm["probability"])

    st.markdown("---")

    # ==============================================
    # SHAP EXPLANATION
    # ==============================================
    st.markdown("### Why did the AI make this decision?")
    st.caption("The highlighted words influenced the AI prediction")

    tab1, tab2 = st.tabs(["TF-IDF Explanation", "XLM-RoBERTa Explanation"])

    def plot_shap(shap_data, title):
        if not shap_data:
            st.info("No explanation data available.")
            return

        shap_data = [item for item in shap_data
                     if abs(item["shap_value"]) > 0.001]

        if not shap_data:
            st.info("No significant features found for this transcript "
                    "in this model's vocabulary.")
            return

        words  = [item["word"] for item in shap_data]
        values = [item["shap_value"] for item in shap_data]
        colors = ["#e63946" if v > 0 else "#4361ee" for v in values]

        fig = go.Figure(go.Bar(
            x            = values,
            y            = words,
            orientation  = 'h',
            marker_color = colors
        ))

        fig.update_layout(
            title         = title,
            xaxis_title   = "SHAP Value",
            yaxis_title   = "Token",
            height        = 500,
            paper_bgcolor = "#1a1a1a",
            plot_bgcolor  = "#1a1a1a",
            font          = dict(color="white", size=13),
            margin        = dict(l=120),
            xaxis         = dict(gridcolor="#333"),
            yaxis         = dict(gridcolor="#333", autorange="reversed")
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔴 Red = pushes toward SCAM   🔵 Blue = pushes toward NORMAL")

    with tab1:
        plot_shap(data.get("tfidf_shap"), "TF-IDF Feature Importance")

    with tab2:
        plot_shap(data.get("xlm_shap"), "XLM-RoBERTa Token Importance")

# ==============================================
# FOOTER
# ==============================================
st.markdown("""
<div class='footer'>
    <p>Universiti Malaya — Faculty of Computer Science & Information Technology</p>
    <p>Research Project 2026 — Chin Yean Yee</p>
</div>
""", unsafe_allow_html=True)