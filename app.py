import streamlit as st
import pandas as pd
import time
import textwrap
from collections import Counter
import os
from evaluator import extract_text_from_pdf, evaluate_resume, evaluate_resumes_batch, ResumeEvaluation, GROQ_MODEL

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Deployment Debug Info ───────────────────────────────────────────────────
api_key_present = bool(os.getenv("GROQ_API_KEY"))
st.sidebar.caption(f"GROQ_API_KEY: {'present' if api_key_present else 'missing'}")
st.sidebar.caption(f"Model: {GROQ_MODEL}")

# ── Premium CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');


/* ── Font: apply Outfit ONLY to content elements, NOT to icon spans ── */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}
p, h1, h2, h3, h4, h5, h6, label, button, textarea, input,
.stMarkdown, .stMarkdown p, .stMarkdown li,
.stButton > button, .stTextArea textarea, .stTextInput input {
    font-family: 'Outfit', sans-serif !important;
}
/* Dark text defaults for Streamlit content elements */
.stMarkdown, .stMarkdown p, .stMarkdown li,
.stText, [data-testid="stText"],
.stAlert p, .stDataFrame, .stTable,
div[data-testid="stExpander"] p,
div[data-testid="stExpander"] pre {
    color: #111827 !important;
}
/* Remove excess Streamlit top padding */
.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}
.stApp {
    background: linear-gradient(135deg, #F0F4FF 0%, #EEF2FF 40%, #F5F0FF 100%);
    min-height: 100vh;
}

/* ── Fix: Radio button (Cards/Table) labels ── */
[data-testid="stRadio"] label p,
div[role="radiogroup"] label p {
    color: #1E293B !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

/* ── Fix: Step labels and section headings ── */
.step-label, .results-header, .results-sub, .list-section-title {
    color: #1E293B !important;
}

/* ── 1. Dropzone area: light background + dashed border ── */
section[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.85) !important;
    border: 2px dashed #C7D2FE !important;
    border-radius: 12px !important;
}
/* Dropzone instructional text (not the button) */
section[data-testid="stFileUploaderDropzone"] * {
    color: #374151 !important;
    background: transparent !important;
    opacity: 1 !important;
}
/* ── 2. Browse files button — keep it styled and visible ── */
section[data-testid="stFileUploaderDropzone"] button {
    background: #4F46E5 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 18px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    cursor: pointer !important;
    opacity: 1 !important;
}
section[data-testid="stFileUploaderDropzone"] button:hover {
    background: #4338CA !important;
}
/* ── 3. Uploaded file list (sibling below dropzone) — dark text ── */
[data-testid="stFileUploaderFile"] *,
[data-testid="stFileUploaderFile"] span,
[data-testid="stFileUploaderFile"] small,
[data-testid="stFileUploaderFile"] p,
.uploadedFileName,
.uploadedFileData *,
div[data-testid="stFileUploaderFile"] {
    color: #111827 !important;
    opacity: 1 !important;
    font-weight: 500 !important;
}
/* Pagination "Showing page X of Y" */
[data-testid="stFileUploader"] {
    padding-bottom: 30px !important;
}
[data-testid="stFileUploader"] > div:last-child > small,
[data-testid="stFileUploader"] > div[data-testid="stBaseButton-secondary"] ~ div small {
    color: #374151 !important;
    opacity: 1 !important;
    font-weight: 500 !important;
}

/* ── Fix: Progress bar text ── */
[data-testid="stProgress"] p,
div[data-testid="stProgressBar"] + div > p,
div.stProgress span,
[data-testid="stSpinner"] p,
[data-testid="stLoadingBlock"] p {
    color: #111827 !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* ── Fix: Expander — prevent arrow_right icon text from bleeding ── */
/* The icon is a Material Icons ligature span — do NOT override its font-family */
[data-testid="stExpander"] details > summary {
    display: flex;
    align-items: center;
    gap: 6px;
    color: #111827 !important;
    font-weight: 600 !important;
}
/* Label text inside summary */
[data-testid="stExpander"] details > summary > div > p {
    color: #111827 !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    font-family: 'Outfit', sans-serif !important;
}

/* ── Fix: Extracted text block in expander ── */
[data-testid="stExpander"] details > div {
    border: 2px dashed #CBD5E1 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    margin-top: 8px !important;
    background: #F8FAFC !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.02) !important;
}
[data-testid="stExpander"] details > div * {
    color: #111827 !important;
    font-size: 0.87rem !important;
    line-height: 1.7 !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
}
    background: #F8FAFC !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.02) !important;
}
[data-testid="stExpander"] details > div * {
    color: #111827 !important;
    font-size: 0.87rem !important;
    line-height: 1.7 !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
}
[data-testid="stExpander"] .streamlit-expanderContent {
    /* Removed redundant border styles */
}

/* ── Hide chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero Section ── */
.hero-wrapper {
    padding: 8px 8px 20px;
    position: relative;
}
.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1.1;
    color: #0F172A;
    margin: 0 0 0;
    text-align: left;
}
.hero-title span {
    background: linear-gradient(110deg, #4F46E5, #7C3AED);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #374151;
    font-weight: 500;
    text-align: center;
    line-height: 1.65;
    margin: 16px 0 28px;
}

/* ── Step Labels ── */
.step-label {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1rem;
    font-weight: 700;
    color: #1E293B;
    margin-bottom: 12px;
    letter-spacing: -0.01em;
}
.step-num {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: linear-gradient(135deg, #4F46E5, #7C3AED);
    color: white;
    font-size: 0.8rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(79, 70, 229, 0.3);
}

/* ── Input Panel Glass Card ── */
.input-card {
    background: rgba(255,255,255,0.82);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.8);
    border-radius: 20px;
    padding: 28px;
    box-shadow: 0 8px 32px rgba(79,70,229,0.06), 0 2px 8px rgba(0,0,0,0.04);
}

/* ── Divider ── */
.section-divider {
    border: none;
    border-top: 1px solid #E2E8F0;
    margin: 40px 0;
}

/* ── Analyze Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    color: white;
    border-radius: 14px;
    padding: 15px 28px;
    font-weight: 700;
    font-size: 1.1rem;
    border: none;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 20px rgba(79, 70, 229, 0.35);
    letter-spacing: -0.01em;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(79, 70, 229, 0.45);
    filter: brightness(1.05);
}
.stButton > button:active { transform: translateY(0px); }

/* ── Results Header ── */
.results-header {
    font-size: 1.7rem;
    font-weight: 800;
    color: #0F172A;
    letter-spacing: -0.04em;
    margin-bottom: 4px;
}
.results-sub {
    font-size: 0.95rem;
    color: #374151;
    margin-bottom: 28px;
}

/* ── Candidate Cards ── */
@keyframes cardIn {
    0%   { opacity: 0; transform: translateY(24px) scale(0.98); }
    100% { opacity: 1; transform: translateY(0)    scale(1); }
}
.candidate-card {
    background: rgba(255,255,255,0.88);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.9);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 20px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.06), 0 2px 6px rgba(0,0,0,0.04);
    transition: transform 0.3s cubic-bezier(0.4,0,0.2,1), box-shadow 0.3s cubic-bezier(0.4,0,0.2,1);
    animation: cardIn 0.5s ease-out forwards;
    position: relative;
    overflow: hidden;
}
.candidate-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 20px 20px 0 0;
}
.candidate-card.strong::before  { background: linear-gradient(90deg, #22C55E, #16A34A); }
.candidate-card.moderate::before { background: linear-gradient(90deg, #F59E0B, #D97706); }
.candidate-card.notfit::before  { background: linear-gradient(90deg, #EF4444, #DC2626); }
.candidate-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 20px 50px rgba(0,0,0,0.1), 0 4px 12px rgba(0,0,0,0.06);
}

/* Rank badge */
.rank-badge {
    position: absolute;
    top: 20px; right: 20px;
    width: 36px; height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, #F8FAFC, #EEF2FF);
    border: 1px solid #E2E8F0;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 700; color: #374151;
}

/* Score dial */
.score-ring {
    width: 80px; height: 80px;
    border-radius: 50%;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
.score-ring.strong  { background: linear-gradient(135deg, #DCFCE7, #BBF7D0); border: 2px solid #86EFAC; }
.score-ring.moderate { background: linear-gradient(135deg, #FEF9C3, #FEF08A); border: 2px solid #FDE047; }
.score-ring.notfit  { background: linear-gradient(135deg, #FEE2E2, #FECACA); border: 2px solid #FCA5A5; }
.score-num { font-size: 1.7rem; font-weight: 800; line-height: 1; letter-spacing: -0.04em; }
.score-unit { font-size: 0.65rem; font-weight: 700; color: #4B5563; letter-spacing: 0.02em; }
.score-num.strong  { color: #15803D; }
.score-num.moderate { color: #B45309; }
.score-num.notfit  { color: #B91C1C; }

/* Rec badge */
.rec-chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.82rem; font-weight: 700;
}
.rec-chip.strong  { background: #DCFCE7; color: #15803D; border: 1px solid #86EFAC; }
.rec-chip.moderate { background: #FEF9C3; color: #B45309; border: 1px solid #FDE047; }
.rec-chip.notfit  { background: #FEE2E2; color: #B91C1C; border: 1px solid #FCA5A5; }

/* Candidate name & file */
.cand-name { font-size: 1.25rem; font-weight: 700; color: #0F172A; letter-spacing: -0.02em; margin: 0 0 4px; }
.cand-file { font-size: 0.82rem; color: #4B5563; font-weight: 500; margin: 0 0 10px; }

/* Section pills */
.list-section-title {
    display: flex; align-items: center; gap: 7px;
    font-size: 0.9rem; font-weight: 700; color: #1E293B;
    margin-bottom: 10px; margin-top: 0;
    letter-spacing: -0.01em;
}
.list-section-title .pill {
    width: 22px; height: 22px; border-radius: 50%;
    font-size: 0.75rem;
    display: flex; align-items: center; justify-content: center;
}
.pill-green { background: #DCFCE7; }
.pill-yellow { background: #FEF9C3; }

.point-list { list-style: none; padding: 0; margin: 0; }
.point-list li {
    padding: 8px 12px;
    background: #F8FAFC;
    border-radius: 8px;
    font-size: 0.88rem;
    color: #111827;
    margin-bottom: 6px;
    line-height: 1.5;
    border-left: 3px solid transparent;
}
.point-list.strength-list li { border-left-color: #22C55E; }
.point-list.gap-list li     { border-left-color: #F59E0B; }

/* Top match glow */
.top-match-label {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 0.75rem; font-weight: 700;
    background: linear-gradient(135deg, #FEF9C3, #FEF08A);
    color: #854D0E;
    border: 1px solid #FDE047;
    border-radius: 999px;
    padding: 3px 10px;
    margin-left: 8px;
    vertical-align: middle;
}

/* Columns separator */
.col-divider {
    width: 1px;
    background: #E2E8F0;
    align-self: stretch;
    margin: 0 4px;
}
</style>
""", unsafe_allow_html=True)

# ── State ─────────────────────────────────────────────────────────────────────
if 'results' not in st.session_state:
    st.session_state.results = []

# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-wrapper'>
    <h1 class='hero-title'><span>TalentLens </span></h1>
    <p class='hero-subtitle'>Upload resumes and a job description. Our AI instantly evaluates, scores, and ranks every candidate so you can focus on the best ones.</p>
</div>
""", unsafe_allow_html=True)

# ── Input Panel ───────────────────────────────────────────────────────────────
col_jd, col_up = st.columns([1, 1], gap="large")

with col_jd:
    st.markdown("""
    <div class='step-label'>
        <div class='step-num'>1</div>Paste the Job Description
    </div>""", unsafe_allow_html=True)
    job_description = st.text_area(
        "job_description",
        height=260,
        placeholder="e.g.\nWe are looking for a Senior Python Developer with 5+ years of experience in building REST APIs, cloud infrastructure (AWS/GCP), and experience with ML pipelines...",
        label_visibility="collapsed"
    )

with col_up:
    st.markdown("""
    <div class='step-label'>
        <div class='step-num'>2</div>Upload Candidate Resumes (PDF)
    </div>""", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "resumes",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

# ── Action Button ─────────────────────────────────────────────────────────────
st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])

with btn_col:
    analyze_clicked = st.button("🔍  Analyze Candidates", type="primary", use_container_width=True)

# ── Processing ────────────────────────────────────────────────────────────────
if analyze_clicked:
    if not job_description.strip():
        st.error("⚠️  Please paste a job description first.")
    elif not uploaded_files:
        st.error("⚠️  Please upload at least one resume (PDF).")
    else:
        st.session_state.results = []
        my_bar = st.progress(0, text="Starting analysis...")

        try:
            my_bar.progress(0.1, text="📄  Reading and preprocessing resumes…")

            # Collect all (filename, pdf_bytes) tuples
            resume_payloads = [(f.name, f.read()) for f in uploaded_files]

            def status_cb(msg):
                my_bar.progress(0.4, text=msg)

            my_bar.progress(0.2, text=f"🤖  Sending {len(resume_payloads)} resume(s) to AI in optimised batch…")
            results_list = evaluate_resumes_batch(job_description, resume_payloads,
                                                  status_callback=status_cb)

            st.session_state.results = results_list

            my_bar.progress(1.0, text="✅  Analysis complete!")
            time.sleep(0.7)
            my_bar.empty()

            # Stable sort: primary = score desc
            st.session_state.results.sort(key=lambda x: x["score"], reverse=True)

        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "rate" in err_msg.lower():
                st.error("⚠️ API rate limit hit. Please wait 15–20 seconds and try again.")
            else:
                st.error(f"An error occurred: {err_msg}")
            my_bar.empty()

# ── Results Dashboard ─────────────────────────────────────────────────────────
if st.session_state.results:
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    results = st.session_state.results
    total = len(results)
    strong_count = sum(1 for r in results if r["recommendation"] == "Strong Fit")
    avg_score = round(sum(r["score"] for r in results) / total)

    # Summary stats row
    s1, s2, s3 = st.columns(3)
    def stat_card(label, value, bg, color):
        return f"""<div style='background:{bg};border-radius:14px;padding:18px 22px;text-align:center;border:1px solid {color}22;'>
            <div style='font-size:2rem;font-weight:800;color:{color};letter-spacing:-0.04em;'>{value}</div>
            <div style='font-size:0.85rem;color:#64748B;font-weight:500;margin-top:4px;'>{label}</div>
        </div>"""

    with s1:
        st.markdown(stat_card("Candidates Analyzed", total, "#EEF2FF", "#4F46E5"), unsafe_allow_html=True)
    with s2:
        st.markdown(stat_card("Strong Fits Found", strong_count, "#F0FDF4", "#16A34A"), unsafe_allow_html=True)
    with s3:
        st.markdown(stat_card("Average Match Score", f"{avg_score}/100", "#FFFBEB", "#D97706"), unsafe_allow_html=True)

    source_counts = Counter(r.get("source", "api") for r in results)
    source_summary = ", ".join(f"{k.upper()}={v}" for k, v in source_counts.items())
    st.caption(f"Evaluation sources: {source_summary}")

    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)

    # View toggle
    h_col, v_col = st.columns([3, 1])
    with h_col:
        st.markdown(f"<div class='results-header'>🏆 Candidate Rankings</div><div class='results-sub'>Sorted by AI match score — highest first.</div>", unsafe_allow_html=True)
    with v_col:
        view_mode = st.radio("view", ["Cards", "Table"], horizontal=True, label_visibility="collapsed")

    # ── Card View ──
    if view_mode == "Cards":
        for idx, result in enumerate(results):
            rec = result["recommendation"]
            cls = "strong" if rec == "Strong Fit" else ("moderate" if rec == "Moderate Fit" else "notfit")
            icon = "🟢" if rec == "Strong Fit" else ("🟡" if rec == "Moderate Fit" else "🔴")
            is_top = (idx == 0 and result["score"] >= 70)
            source = (result.get("source") or "api").lower()

            top_label = "<span class='top-match-label'>🏅 Top Match</span>" if is_top else ""

            strengths_html = "".join([f"<li>{s}</li>" for s in result["strengths"]])
            gaps_html      = "".join([f"<li>{g}</li>" for g in result["gaps"]])

            source_badge = ""
            if source in ("fallback", "filter"):
                source_badge = f"<span style='margin-left:10px;font-size:0.75rem;font-weight:700;color:#92400E;background:#FFFBEB;border:1px solid #FDE68A;padding:3px 10px;border-radius:999px;'>⚠️ {source.upper()}</span>"
            elif source == "cache":
                source_badge = f"<span style='margin-left:10px;font-size:0.75rem;font-weight:700;color:#065F46;background:#ECFDF5;border:1px solid #6EE7B7;padding:3px 10px;border-radius:999px;'>⚡ CACHE</span>"

            card = f"""
<div class='candidate-card {cls}'>
    <div class='rank-badge'>#{idx+1}</div>
    <div style='display:flex;align-items:flex-start;gap:20px;'>
        <div class='score-ring {cls}'>
            <span class='score-num {cls}'>{result['score']}</span>
            <span class='score-unit'>/ 100</span>
        </div>
        <div style='flex:1;min-width:0;'>
            <div style='display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:4px;'>
                <span class='cand-name'>{result['candidate_name']}</span>{top_label}{source_badge}
            </div>
            <div class='cand-file'>📄 {result['filename']}</div>
            <span class='rec-chip {cls}'>{icon} {rec}</span>
        </div>
    </div>
    <div style='display:flex;gap:20px;margin-top:24px;'>
        <div style='flex:1;'>
            <div class='list-section-title'><span class='pill pill-green'>✅</span>Key Strengths</div>
            <ul class='point-list strength-list'>{strengths_html}</ul>
        </div>
        <div style='width:1px;background:#E2E8F0;'></div>
        <div style='flex:1;'>
            <div class='list-section-title'><span class='pill pill-yellow'>⚠️</span>Potential Gaps</div>
            <ul class='point-list gap-list'>{gaps_html}</ul>
        </div>
    </div>
</div>"""
            st.markdown(card.replace('\n', ''), unsafe_allow_html=True)
            if source == "fallback":
                api_error = result.get("api_error", "")
                if isinstance(api_error, str) and api_error.strip():
                    st.warning(f"LLM error (used fallback): {api_error[:200]}")
            with st.expander("🔍 View Extracted Text (For debugging AI scores)"):
                st.text(result.get("extracted_text", "No text available"))

    # ── Table View ──
    else:
        df_data = []
        for r in results:
            df_data.append({
                "Rank": f"#{results.index(r)+1}",
                "Candidate": r["candidate_name"],
                "Score": r["score"],
                "Recommendation": r["recommendation"],
                "Source": (r.get("source") or "api").upper(),
                "Top Strength": r["strengths"][0] if r["strengths"] else "—",
                "Top Gap": r["gaps"][0] if r["gaps"] else "—",
            })

        df = pd.DataFrame(df_data)
        st.dataframe(
            df,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score", format="%d", min_value=0, max_value=100
                ),
                "Recommendation": st.column_config.TextColumn("Recommendation"),
                "Source": st.column_config.TextColumn("Source"),
            },
            hide_index=True,
            use_container_width=True
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:48px 0 24px;color:#CBD5E1;font-size:0.82rem;'>
    AI Resume Screener &nbsp;·&nbsp; Powered by Gemini 2.5 Flash &nbsp;·&nbsp; Built for speed & clarity
</div>
""", unsafe_allow_html=True)
