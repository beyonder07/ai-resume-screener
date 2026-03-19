# AI Resume Screening System - MVP

An automated, AI-powered pipeline that instantly evaluates candidate resumes against a given Job Description. Built as an MVP for the AI Automation Intern Assessment.

## 🎯 Problem Solved (Problem 1)
Recruiters spend hours manually scanning resumes for initial shortlisting. This system completely automates the top-of-funnel screening by extracting PDF text and using an LLM to objectively rank candidates based on skill match and experience.

## ✨ Features
- **Drag-and-Drop Interface**: Upload multiple PDF resumes at once.
- **Instant AI Analysis**: Uses Google Gemini (via OpenAI compatibility layer) to evaluate candidates.
- **Structured Output**: Enforces strict JSON responses ensuring consistent scoring, strengths, gaps, and recommendations.
- **Beautiful Dashboard**: Custom-styled UI with Candidate Cards, colored badges (Strong Fit 🟢, Moderate Fit 🟡, Not Fit 🔴), and a toggleable Data Table view.

## 🛠️ Tools Used
- **Backend/Frontend**: Python & Streamlit
- **PDF Extraction**: PyMuPDF (`fitz`)
- **AI Brain**: Google Gemini 2.5 Flash (`generativelanguage.googleapis.com`)
- **Data Structuring**: Pydantic (Strict JSON Schemas)

## 🚀 How to Run Locally

1. **Clone the repository and enter the directory**
```bash
git clone <your-repo-link>
cd resume-screener
```

2. **Create and activate a virtual environment**
```bash
python -m venv venv
venv\\Scripts\\activate  # On Windows
# source venv/bin/activate  # On Mac/Linux
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Add API Key**
Create a `.env` file in the root directory and add your Google Gemini API Key:
```env
GEMINI_API_KEY=your_api_key_here
```
*(Note: The system contains a built-in mock fallback feature. If the API key is missing or out of quota, the UI will still safely generate a mock response so the functionality can be evaluated!)*

5. **Run the App**
```bash
streamlit run app.py
```

## 🧠 Approach Explanation (For Submission Form)
*This project tackles the top-of-funnel hiring bottleneck. I built a highly practical, single-application MVP using Streamlit for an immediate, non-technical HR interface. For the intelligence layer, I integrated the Google Gemini API (via the OpenAI SDK compatibility layer) because it is exceptionally fast and supports strictly structured Pydantic JSON outputs. This guarantees the AI evaluates every candidate objectively and returns exactly the schema required: a Match Score, explicit Strengths/Gaps, and a Fit Recommendation, eliminating prompt-drift.*

*I specifically avoided over-engineering with a heavy React/FastAPI stack, choosing instead to write custom CSS inside Streamlit. This achieved a premium, glassmorphism-inspired UI while keeping the codebase lightweight, Pythonic, and incredibly easy for a company to deploy or maintain tomorrow.*
