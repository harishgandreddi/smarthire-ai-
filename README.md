# 🤖 SmartHire AI
### Intelligent Candidate Discovery & Ranking Engine
> Built for **INDIA RUNS Hackathon** — Track 01: Data & AI Challenge | Hack2skill × Redrob

---

## 🎯 Problem Statement
Companies miss out on thousands of brilliant candidates because old-school recruitment tools rely on basic keyword matching. SmartHire AI solves this by understanding candidates the way a great recruiter would — not by matching keywords, but by actually understanding who fits the role.

---

## 🚀 What It Does
SmartHire AI is an end-to-end intelligent candidate ranking system that:
- Reads a job description and **semantically understands** what the role needs
- Evaluates the **full candidate profile** — career history, skills, behavioral signals, platform activity
- Delivers a **ranked shortlist of 100 candidates** with explainable scores
- Shows **strengths, gaps, missing skills** and a recruiter-style summary for every candidate

---

## 🏗️ System Architecture

```
Job Description + candidates.jsonl (487MB)
        ↓
   [Parser] JSON extraction → structured profiles
        ↓
   [Embedding Engine] sentence-transformers (all-MiniLM-L6-v2) → FAISS Index
        ↓
   [Retrieval] FAISS semantic search → Top 200 candidates
        ↓
   [Behavioral Scoring] 7 signals (active date, open to work, notice period, etc.)
        ↓
   [Groq LLM] llama-3.3-70b-versatile → Score + Reasoning per candidate
        ↓
   [Hybrid Fusion] 50% LLM + 30% Vector + 20% Behavioral
        ↓
   [Output] Top 100 ranked candidates → submission.csv + Streamlit UI
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq — llama-3.3-70b-versatile |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | FAISS IndexFlatL2 |
| PDF Parser | PyMuPDF (fitz) |
| UI | Streamlit |
| Language | Python 3.12 |
| Version Control | GitHub |

---

## 📁 Project Structure

```
smarthire-ai/
├── app.py                    # Streamlit web UI
├── generate_submission.py    # Generates ranked submission CSV
├── submission.csv            # Top 100 ranked candidates output
├── src/
│   ├── parser.py             # PDF/text resume parser
│   ├── embeddings.py         # FAISS vector engine
│   ├── ranker.py             # Groq LLM ranking engine
│   └── rag_pipeline.py       # End-to-end RAG pipeline
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## ⚡ Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/harishgandreddi/smarthire-ai-.git
cd smarthire-ai-

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
echo "GROQ_API_KEY=your_groq_api_key_here" > .env

# 4. Run the web app
streamlit run app.py
```

---

## 📊 Generate Submission CSV

```bash
# Place candidates.jsonl in data/ folder
mkdir data
# copy candidates.jsonl to data/

# Run the submission generator
python generate_submission.py
# Output: submission.csv with top 100 ranked candidates
```

---

## 🐳 Run with Docker

```bash
docker-compose up --build
# Open: http://localhost:8501
```

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Dataset processed | 487MB / 50,000+ candidates |
| Final ranked output | 100 candidates |
| Top candidate score | 0.8549 |
| Scoring dimensions | 3 (LLM + Vector + Behavioral) |

---

## 🔍 How Scoring Works

Every candidate receives:
- **match_score** (0-100) based on strict rubric
- **strengths** — specific matching skills/experience
- **gaps** — important missing areas
- **missing_required_skills** — exact JD skills not found
- **summary** — 2-sentence recruiter evaluation
- **recommendation** — Strongly Recommended / Recommended / Maybe / Not Recommended

Hybrid formula: `Final Score = 50% LLM Score + 30% Vector Similarity + 20% Behavioral Score`

---

## 👨‍💻 Built By
**Gandreddi Harish** | B.Tech ECE, ANITS Visakhapatnam (2026)
- 📧 harishgandreddi18@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/harishgandreddi)
- 🐙 [GitHub](https://github.com/harishgandreddi)

---

**INDIA RUNS Hackathon 2026 | Hack2skill × Redrob | #INDIARUNS**