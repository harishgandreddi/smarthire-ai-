# 🤖 SmartHire AI
### Intelligent Candidate Discovery & Ranking System
> Built for INDIA RUNS Hackathon — Track 01: Data & AI Challenge

---

## 🚀 What it does
SmartHire AI uses RAG (Retrieval Augmented Generation) + Google Gemini to:
- Parse resumes (PDF) and build a semantic vector index
- Match candidates to job descriptions using sentence embeddings
- Rank candidates with AI-generated scores, strengths, gaps & recommendations
- Export results as CSV

---

## 🛠️ Tech Stack
| Layer | Technology |
|-------|-----------|
| LLM | Google Gemini 1.5 Flash |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | FAISS |
| RAG Framework | LangChain |
| UI | Streamlit |
| Container | Docker |
| Cloud | AWS EC2 |

---

## ⚡ Quick Start (Local)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/smarthire-ai.git
cd smarthire-ai

# 2. Set up environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## 🐳 Run with Docker

```bash
# Build and run
docker-compose up --build

# Open browser
http://localhost:8501
```

---

## ☁️ Deploy to AWS EC2

```bash
# 1. SSH into your EC2 instance
ssh -i your-key.pem ec2-user@YOUR_EC2_IP

# 2. Install Docker
sudo yum update -y
sudo yum install -y docker
sudo service docker start

# 3. Clone and run
git clone https://github.com/YOUR_USERNAME/smarthire-ai.git
cd smarthire-ai
echo "GEMINI_API_KEY=your_key" > .env
docker-compose up -d

# 4. Open port 8501 in EC2 Security Group
# Access: http://YOUR_EC2_IP:8501
```

---

## 📁 Project Structure
```
smarthire-ai/
├── app.py                  # Streamlit UI
├── src/
│   ├── parser.py           # PDF resume parser
│   ├── embeddings.py       # FAISS vector engine
│   ├── ranker.py           # Gemini LLM ranker
│   └── rag_pipeline.py     # Full RAG pipeline
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 👨‍💻 Built by
**Gandreddi Harish** | INDIA RUNS Hackathon 2026
