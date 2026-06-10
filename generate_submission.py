"""
SmartHire AI — Submission Generator
INDIA RUNS Hackathon | Track 01: Data & AI Challenge
Generates top 100 ranked candidates for the given Job Description.
"""

import json
import os
import csv
import re
import time
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────
CANDIDATES_FILE = "data/candidates.jsonl"
OUTPUT_FILE = "submission.csv"
TOP_K_RETRIEVE = 200   # Retrieve top 200 via vector search
FINAL_TOP_N = 100      # Re-rank to final top 100 using LLM
MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

JOB_DESCRIPTION = """
Senior AI Engineer — Founding Team | Redrob AI (Series A)
Location: Pune/Noida, India | Experience: 5-9 years

MUST HAVE:
- Production experience with embeddings-based retrieval systems (sentence-transformers, BGE, E5)
- Production experience with vector databases (FAISS, Pinecone, Weaviate, Qdrant, Milvus)
- Strong Python skills
- Experience designing evaluation frameworks for ranking systems (NDCG, MRR, MAP)
- Applied ML/AI at product companies (NOT pure services/consulting)
- Shipped end-to-end ranking, search, or recommendation systems to real users

NICE TO HAVE:
- LLM fine-tuning (LoRA, QLoRA, PEFT)
- Learning-to-rank models
- HR-tech or marketplace product experience
- Open-source AI/ML contributions

DISQUALIFIERS:
- Pure research background without production deployment
- Only consulting firm experience (TCS, Infosys, Wipro, Accenture, Cognizant)
- Primary expertise in CV/speech/robotics without NLP/IR
- Title-chaser (switching every 1.5 years for titles)

BEHAVIORAL SIGNALS MATTER:
- Active on platform (recently logged in)
- High recruiter response rate
- Open to work
- Willing to relocate to Pune/Noida
- Notice period under 30 days preferred
"""

# ── Step 1: Load candidates ──────────────────────────────────────────────────
def load_candidates(filepath):
    print(f"Loading candidates from {filepath}...")
    candidates = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates)} candidates.")
    return candidates

# ── Step 2: Convert candidate to text for embedding ──────────────────────────
def candidate_to_text(c):
    profile = c.get("profile", {})
    skills = c.get("skills", [])
    career = c.get("career_history", [])
    education = c.get("education", [])
    signals = c.get("redrob_signals", {})

    skill_names = [s["name"] for s in skills]
    career_text = " | ".join([
        f"{x.get('title','')} at {x.get('company','')} ({x.get('duration_months',0)//12}yr)"
        for x in career[:4]
    ])
    edu_text = " | ".join([
        f"{x.get('degree','')} {x.get('field_of_study','')} from {x.get('institution','')}"
        for x in education[:2]
    ])

    text = f"""
    {profile.get('headline', '')}
    {profile.get('summary', '')[:500]}
    Skills: {', '.join(skill_names)}
    Experience: {profile.get('years_of_experience', 0)} years
    Current: {profile.get('current_title', '')} at {profile.get('current_company', '')}
    Industry: {profile.get('current_industry', '')}
    Career: {career_text}
    Education: {edu_text}
    Location: {profile.get('location', '')} {profile.get('country', '')}
    """
    return text.strip()

# ── Step 3: Build FAISS index ─────────────────────────────────────────────────
def build_faiss_index(candidates, model):
    print("Building FAISS vector index...")
    texts = [candidate_to_text(c) for c in candidates]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
    embeddings = embeddings.astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    print(f"Index built with {index.ntotal} vectors.")
    return index, embeddings

# ── Step 4: Compute behavioral score ─────────────────────────────────────────
def behavioral_score(c):
    signals = c.get("redrob_signals", {})
    score = 0.0

    # Active recently (last 30 days)
    last_active = signals.get("last_active_date", "")
    if "2026" in last_active:
        score += 0.20

    # Open to work
    if signals.get("open_to_work_flag", False):
        score += 0.15

    # High recruiter response rate
    rr = signals.get("recruiter_response_rate", 0)
    score += rr * 0.20

    # Low notice period
    notice = signals.get("notice_period_days", 90)
    if notice <= 30:
        score += 0.15
    elif notice <= 60:
        score += 0.08

    # Willing to relocate
    if signals.get("willing_to_relocate", False):
        score += 0.10

    # GitHub activity
    github = signals.get("github_activity_score", 0)
    score += min(github / 100, 0.10)

    # Profile completeness
    completeness = signals.get("profile_completeness_score", 0)
    score += min(completeness / 100, 0.10)

    return round(score, 4)

# ── Step 5: LLM re-ranking ────────────────────────────────────────────────────
def llm_rank_candidate(client, candidate_text, candidate_id):
    prompt = f"""You are an expert AI recruiter at Redrob AI. 
Rate this candidate for the Senior AI Engineer role (5-9 years experience needed).

KEY REQUIREMENTS:
- Production ML/AI experience at product companies
- Embeddings, vector search, ranking systems
- Python, FAISS/Pinecone/similar, evaluation frameworks
- NOT from pure consulting background
- Active and available

CANDIDATE PROFILE:
{candidate_text[:1500]}

Return ONLY a JSON object:
{{"score": 0.85, "reasoning": "one sentence explanation"}}

Score rules: 0.0-1.0. Be strict. Only score >0.8 if truly excellent match."""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        text = response.choices[0].message.content.strip()
        text = re.sub(r"```json|```", "", text).strip()
        result = json.loads(text)
        return float(result.get("score", 0.0)), result.get("reasoning", "")
    except Exception as e:
        return 0.0, f"Error: {str(e)}"

# ── Main Pipeline ─────────────────────────────────────────────────────────────
def main():
    print("\n🚀 SmartHire AI — Submission Generator")
    print("=" * 50)

    # Load candidates
    candidates = load_candidates(CANDIDATES_FILE)

    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    # Build index
    index, _ = build_faiss_index(candidates, model)

    # Search for top candidates using JD
    print(f"\nSearching top {TOP_K_RETRIEVE} candidates via vector search...")
    jd_embedding = model.encode([JOB_DESCRIPTION]).astype("float32")
    distances, indices = index.search(jd_embedding, TOP_K_RETRIEVE)

    # Get retrieved candidates
    retrieved = []
    for dist, idx in zip(distances[0], indices[0]):
        c = candidates[idx]
        vector_score = float(1 / (1 + dist))
        b_score = behavioral_score(c)
        combined = (vector_score * 0.6) + (b_score * 0.4)
        retrieved.append({
            "candidate": c,
            "vector_score": vector_score,
            "behavioral_score": b_score,
            "combined_score": combined,
            "text": candidate_to_text(c)
        })

    # Sort by combined score
    retrieved.sort(key=lambda x: x["combined_score"], reverse=True)
    top_candidates = retrieved[:150]

    # LLM re-ranking on top 150
    print(f"\nRe-ranking top {len(top_candidates)} candidates with Groq LLM...")
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    final_results = []
    for i, item in enumerate(top_candidates):
        c = item["candidate"]
        cid = c["candidate_id"]
        print(f"  [{i+1}/{len(top_candidates)}] Ranking {cid}...")

        llm_score, reasoning = llm_rank_candidate(client, item["text"], cid)

        # Final score: 50% LLM + 30% vector + 20% behavioral
        final_score = (llm_score * 0.50) + (item["vector_score"] * 0.30) + (item["behavioral_score"] * 0.20)
        final_score = round(min(final_score, 0.9999), 4)

        final_results.append({
            "candidate_id": cid,
            "score": final_score,
            "reasoning": reasoning[:200] if reasoning else "Strong technical match"
        })

        # Rate limit protection
        if (i + 1) % 10 == 0:
            time.sleep(1)

    # Sort and take top 100
    final_results.sort(key=lambda x: x["score"], reverse=True)
    top_100 = final_results[:100]

    # Ensure unique scores (avoid ties breaking submission validator)
    for i, r in enumerate(top_100):
        r["rank"] = i + 1
        r["score"] = round(r["score"] - (i * 0.0001), 4)

    # Write submission CSV
    print(f"\nWriting submission to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for r in top_100:
            writer.writerow({
                "candidate_id": r["candidate_id"],
                "rank": r["rank"],
                "score": r["score"],
                "reasoning": r["reasoning"]
            })

    print(f"\n✅ Submission generated: {OUTPUT_FILE}")
    print(f"   Top candidate: {top_100[0]['candidate_id']} (score: {top_100[0]['score']})")
    print(f"   Total ranked: {len(top_100)}")

if __name__ == "__main__":
    main()
