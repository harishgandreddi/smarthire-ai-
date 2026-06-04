import os
import json
import re
from groq import Groq

def rank_candidate(client, job_description, candidate):
    prompt = f"""You are an expert AI recruiter. Evaluate the candidate resume against the job description.

JOB DESCRIPTION:
{job_description}

CANDIDATE RESUME:
{candidate['raw_text'][:3000]}

Return ONLY a JSON object, no markdown, no extra text:
{{"match_score": 85, "strengths": ["strength1","strength2","strength3"], "gaps": ["gap1","gap2"], "summary": "2 sentence summary", "recommendation": "Strongly Recommended"}}"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.choices[0].message.content.strip()
        text = re.sub(r"```json|```", "", text).strip()
        result = json.loads(text)
        result["name"] = candidate.get("name", "Unknown")
        result["email"] = candidate.get("email", "N/A")
        result["phone"] = candidate.get("phone", "N/A")
        result["vector_score"] = candidate.get("vector_score", 0)
        return result
    except Exception as e:
        return {"name": candidate.get("name", "Unknown"), "email": candidate.get("email", "N/A"),
                "phone": candidate.get("phone", "N/A"), "match_score": 0, "strengths": [],
                "gaps": [str(e)], "summary": str(e), "recommendation": "Not Recommended", "vector_score": 0}

def rank_all_candidates(job_description, candidates):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    ranked = [rank_candidate(client, job_description, c) for c in candidates]
    ranked.sort(key=lambda x: x["match_score"], reverse=True)
    return ranked