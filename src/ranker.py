import os
import json
import re
from groq import Groq


def rank_candidate(client, job_description, candidate):
    prompt = f"""
You are a SENIOR TECHNICAL RECRUITER.

Your task is to STRICTLY evaluate the candidate against the job description.

IMPORTANT RULES:
1. Do NOT assume skills.
2. Only consider skills explicitly mentioned in the resume.
3. Missing required skills must heavily reduce the score.
4. Do NOT give scores above 80 unless the candidate matches most requirements.
5. Do NOT give scores above 90 unless the candidate is an excellent fit.
6. Freshers should not receive high scores for senior roles.
7. Be conservative and realistic.

SCORING RUBRIC:
90-100: Excellent match, nearly all required skills present, relevant experience
70-89: Good match, most required skills present
50-69: Partial match, several important skills missing
30-49: Weak match, few relevant skills
0-29: Poor match, missing most requirements

JOB DESCRIPTION:
{job_description}

CANDIDATE RESUME:
{candidate['raw_text'][:4000]}

Return ONLY valid JSON:
{{
    "match_score": <integer 0-100>,
    "strengths": ["strength1","strength2","strength3"],
    "gaps": ["gap1","gap2","gap3"],
    "missing_required_skills": ["skill1","skill2"],
    "summary": "Short honest evaluation summary",
    "recommendation": "<Strongly Recommended|Recommended|Maybe|Not Recommended>"
}}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict ATS and technical recruiter. Be conservative when scoring."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1
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
        return {
            "name": candidate.get("name", "Unknown"),
            "email": candidate.get("email", "N/A"),
            "phone": candidate.get("phone", "N/A"),
            "match_score": 0,
            "strengths": [],
            "gaps": [str(e)],
            "missing_required_skills": [],
            "summary": str(e),
            "recommendation": "Not Recommended",
            "vector_score": 0
        }


def rank_all_candidates(job_description, candidates):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    ranked = []
    for candidate in candidates:
        result = rank_candidate(client, job_description, candidate)
        vector_score = float(candidate.get("vector_score", 0))
        final_score = (0.6 * result["match_score"]) + (0.4 * vector_score)
        result["final_score"] = round(final_score, 2)
        ranked.append(result)
    ranked.sort(key=lambda x: x["match_score"], reverse=True)
    return ranked