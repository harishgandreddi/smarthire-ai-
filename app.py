import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv
from src.rag_pipeline import SmartHirePipeline

load_dotenv()

st.set_page_config(page_title="SmartHire AI", page_icon="🤖", layout="wide")

st.title("🤖 SmartHire AI")
st.markdown("##### Intelligent Candidate Discovery & Ranking powered by Groq + RAG")
st.divider()

if "pipeline" not in st.session_state:
    st.session_state.pipeline = SmartHirePipeline()
if "results" not in st.session_state:
    st.session_state.results = []

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    top_k = st.slider("Top candidates to retrieve", 3, 20, 10)
    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Upload resumes (PDF)\n2. Paste job description\n3. AI ranks candidates\n4. Download results")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📄 Job Description")
    job_desc = st.text_area("Paste the job description here", height=300,
        placeholder="e.g. We are looking for a Python developer with 3+ years of experience in ML, NLP...")

with col2:
    st.subheader("📁 Upload Resumes")
    uploaded_files = st.file_uploader("Upload candidate resumes (PDF)", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} resume(s) uploaded")

st.divider()

if st.button("🚀 Analyze & Rank Candidates", type="primary", use_container_width=True):
    if not job_desc.strip():
        st.error("Please enter a job description.")
    elif not uploaded_files:
        st.error("Please upload at least one resume.")
    elif not os.getenv("GROQ_API_KEY"):
        st.error("Please enter your Groq API key in the sidebar.")
    else:
        with st.spinner("🔄 Parsing resumes and building vector index..."):
            count = st.session_state.pipeline.load_resumes(uploaded_files)
            st.info(f"Indexed {count} resumes.")
        with st.spinner("🧠 Ranking candidates with Groq AI..."):
            results = st.session_state.pipeline.run(job_desc, top_k=top_k)
            st.session_state.results = results
        st.success("✅ Analysis complete!")

if st.session_state.results:
    results = st.session_state.results
    st.subheader("🏆 Ranked Candidates")

    names = [r.get("name", "Unknown")[:20] for r in results]
    scores = [r.get("match_score", 0) for r in results]
    fig = px.bar(x=scores, y=names, orientation="h", color=scores,
        color_continuous_scale="teal", labels={"x": "Match Score", "y": "Candidate"},
        title="Candidate Match Scores")
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    for i, r in enumerate(results):
        score = r.get("match_score", 0)
        color = "🟢" if score >= 75 else "🟡" if score >= 50 else "🔴"
        rec = r.get("recommendation", "N/A")
        with st.expander(f"{color} #{i+1} — {r.get('name','Unknown')} | Score: {score}/100 | {rec}"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Match Score", f"{score}/100")
            c2.metric("Recommendation", rec)
            c3.metric("Email", r.get("email", "N/A"))
            st.markdown(f"**Summary:** {r.get('summary', '')}")
            col_s, col_g = st.columns(2)
            with col_s:
                st.markdown("**✅ Strengths:**")
                for s in r.get("strengths", []):
                    st.markdown(f"- {s}")
            with col_g:
                st.markdown("**⚠️ Gaps:**")
                for g in r.get("gaps", []):
                    st.markdown(f"- {g}")

    st.divider()
    df = pd.DataFrame([{"Rank": i+1, "Name": r.get("name"), "Email": r.get("email"),
        "Score": r.get("match_score"), "Recommendation": r.get("recommendation"),
        "Summary": r.get("summary")} for i, r in enumerate(results)])
    st.download_button("⬇️ Download Results as CSV", df.to_csv(index=False),
        "smarthire_results.csv", "text/csv", use_container_width=True)