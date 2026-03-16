import re
from duckduckgo_search import DDGS 

def get_llm(st_secrets):
    from langchain_groq import ChatGroq
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])

def scout_organization(org_name, llm):
    """HUNTER: Pulls March 2026 ground truth."""
    try:
        with DDGS() as ddgs:
            # Targets 4,000 layoffs (Feb 26) or 21 clinical subjects (Jan 28)
            query = f"March 2026 {org_name} AI headcount reduction SB 24-205"
            results = list(ddgs.text(query, max_results=5))
            news_text = "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except: news_text = "Search offline."

    analysis_prompt = f"""
    Sift '{org_name}' for 2026 Statutory Debt. 
    Context: {news_text[:1500]}
    
    REQUIRED OUTPUT: Industry | Beast Count | Specialist Hole
    - If Block: Use 4000 | Human Appeal Path
    - If Neuralink: Use 21 | FDA PCCP Gap
    """
    return llm.invoke(analysis_prompt).content

def perform_remediation(org_name, hole, count, llm):
    """MECHANIC: Builds the actual legal artifacts from your database."""
    remediation_prompt = f"""
    Draft a REMEDIATION AUDIT for {org_name}.
    Target: {count} AI-driven decisions (Statutory Risk: ${count * 20000:,}).
    Hole: {hole}.
    Artifacts to Draft: 
    1. Adverse Decision Notice (Principal Reasons Disclosure)
    2. Human-in-the-Loop Appeal Protocol
    Deadline: June 30, 2026 (SB 25B-004).
    """
    return llm.invoke(remediation_prompt).content
