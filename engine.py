import re
from duckduckgo_search import DDGS 

def get_llm(st_secrets):
    from langchain_groq import ChatGroq
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])

def scout_organization(org_name, llm):
    """HUNTER: Finds the 2026 Beast Triggers."""
    try:
        with DDGS() as ddgs:
            query = f"March 2026 {org_name} AI layoffs clinical trials compliance"
            results = list(ddgs.text(query, max_results=5))
            news_text = "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except: news_text = "Search offline."

    analysis_prompt = f"""
    Sift '{org_name}' for 2026 Debt. Context: {news_text[:1200]}
    Format: Industry | Beast Count | Specific Hole
    Hole must be: 'Human Appeal Path' (Block), 'FDA PCCP' (Neuralink), or 'NIST 800-171'.
    """
    return llm.invoke(analysis_prompt).content

def perform_remediation_audit(org_name, hole, llm):
    """MECHANIC: Builds the actual remediation plan using your DB folders."""
    remediation_prompt = f"""
    Create a REMEDIATION AUDIT for {org_name}.
    Target Hole: {hole}. 
    Framework: NIST AI RMF / SB 24-205 / SB 25B-004.
    Task: Draft the 'Affirmative Defense' artifacts (Impact Assessment & Appeal Protocol).
    Deadline: June 30, 2026.
    """
    return llm.invoke(remediation_prompt).content
