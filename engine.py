import re
from duckduckgo_search import DDGS 

def get_llm(st_secrets):
    from langchain_groq import ChatGroq
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])

def safe_int(text, fallback=1):
    """Kills the ValueError by ensuring we always have a digit."""
    digits = re.sub(r"\D", "", str(text))
    return int(digits) if digits else fallback

def scout_organization(org_name, llm):
    try:
        with DDGS() as ddgs:
            # Targets 4,000 layoffs (Feb 26) or 21 clinical subjects (Jan 28)
            query = f"March 2026 {org_name} AI layoffs headcount risk SB 24-205"
            results = list(ddgs.text(query, max_results=5))
            news_text = "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except: news_text = "Search offline."

    analysis_prompt = f"""
    Sift '{org_name}' for 2026 Debt. Context: {news_text[:1200]}
    REQUIRED OUTPUT: Industry | Beast Count | Specialist Hole
    Hole must be: 'Human Appeal Path', 'FDA PCCP Gap', or 'NIST 800-171'.
    """
    return llm.invoke(analysis_prompt).content

def perform_gap_analysis(file_content, db_context, llm):
    """The Mechanic: Finds holes in uploaded technical files."""
    gap_prompt = f"""
    Perform a DEEP GAP ANALYSIS on this Technical File.
    Target Laws: SB 24-205 (Colorado), SB 25B-004 (Sunshine Act), NIST AI RMF.
    
    Technical File: {file_content[:3000]}
    
    TASKS:
    1. Identify if a 'Human Appeal Path' is documented.
    2. Check for an 'Adverse Decision Notice' template.
    3. List specific gaps for the June 30, 2026 cliff.
    """
    return llm.invoke(gap_prompt).content
