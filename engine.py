import re
from duckduckgo_search import DDGS 

def get_llm(st_secrets):
    from langchain_groq import ChatGroq
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])

def scout_organization(org_name, llm):
    """HUNTER: Pulls March 2026 ground truth for pitches."""
    try:
        with DDGS() as ddgs:
            # Targets 4,000 layoffs (Feb 26) or 21 clinical subjects (Jan 28)
            query = f"March 2026 {org_name} AI layoffs headcount reduction SB 24-205"
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

def perform_gap_analysis(file_content, llm):
    """MECHANIC: Compares technical files against the law database."""
    gap_prompt = f"""
    Perform a DEEP GAP ANALYSIS. 
    Compare the following CONTENT against SB 24-205 and NIST AI RMF.
    Content: {file_content[:3000]}
    
    REQUIRED: Identify missing Affirmative Defense artifacts:
    1. Adverse Decision Notice (Principal Reasons Disclosure)
    2. Human-in-the-Loop Appeal Protocol
    3. Annual Impact Assessment (June 30, 2026 Mandate)
    """
    return llm.invoke(gap_prompt).content
