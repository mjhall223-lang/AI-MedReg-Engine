import re
from duckduckgo_search import DDGS 

def get_llm(st_secrets):
    from langchain_groq import ChatGroq
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])

def scout_organization(org_name, llm):
    """SIFTER: Only pulls March 2026 'Beast' triggers from your DB folders."""
    try:
        with DDGS() as ddgs:
            # Targets 4,000 layoffs (Feb 26) or 21 trials (Jan 28)
            query = f"March 2026 {org_name} AI layoffs headcount clinical trials"
            results = list(ddgs.text(query, max_results=5))
            news_text = "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except: news_text = "Search offline."

    # PROMPT: Forces the AI to use your specific folder hierarchy
    analysis_prompt = f"""
    Analyze '{org_name}' for 2026 Statutory Debt using our Multi-Layer Database:
    Context: {news_text[:1200]}
    
    FOLDER TRIGGERS:
    - Regulations/Regulations/Colorado: SB 24-205 (Target: 4000 layoffs)
    - Regulations/Regulations/Federal: FDA PCCP (Target: 21 trial subjects)
    - Regulations/Regulations/Texas: Tax/CPA Compliance
    
    REQUIRED OUTPUT: Industry | Beast Number | Specialist Hole
    - Hole MUST be: 'Human Appeal Path', 'FDA PCCP Gap', or 'CMMC Level 2'
    """
    analysis = llm.invoke(analysis_prompt).content
    return news_text, analysis
