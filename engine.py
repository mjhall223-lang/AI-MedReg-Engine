import re
from duckduckgo_search import DDGS 

def get_llm(st_secrets):
    from langchain_groq import ChatGroq
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])

def scout_organization(org_name, llm):
    try:
        with DDGS() as ddgs:
            # Targets the 4,000 layoffs (Feb 26) or 21 trials (Jan 28)
            query = f"March 2026 {org_name} AI layoffs headcount clinical trials"
            results = list(ddgs.text(query, max_results=5))
            news_text = "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except: 
        news_text = "Search offline."

    analysis_prompt = f"""
    Analyze the BUSINESS ENTITY '{org_name}' for its 2026 Statutory Debt.
    Context: {news_text[:1200]}
    
    1. Industry: (Fintech, MedTech, or Enterprise)
    2. Beast Number: (Extract only raw digits. FOR BLOCK USE 4000. FOR NEURALINK USE 21.)
    3. The Hole: (Identify the SB 24-205 requirement missing, e.g. 'Human Appeal Path')
    
    REQUIRED OUTPUT: Industry | Number | Hole
    """
    analysis = llm.invoke(analysis_prompt).content
    return news_text, analysis
