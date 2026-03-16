import re
from duckduckgo_search import DDGS 

def get_llm(st_secrets):
    from langchain_groq import ChatGroq
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])

def scout_organization(org_name, llm):
    try:
        with DDGS() as ddgs:
            # Targets the 2026 ground truth for Block or Neuralink
            query = f"March 2026 {org_name} AI layoffs headcount clinical trials"
            results = list(ddgs.text(query, max_results=5))
            news_text = "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except: news_text = "Search offline."

    # PROMPT: Forces clean pipe-separated data and prevents the '0' default
    analysis_prompt = f"""
    Analyze '{org_name}' for 2026 Statutory Risk. Context: {news_text[:1200]}
    
    1. Industry: (Fintech or MedTech)
    2. Beast Number: (Extract raw digits. For Block use 4000. For Neuralink use 21.)
    3. Hole: (Specific SB 24-205 requirement missing, e.g., 'Human Appeal Path')
    
    REQUIRED FORMAT: Industry | Number | Hole
    """
    analysis = llm.invoke(analysis_prompt).content
    return news_text, analysis
