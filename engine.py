import re
from duckduckgo_search import DDGS 

def get_llm(st_secrets):
    from langchain_groq import ChatGroq
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])

def scout_organization(org_name, llm):
    try:
        with DDGS() as ddgs:
            # Targets the actual 4,000 layoffs confirmed Feb 2026
            query = f"March 2026 {org_name} AI layoffs headcount risk"
            results = list(ddgs.text(query, max_results=5))
            news_text = "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except: news_text = "Search offline."

    # FORMAT ENFORCER: Rips the 'Beast' count out for the sidebar
    analysis_prompt = f"""
    Analyze '{org_name}' for 2026 Statutory Debt. Context: {news_text[:1200]}
    
    REQUIRED OUTPUT: Industry | Number | Hole
    - Number: Use the raw digit for layoffs (e.g., 4000). 
    - Hole: Identify the SB 24-205 requirement (e.g., 'Human Appeal Path').
    """
    analysis = llm.invoke(analysis_prompt).content
    return news_text, analysis

class SpecialistMath:
    @staticmethod
    def calculate(count):
        # Penalty: $20,000 per violation (per person)
        statutory = count * 20000 
        return {"statutory": statutory, "total": round(statutory * 1.25, 2)}
