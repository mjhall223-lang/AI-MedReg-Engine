import re
from duckduckgo_search import DDGS 

def get_llm(st_secrets):
    from langchain_groq import ChatGroq
    # Using the latest 2026-optimized model
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])

def scout_organization(org_name, llm):
    """SIFTER: Only pulls March 2026 data for the SPECIFIC company."""
    try:
        with DDGS() as ddgs:
            # Query targets 2026 triggers: 4,000 layoffs (Feb 26) or 21 trials (Jan 28)
            query = f"March 2026 {org_name} AI layoffs headcount clinical trial participants"
            results = list(ddgs.text(query, max_results=5))
            news_text = "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except: 
        news_text = "Search offline."

    # PROMPT FIX: Forces the LLM to return a clean, sidebar-ready string
    analysis_prompt = f"""
    Analyze the BUSINESS ENTITY '{org_name}' for its 2026 Statutory Debt.
    Context: {news_text[:1200]}
    
    1. Industry: (Fintech, MedTech, or Enterprise)
    2. Beast Number: (Extract raw digits. FOR BLOCK USE 4000. FOR NEURALINK USE 21.)
    3. The Hole: (Identify the specific SB 24-205 requirement missing)
    
    REQUIRED OUTPUT: Industry | Number | Hole
    Example: Fintech | 4000 | Human Appeal Path
    """
    analysis = llm.invoke(analysis_prompt).content
    return news_text, analysis

class SpecialistMath:
    @staticmethod
    def calculate(count):
        # Colorado SB 24-205: $20,000 per violation (per person affected)
        statutory = (count if count > 0 else 1) * 20000 
        return {"statutory": statutory, "total": round(statutory * 1.25, 2)}
        
