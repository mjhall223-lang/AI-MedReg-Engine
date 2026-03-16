import re
from fpdf import FPDF
from duckduckgo_search import DDGS 

def get_llm(st_secrets):
    from langchain_groq import ChatGroq
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])

def scout_organization(org_name, llm):
    """SIFTER: Only pulls 2026 data for the SPECIFIC company typed."""
    try:
        with DDGS() as ddgs:
            # SEARCH FIX: Targets 2026 news and forces business context
            query = f"March 2026 {org_name} company AI clinical trial participants layoffs news"
            results = list(ddgs.text(query, max_results=5))
            news_text = "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except: news_text = "Search offline."

    # ENTITY CLASSIFIER: Forces the AI to treat the name as a Business Entity
    analysis_prompt = f"""
    Analyze the BUSINESS ENTITY named '{org_name}' based on this news: {news_text[:1200]}
    
    1. Industry: (Fintech, MedTech, or Enterprise)
    2. Beast Number: (Extract the raw digits for staff affected OR trial subjects)
    3. The Hole: (Which SB 24-205 requirement is this specific company missing?)
    
    Return ONLY in this format: Industry | Number | Hole
    """
    analysis = llm.invoke(analysis_prompt).content
    return news_text, analysis

class SpecialistMath:
    @staticmethod
    def calculate(count):
        # Colorado SB 24-205: $20,000 per violation
        # Each subject (layoff/participant) is a statutory count.
        statutory = (count if count > 0 else 10) * 20000 
        return {"statutory": statutory, "total": round(statutory * 1.25, 2)}
