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
            # Query targets 2026 news and isolates the lead
            query = f"March 2026 {org_name} company AI clinical trial participants layoffs news"
            results = list(ddgs.text(query, max_results=5))
            news_text = "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except: news_text = "Search offline."

    # FORMAT ENFORCER: Prevents the LLM from sending messy text to the sidebar
    analysis_prompt = f"""
    Analyze the BUSINESS ENTITY '{org_name}' based on this news: {news_text[:1200]}
    
    1. Industry: (Fintech, MedTech, or Enterprise)
    2. Beast Number: (Extract only the raw digits for staff affected OR trial subjects)
    3. The Hole: (Identify the specific SB 24-205 requirement missing)
    
    REQUIRED OUTPUT: Industry | Number | Hole
    Example: MedTech | 21 | Substantial Modification Audit
    """
    analysis = llm.invoke(analysis_prompt).content
    return news_text, analysis

class SpecialistMath:
    @staticmethod
    def calculate(count):
        # Colorado SB 24-205: $20,000 per violation
        statutory = (count if count > 0 else 1) * 20000 
        return {"statutory": statutory, "total": round(statutory * 1.25, 2)}
