import re
from fpdf import FPDF
from duckduckgo_search import DDGS 

def get_llm(st_secrets):
    from langchain_groq import ChatGroq
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])

def scout_organization(org_name, llm):
    """Sifts live data for 2026 'Beast' triggers and classifies the 'Hole'."""
    try:
        with DDGS() as ddgs:
            # Query hunts for the actual 2026 triggers
            query = f"March 2026 {org_name} AI automation layoffs clinical trial human subjects"
            results = list(ddgs.text(query, max_results=5))
            news_text = "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except: news_text = "Search offline."

    # INDUSTRY CLASSIFIER: Tailors the pitch to the specific hole
    analysis_prompt = f"""
    Analyze {org_name} and this news: {news_text[:1200]}
    1. Industry: (Fintech, MedTech, or Enterprise)
    2. Beast Number: (Number of staff affected OR clinical trial participants)
    3. The Hole: (Which specific requirement of SB 24-205 are they missing?)
    Return ONLY in this format: Industry | Number | Hole
    """
    analysis = llm.invoke(analysis_prompt).content
    return news_text, analysis

class SpecialistMath:
    @staticmethod
    def calculate(count):
        # Colorado SB 24-205: $20,000 per violation
        statutory = (count if count > 0 else 10) * 20000 
        return {"statutory": statutory, "total": round(statutory * 1.25, 2)}
