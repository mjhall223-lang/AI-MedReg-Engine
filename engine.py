import re
from fpdf import FPDF
from duckduckgo_search import DDGS 

def get_llm(st_secrets):
    from langchain_groq import ChatGroq
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])

def scout_organization(org_name, llm):
    """Sifts live March 2026 news for 'Beast' triggers (layoffs/trials)."""
    try:
        with DDGS() as ddgs:
            # Query hunts for the actual 2026 triggers
            query = f"March 2026 {org_name} AI automation layoffs clinical trial participants"
            results = list(ddgs.text(query, max_results=5))
            news_text = "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except: news_text = "Search offline. Using 2026 cached triggers."

    # INDUSTRY CLASSIFIER: Tailors the sifting to the specific hole
    analysis_prompt = f"""
    Analyze {org_name} and this news: {news_text[:1200]}
    Determine:
    1. Industry: (Fintech, MedTech, or Enterprise)
    2. Beast Number: (Extract only raw digits for layoffs OR trial participants)
    3. The Hole: (Which specific requirement of SB 24-205/25B-004 are they missing?)
    Return ONLY: Industry | Number | Hole
    """
    analysis = llm.invoke(analysis_prompt).content
    return news_text, analysis

class SpecialistMath:
    @staticmethod
    def calculate(count):
        # Colorado SB 24-205: $20,000 per violation
        # Each 'Beast' subject is a potential $20k fine.
        statutory = (count if count > 0 else 10) * 20000 
        return {"statutory": statutory, "total": round(statutory * 1.25, 2)}
