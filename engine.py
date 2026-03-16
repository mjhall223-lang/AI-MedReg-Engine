import re
from fpdf import FPDF
from duckduckgo_search import DDGS 

def get_llm(st_secrets):
    from langchain_groq import ChatGroq
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])

def scout_organization(org_name, llm):
    """Sifts live 2026 data and classifies the risk type."""
    try:
        with DDGS() as ddgs:
            # Sifts for the actual 2026 triggers
            query = f"March 2026 {org_name} AI automation layoffs clinical trial participants"
            results = list(ddgs.text(query, max_results=5))
            news_text = "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except:
        news_text = "Search offline."

    # INDUSTRY CLASSIFIER: This is how the tool 'tailors' itself
    classification_prompt = f"""
    Analyze {org_name} and this news: {news_text[:1000]}
    Determine:
    1. Industry: (Fintech, MedTech, or Enterprise)
    2. Beast Number: (Number of laid-off staff OR clinical trial participants)
    3. Legal Trigger: (e.g., 'Consequential Decision' or 'Substantial Modification')
    Return as: Industry | Number | Trigger
    """
    analysis = llm.invoke(classification_prompt).content
    return news_text, analysis

class SpecialistMath:
    @staticmethod
    def calculate(org_type, count):
        # Colorado SB 24-205: $20,000 per violation
        statutory = (count if count > 0 else 10) * 20000 
        return {"statutory": statutory, "total": round(statutory * 1.25, 2)}

def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=11)
    clean = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=clean)
    return bytes(pdf.output())
