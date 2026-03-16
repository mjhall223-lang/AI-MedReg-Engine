import re
import os
from fpdf import FPDF
from duckduckgo_search import DDGS 

def get_llm(st_secrets):
    from langchain_groq import ChatGroq
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])

def find_live_news(company_name):
    """SIFTER: Scrapes for the March 2026 'Beast' data points."""
    try:
        # 2026 DDGS syntax requires the context manager
        with DDGS() as ddgs:
            query = f"March 2026 {company_name} AI automation layoffs clinical trial"
            results = list(ddgs.text(query, max_results=5))
            if not results: return "No recent triggers found for this company."
            return "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Sifter connection error: {str(e)}"

def extract_headcount(text, llm):
    """The Logic: Pulls the specific number (e.g., 4237 for Block) from raw news."""
    if "Sifter connection error" in text: return 10
    prompt = f"Identify the specific number of people affected by AI automation or clinical trials in this text: {text[:2500]}. Output ONLY the digits."
    response = llm.invoke(prompt).content
    number = re.sub(r"\D", "", response)
    # Block is ~4000, Synchron is ~50.
    return int(number) if (number and 0 < len(number) < 8) else 10

class LiabilityEngine:
    @staticmethod
    def run_math(headcount):
        # Colorado SB 24-205 Statutory Rate: $20,000 per violation
        statutory = headcount * 20000 
        return {"statutory": statutory, "total": round(statutory * 1.25, 2)}

def create_pdf_bytes(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=11)
    clean = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=clean)
    return bytes(pdf.output())
