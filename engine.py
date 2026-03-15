import os
import re
from fpdf import FPDF
from duckduckgo_search import DDGS 

def extract_headcount(text, llm):
    """Dynamic Sifter: Finds the 'Magic Number' (headcount) in raw news."""
    prompt = f"Identify the specific number of people affected (layoffs/participants) in this text: {text[:2500]}. Output ONLY the digits."
    response = llm.invoke(prompt).content
    number = re.sub(r"\D", "", response)
    return int(number) if (number and 0 < len(number) < 8) else 10

def find_and_scrape_live_news(company_name):
    """Web Sifter: Scrapes 2026 headlines for liability triggers."""
    try:
        with DDGS() as ddgs:
            # Sifts for specific 2026 keywords
            query = f"March 2026 {company_name} AI automation layoffs clinical trial"
            results = list(ddgs.text(query, max_results=5))
            return "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except: return "Search offline."

def create_pdf(text):
    """Generates PDF as bytes for stable Streamlit downloading."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=11)
    clean = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=clean)
    return bytes(pdf.output())
