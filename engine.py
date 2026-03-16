import re
from fpdf import FPDF
from duckduckgo_search import DDGS 

def get_llm(st_secrets):
    from langchain_groq import ChatGroq
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])

def extract_headcount(text, llm):
    """SIFTER: Finds the actual 'Beast' number (e.g., 4000 for Block) in raw news."""
    prompt = f"Identify the specific number of people affected by AI automation or clinical trials in this text: {text[:2500]}. Output ONLY the digits."
    response = llm.invoke(prompt).content
    number = re.sub(r"\D", "", response)
    # Block 2026 is 4000+, Synchron is 50. This handles those specifically.
    return int(number) if (number and 0 < len(number) < 8) else 10

def find_live_news(company_name):
    """2026 Web Sifter: Scrapes for specific March 2026 triggers."""
    try:
        with DDGS() as ddgs:
            # Sifts for the actual 2026 "Beast" news
            query = f"March 2026 {company_name} AI automation layoffs clinical trial participants"
            results = list(ddgs.text(query, max_results=5))
            return "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except Exception: return "Search offline."

class EconomicImpact:
    @staticmethod
    def calculate_liability(headcount=0):
        # Colorado SB 24-205: $20,000 per violation
        statutory = headcount * 20000 
        # Total debt including legal/audit/governance (25% overhead)
        return {"statutory": statutory, "total": round(statutory * 1.25, 2)}

def create_pdf_bytes(text):
    """Returns raw bytes to prevent Streamlit download crashes."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=11)
    clean = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=clean)
    return bytes(pdf.output())
