import re
from fpdf import FPDF
from duckduckgo_search import DDGS 

def get_llm(st_secrets):
    from langchain_groq import ChatGroq
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])

def find_live_news(company_name):
    """SIFTER: Scrapes for specific March 2026 AI triggers."""
    try:
        with DDGS() as ddgs:
            # Query hunts for the 4,000 headcount or 50-person trial triggers
            query = f"March 2026 {company_name} AI automation layoffs clinical trial participants"
            results = list(ddgs.text(query, max_results=5))
            return "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except Exception: return "Search offline."

def extract_headcount(text, llm):
    """THE LOGIC: Pulls the 'Beast' number (e.g. 4000) from raw 2026 news."""
    # If sifter failed, default to 10 to avoid $0
    if "offline" in text: return 10
    prompt = f"Identify the number of people affected (layoffs or trial participants) in this text: {text[:2500]}. Output ONLY digits."
    response = llm.invoke(prompt).content
    number = re.sub(r"\D", "", response)
    # Block is ~4000, Synchron is ~50. 
    return int(number) if (number and 0 < len(number) < 8) else 10

class EconomicImpact:
    @staticmethod
    def calculate_liability(headcount=0):
        # Colorado SB 24-205 Statutory Rate: $20,000 per violation
        statutory = headcount * 20000 
        return {"statutory": statutory, "total": round(statutory * 1.25, 2)}

def create_pdf_bytes(text):
    """Fixes Streamlit's 'unsupported_error' by returning raw bytes."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=11)
    # Sanitize for latin-1
    clean = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=clean)
    return bytes(pdf.output())
