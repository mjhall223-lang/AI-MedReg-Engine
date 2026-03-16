import re
from fpdf import FPDF
from duckduckgo_search import DDGS 

def get_llm(st_secrets):
    from langchain_groq import ChatGroq
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])

def find_live_news(company_name):
    """SIFTER: Scrapes for the March 2026 'Beast' data points."""
    try:
        with DDGS() as ddgs:
            # Query for actual 2026 triggers: Block (4k layoffs) or Synchron (50 trials)
            query = f"March 2026 {company_name} AI automation layoffs clinical trial"
            results = list(ddgs.text(query, max_results=5))
            return "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except Exception: return "Sifter offline."

def extract_headcount(text, llm):
    """THE LOGIC: Pulls the specific number (e.g., 4000 for Block) from raw news."""
    prompt = f"Identify the specific number of people affected by AI automation or clinical trials in this text: {text[:2500]}. Output ONLY the digits."
    response = llm.invoke(prompt).content
    number = re.sub(r"\D", "", response)
    # 2026 Ground Truth: Block is 4000, Synchron is 50.
    return int(number) if (number and 0 < len(number) < 8) else 10

class EconomicImpact:
    @staticmethod
    def calculate_liability(headcount=0):
        # Colorado SB 24-205 Statutory Rate: $20,000 per violation
        statutory = headcount * 20000 
        return {"statutory": statutory, "total": round(statutory * 1.25, 2)}

def create_pdf(text):
    """Returns raw bytes to prevent Streamlit download crashes."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=11)
    # Sanitizing for 2026 Byte-handling
    clean = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=clean)
    return bytes(pdf.output())
