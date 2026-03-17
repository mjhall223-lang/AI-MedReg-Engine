import io, requests
from pathlib import Path
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain_groq import ChatGroq
from pypdf import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime

def get_llm(st_secrets):
    """Initializes the Llama-3.3-70B model via Groq."""
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])

def list_all_laws(base_dir="Regulations"):
    """Scans the local directory for regulatory PDF files."""
    path_root = Path(base_dir)
    if not path_root.exists():
        return []
    return sorted([str(f.relative_to(path_root.parent)) for f in path_root.rglob('*.pdf')])

def extract_pdf_text(uploaded_file):
    """Extracts raw text from an uploaded PDF file."""
    reader = PdfReader(io.BytesIO(uploaded_file.read()))
    return "".join([p.extract_text() for p in reader.pages if p.extract_text()])

def smart_web_sifter(org_name):
    """Scrapes the web for 2026-specific clinical and governance data."""
    try:
        with DDGS() as ddgs:
            q = f"{org_name} Chiral AI cognitive governance clinical ethics 2026"
            results = list(ddgs.text(q, max_results=3))
            if not results: return "Error: No public results found."
            
            url = results[0]['href']
            res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            for junk in soup(["nav", "footer", "script", "style"]): junk.extract()
            return f"SOURCE: {url}\n\n" + soup.get_text(separator=' ', strip=True)
    except Exception as e:
        return f"Web Sifter Error: {str(e)}"

def perform_gap_analysis(content, laws, org, llm):
    """Main audit function comparing ingested content against 2026 statutes."""
    today = datetime.now().strftime("%B %d, %Y")
    system_prompt = f"""
    You are a Lead Compliance Strategist. Today: {today}. Statutory Cliff: June 30, 2026.
    Audit the provided content for {org} against these frameworks: {laws}. 
    
    STRICT AUDIT PARAMETERS:
    1. Identify 'Intention-to-Action' (Cognitive AI) technologies.
    2. Colorado SB 24-205: Flag if there is no documented 'Human Appeal Path' for AI decisions.
    3. NIST 800-171 Rev 3: Check for 'Supply Chain Risk Management' (SCRM) regarding hardware partners.
    4. CITE PENALTY: Flag a $20,000 per-violation risk for non-compliance.
    """
    response = llm.invoke([("system", system_prompt), ("human", content[:7000])])
    return response.content

def ask_regulatory_chat(prompt, audit_context, llm):
    """Powers the Q&A feature using the audit results as background context."""
    system_msg = f"""
    You are a 2026 Regulatory Expert. 
    Use the following Audit Results to answer the user's question. 
    If the answer isn't in the context, use your knowledge of 2026 laws.
    
    CONTEXT:
    {audit_context[:5000]}
    """
    response = llm.invoke([("system", system_msg), ("human", prompt)])
    return response.content

def generate_pdf_report(results, org_name, laws):
    """Generates a professional PDF remediation report."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"STATUTORY REMEDIATION: {org_name}", styles['Title']), Spacer(1,12)]
    story.append(Paragraph(f"Audited Frameworks: {', '.join(laws)}", styles['Italic']))
    story.append(Spacer(1,12))
    for line in results.split('\n'):
        if line.strip(): story.append(Paragraph(line, styles['Normal']))
    doc.build(story)
    buffer.seek(0)
    return buffer
