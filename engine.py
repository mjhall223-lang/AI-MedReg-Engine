import io, requests
from pathlib import Path
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain_groq import ChatGroq
from pypdf import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def get_llm(st_secrets):
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])

def list_all_laws(base_dir="Regulations"):
    path_root = Path(base_dir)
    return sorted([str(f.relative_to(path_root.parent)) for f in path_root.rglob('*.pdf')])

def extract_pdf_text(uploaded_file):
    reader = PdfReader(io.BytesIO(uploaded_file.read()))
    return "".join([p.extract_text() for p in reader.pages if p.extract_text()])

def smart_web_sifter(org_name):
    """Targets 2026 Chiral™ & Cognitive AI clinical roadmaps."""
    try:
        with DDGS() as ddgs:
            # Query updated for Synchron's 2026 shift to 'Cognitive AI'
            q = f"{org_name} Chiral AI cognitive governance clinical ethics 2026"
            results = list(ddgs.text(q, max_results=3))
            
            if not results:
                return "Error: No public results. Entity may be in clinical stealth mode."
            
            url = results[0]['href']
            headers = {'User-Agent': 'Mozilla/5.0'}
            res = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            for junk in soup(["nav", "footer", "script", "style"]): junk.extract()
            return f"SOURCE: {url}\n\n" + soup.get_text(separator=' ', strip=True)
    except Exception as e:
        return f"Web Sifter Error: {str(e)}"

def perform_gap_analysis(content, laws, org, llm):
    """STRICT 2026 AUDITOR: Identifies missing BCI appeal paths."""
    system_prompt = f"""
    You are a Lead Compliance Strategist. Today: March 16, 2026. Cliff: June 30, 2026.
    Audit {org} against {laws}. 

    STRICT REQUIREMENTS:
    1. Flag if 'Intention-to-Action' (Cognitive AI) is used.
    2. Colorado SB 24-205: Identify if they lack a 'Human Appeal Path' for neural commands.
    3. NIST 800-171 Rev 3: Flag if they lack 'Supply Chain Risk Management' for partners like NVIDIA.
    4. CITE PENALTY: $20,000 per violation for missing adverse-decision notices.
    """
    
    response = llm.invoke([("system", system_prompt), ("human", content[:5000])])
    return response.content

def generate_pdf_report(results, org_name, laws):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"STATUTORY REMEDIATION: {org_name}", styles['Title']), Spacer(1,12)]
    story.append(Paragraph(f"Audited Against: {', '.join(laws)}", styles['Italic']))
    story.append(Spacer(1,12))
    for line in results.split('\n'):
        if line.strip(): story.append(Paragraph(line, styles['Normal']))
    doc.build(story)
    buffer.seek(0)
    return buffer
