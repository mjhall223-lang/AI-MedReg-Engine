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
    """ULTRA-RECURSIVE: Handles nested folders like Regulations/Regulations/Federal."""
    path_root = Path(base_dir)
    if not path_root.exists(): return []
    return sorted([str(f.relative_to(path_root.parent)) for f in path_root.rglob('*.pdf')])

def extract_pdf_text(uploaded_file):
    reader = PdfReader(io.BytesIO(uploaded_file.read()))
    return "".join([p.extract_text() for p in reader.pages if p.extract_text()])

def smart_web_sifter(org_name):
    """SMART SEARCH: Targets 2026 Neurotech/BCI governance specifically."""
    try:
        with DDGS() as ddgs:
            # Query updated for Synchron's 2026 Chiral model & neural data laws
            q = f"{org_name} Chiral AI governance ethics clinical data privacy 2026"
            results = list(ddgs.text(q, max_results=3))
            
            if not results:
                # Fallback for stealth-mode BCIs
                results = list(ddgs.text(f"{org_name} neural data privacy", max_results=1))
            
            if not results: return "Error: No public results. Entity may be in clinical stealth mode."
            
            url = results[0]['href']
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            res = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            
            # Strip junk (nav, footer, ads) to avoid context pollution
            for junk in soup(["nav", "footer", "script", "style"]): junk.extract()
            return f"SOURCE: {url}\n\n" + soup.get_text(separator=' ', strip=True)
    except Exception as e:
        return f"Web Sifter Error: {str(e)}"

def perform_gap_analysis(content, laws, org, llm):
    # Enforcement date updated to June 30, 2026 per Colorado SB 25B-004
    prompt = f"Audit {org} against {laws}. Today: March 16, 2026. Cliff: June 30, 2026. Find the 'Holes'. Content: {content[:4000]}"
    return llm.invoke(prompt).content

def generate_pdf_report(results, org_name, hole_type, laws):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"REMEDIATION AUDIT: {org_name}", styles['Title']), Spacer(1,12)]
    
    # Financial Risk Logic
    penalty = "$20,000 per violation" if "Human" in hole_type else "$10,000 per violation"
    story.append(Paragraph("FINANCIAL EXPOSURE SUMMARY", styles['Heading2']))
    story.append(Paragraph(f"Identified Statutory Debt: {penalty}", styles['Normal']))
    story.append(Paragraph(f"Audited Against: {', '.join(laws)}", styles['Italic']))
    story.append(Spacer(1,12))

    for line in results.split('\n'):
        if line.strip(): story.append(Paragraph(line, styles['Normal']))
    doc.build(story)
    buffer.seek(0)
    return buffer
