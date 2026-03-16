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

def extract_pdf_text(f):
    reader = PdfReader(io.BytesIO(f.read()))
    return "".join([p.extract_text() for p in reader.pages if p.extract_text()])

def smart_web_sifter(org_name):
    """Targets 2026 Chiral™ & Cognitive AI docs with search fallback."""
    try:
        with DDGS() as ddgs:
            # Attempt 1: Specific 2026 rebrand
            q1 = f"{org_name} Chiral AI governance policy 2026"
            results = list(ddgs.text(q1, max_results=2))
            
            # Attempt 2: Broadened name
            if not results:
                q2 = f"{org_name} neural data privacy clinical ethics"
                results = list(ddgs.text(q2, max_results=2))
            
            if not results:
                return "Error: Search returned no results. Streamlit Cloud might be IP-blocked."

            url = results[0]['href']
            res = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(res.text, 'html.parser')
            for junk in soup(["nav", "footer", "script", "style"]): junk.extract()
            return f"SOURCE: {url}\n\n" + soup.get_text(separator=' ', strip=True)
    except Exception as e:
        return f"Web Sifter Error: {e}"

def perform_gap_analysis(content, laws, org, llm):
    # March 16, 2026 baseline for audit
    prompt = f"Audit {org} against {laws}. Today: March 16, 2026. Cliff: June 30, 2026. Find the 'Holes'. Content: {content[:4000]}"
    return llm.invoke(prompt).content

def generate_pdf_report(results, org_name, hole_type, laws):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"REMEDIATION AUDIT: {org_name}", styles['Title']), Spacer(1,12)]
    penalty = "$20,000 per violation" if "Human" in hole_type else "$10,000 per violation"
    story.append(Paragraph(f"Financial Risk: {penalty}", styles['Normal']))
    story.append(Paragraph(f"Audited Against: {', '.join(laws)}", styles['Italic']))
    story.append(Spacer(1,12))
    for line in results.split('\n'):
        if line.strip(): story.append(Paragraph(line, styles['Normal']))
    doc.build(story)
    buffer.seek(0)
    return buffer
