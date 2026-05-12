import io, requests, os
from pathlib import Path
from firecrawl import FirecrawlApp
from langchain_groq import ChatGroq
from pypdf import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime

# --- A-LEVEL INSTRUMENTATION ---
def _enable_instrumentation(st_secrets):
    if "LANGCHAIN_API_KEY" in st_secrets:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = st_secrets["LANGCHAIN_API_KEY"]
        os.environ["LANGCHAIN_PROJECT"] = st_secrets.get("LANGCHAIN_PROJECT", "AI-MedReg-Engine-2026")

# --- REMEDIATION TEMPLATES ---
REMEDIATION_TEMPLATES = {
    "Colorado SB 24-205": """
    <b>[REMEDIATION ATTACHMENT: Human Appeal Path Template]</b><br/><br/>
    <b>Scope:</b> This path applies to AI decisions made by neural clicks related to user intent.<br/>
    <b>Review Process:</b> Users may request a review of AI decisions via the 'Calibrate' menu in the BCI dashboard.<br/>
    <b>Review Team:</b> Dedicated Neuromodulation Specialists and Compliance Officers.<br/>
    <b>Timeline:</b> Initial response within 2 business days; full resolution within 5 business days.<br/>
    <b>Outcome:</b> Successful appeals will trigger a manual re-calibration of the Chiral™ decoder baseline.
    """
}

# --- CORE UTILITIES ---
def get_llm(st_secrets):
    _enable_instrumentation(st_secrets)
    return ChatGroq(
        temperature=0, 
        model_name="llama-3.3-70b-versatile", 
        api_key=st_secrets["GROQ_API_KEY"]
    )

def list_all_laws(base_dir="Regulations"):
    path_root = Path(base_dir)
    return sorted([str(f.relative_to(path_root.parent)) for f in path_root.rglob('*.pdf')]) if path_root.exists() else []

def extract_pdf_text(uploaded_file):
    reader = PdfReader(io.BytesIO(uploaded_file.read()))
    return "".join([p.extract_text() for p in reader.pages if p.extract_text()])

def smart_web_sifter(org_name, st_secrets):
    """Robust sifter that handles dictionary, .data, or .results return types."""
    try:
        api_key = st_secrets.get("FIRECRAWL_API_KEY")
        if not api_key:
            return "Error: FIRECRAWL_API_KEY missing from secrets."
        
        app = FirecrawlApp(api_key=api_key)
        query = f"{org_name} Chiral AI cognitive governance clinical ethics 2026"
        
        # Call search with standard v2 formatting
        search_result = app.search(
            query,
            limit=3,
            scrape_options={'formats': ['markdown']}
        )
        
        # --- VERSION-AGNOSTIC DATA EXTRACTION ---
        results_list = []
        if hasattr(search_result, 'data'):
            results_list = search_result.data
        elif hasattr(search_result, 'results'):
            results_list = search_result.results
        elif isinstance(search_result, dict):
            results_list = search_result.get('data', []) or search_result.get('results', [])
        
        if not results_list:
            return "Error: No public results found for this entity."
            
        combined_markdown = ""
        for page in results_list:
            # Handle both object-based and dict-based items
            if isinstance(page, dict):
                url = page.get('url', 'Unknown Source')
                md = page.get('markdown', 'No content found.')
            else:
                url = getattr(page, 'url', 'Unknown Source')
                md = getattr(page, 'markdown', 'No content found.')
            
            combined_markdown += f"SOURCE: {url}\n\n{md}\n\n---\n\n"
            
        return combined_markdown
    except Exception as e:
        return f"Firecrawl Error: {str(e)}"

def scrape_url(url, st_secrets):
    """Robust single URL scrape for Firecrawl."""
    try:
        api_key = st_secrets.get("FIRECRAWL_API_KEY")
        app = FirecrawlApp(api_key=api_key)
        scrape_result = app.scrape_url(url, formats=['markdown'])
        
        if isinstance(scrape_result, dict):
            md = scrape_result.get('markdown', 'No content found.')
        else:
            md = getattr(scrape_result, 'markdown', 'No content found.')
            
        return f"SOURCE: {url}\n\n" + md
    except Exception as e:
        return f"Scrape Error: {str(e)}"

# --- AI LOGIC ---
def perform_gap_analysis(content, laws, org, llm):
    today = datetime.now().strftime("%B %d, %Y")
    system_prompt = f"""
    You are a Lead Compliance Strategist. Today: {today}. Statutory Cliff: June 30, 2026.
    Audit the content for {org} against: {laws}. 
    
    STRICT PARAMETERS:
    1. Flag 'Intention-to-Action' (Cognitive AI).
    2. Colorado SB 24-205: Flag missing 'Human Appeal Path'.
    3. NIST 800-171 Rev 3: Check 'Supply Chain Risk Management' (SCRM).
    4. CITE PENALTY: $20,000 per violation.
    """
    response = llm.invoke([("system", system_prompt), ("human", content[:15000])])
    return response.content

def ask_regulatory_chat(prompt, audit_context, llm):
    system_msg = f"You are a 2026 Regulatory Expert. Context: {audit_context[:7000]}"
    response = llm.invoke([("system", system_msg), ("human", prompt)])
    return response.content

def generate_pdf_report(results, org_name, laws):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"STATUTORY REMEDIATION: {org_name}", styles['Title']), Spacer(1,12)]
    story.append(Paragraph(f"Audited Frameworks: {', '.join(laws)}", styles['Italic']))
    
    for line in results.split('\n'):
        if line.strip(): story.append(Paragraph(line, styles['Normal']))
    
    if "Colorado SB 24-205" in str(laws) and ("FLAGGED" in results or "missing" in results.lower()):
        story.append(PageBreak())
        story.append(Paragraph("AUTO-GENERATED REMEDIATION PLAN", styles['Heading2']))
        story.append(Spacer(1,12))
        story.append(Paragraph(REMEDIATION_TEMPLATES["Colorado SB 24-205"], styles['Normal']))
        
    doc.build(story)
    buffer.seek(0)
    return buffer
