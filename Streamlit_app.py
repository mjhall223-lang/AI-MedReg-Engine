import streamlit as st
import io, pypdf, requests, os, sys
from pathlib import Path
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain_groq import ChatGroq
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# --- 1. THE ENGINE ---
def get_llm():
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st.secrets["GROQ_API_KEY"])

def list_all_laws(base_dir="Regulations"):
    """TARGETED RECURSIVE SIFTER: Finds PDFs in your nested 'Regulations/Regulations' structure."""
    path_root = Path(base_dir)
    if not path_root.exists():
        return []
    return sorted([str(f.relative_to(path_root.parent)) for f in path_root.rglob('*.pdf')])

def web_sifter(org_name):
    """ACTUAL WEB SEARCH: Hunts for public policies and returns text."""
    try:
        with DDGS() as ddgs:
            # Targeted search for 2026 AI compliance docs
            search_query = f"{org_name} AI governance policy 2026"
            results = list(ddgs.text(search_query, max_results=1))
            
            if not results:
                return "Error: No public results found for this entity."
            
            url = results[0]['href']
            res = requests.get(url, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            
            # Clean up the text
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text(separator=' ', strip=True)
            return f"SOURCE: {url}\n\n{text}"
    except Exception as e:
        return f"Web Sifter Error: {str(e)}"

def generate_pdf_report(results, org_name, hole_type, laws):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"REMEDIATION AUDIT: {org_name}", styles['Title']), Spacer(1,12)]
    
    penalty = "$20,000 per violation" if "Human" in hole_type else "$10,000 per violation"
    story.append(Paragraph("FINANCIAL EXPOSURE SUMMARY", styles['Heading2']))
    story.append(Paragraph(f"Statutory Risk: {penalty}", styles['Normal']))
    story.append(Paragraph(f"Laws Audited: {', '.join(laws)}", styles['Italic']))
    story.append(Spacer(1, 18))

    for line in results.split('\n'):
        if line.strip():
            story.append(Paragraph(line, styles['Normal']))
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- 2. UI & SESSION STATE ---
st.set_page_config(page_title="ReadyAudit: Remediation Engine", layout="wide")

# Initialize persistent memory
if "audit_content" not in st.session_state: st.session_state.audit_content = ""
if "audit_results" not in st.session_state: st.session_state.audit_results = ""
if "hole_type" not in st.session_state: st.session_state.hole_type = "General"

all_laws = list_all_laws()

# Safety Check for Blank Screen
if not all_laws:
    st.error("⚠️ No Laws Found. Check your /Regulations folder.")
    st.stop()

# Callback for Toggle logic
def sync_selections():
    if st.session_state.select_all_check:
        st.session_state.selected_laws = all_laws
    else:
        st.session_state.selected_laws = []

with st.sidebar:
    st.header("🛡️ LAW DATABASE")
    st.checkbox("Select All Laws", value=True, key="select_all_check", on_change=sync_selections)
    active_selections = st.multiselect("Audit Focus:", options=all_laws, key="selected_laws")
    st.divider()
    st.error("Enforcement Cliff: June 30, 2026")

st.header("📁 Remediation & Gap Analysis Engine")
org_name = st.text_input("Lead Entity", value="Synchron")
mode = st.radio("Mode:", ["Web Sifter", "File Upload"])

# --- STEP 1: CONTENT LOADING ---
if mode == "Web Sifter":
    if st.button("🔍 Sift Public Web"):
        with st.spinner(f"Sifting the web for {org_name}..."):
            st.session_state.audit_content = web_sifter(org_name)
            if "Error:" not in st.session_state.audit_content:
                st.success("Public Policy Loaded into Memory.")
            else:
                st.error(st.session_state.audit_content)

elif mode == "File Upload":
    f = st.file_uploader("Upload PDF", type=['pdf'])
    if f:
        reader = pypdf.PdfReader(io.BytesIO(f.read()))
        st.session_state.audit_content = "".join([p.extract_text() for p in reader.pages if p.extract_text()])

# Show a preview so you know content is loaded
if st.session_state.audit_content:
    with st.expander("📄 Loaded Policy Content Preview"):
        st.text(st.session_state.audit_content[:1000] + "...")

# --- STEP 2: RUN AUDIT ---
if st.button("🛠️ Run Remediation Audit"):
    if st.session_state.audit_content and active_selections:
        with st.status("Analyzing Statutory Gaps..."):
            llm = get_llm()
            prompt = f"""
            Audit {org_name} against these laws: {active_selections}. 
            Identify the 'Holes' (specific violations). 
            Today's Date: March 16, 2026. Cliff: June 30, 2026.
            Content: {st.session_state.audit_content[:4000]}
            """
            st.session_state.audit_results = llm.invoke(prompt).content
            
            # Risk Coster
            if "human" in st.session_state.audit_results.lower():
                st.session_state.hole_type = "Human Appeal"
            
            st.markdown(st.session_state.audit_results)
    else:
        st.warning("Please load content (Sift or Upload) and select laws first.")

# --- STEP 3: EXPORT ---
if st.session_state.audit_results:
    pdf = generate_pdf_report(st.session_state.audit_results, org_name, st.session_state.hole_type, active_selections)
    st.download_button("📥 Download Remediation Report", data=pdf, file_name=f"{org_name}_Audit_2026.pdf")
