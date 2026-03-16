import streamlit as st
import io, pypdf, requests, os
from pathlib import Path
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain_groq import ChatGroq
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# --- 1. THE RECURSIVE ENGINE ---
def get_llm():
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st.secrets["GROQ_API_KEY"])

def list_all_laws(base_dir="Regulations"):
    """TARGETED RECURSIVE SIFTER: Finds PDFs in your nested 'Regulations/Regulations' structure."""
    path_root = Path(base_dir)
    # .rglob handles any number of nested folders automatically
    return sorted([str(f.relative_to(path_root.parent)) for f in path_root.rglob('*.pdf')])

def generate_pdf_report(results, org_name, hole_type, laws):
    """Generates the professional report with the $20k penalty warning."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"REMEDIATION AUDIT: {org_name}", styles['Title']), Spacer(1,12)]
    
    # Financial Exposure Box
    penalty = "$20,000 per violation" if "Human" in hole_type else "$10,000 per violation"
    story.append(Paragraph("FINANCIAL EXPOSURE SUMMARY", styles['Heading2']))
    story.append(Paragraph(f"Statutory Risk: {penalty}", styles['Normal']))
    story.append(Paragraph(f"Enforcement Deadline: June 30, 2026", styles['Normal']))
    story.append(Paragraph(f"Audited Laws: {', '.join(laws)}", styles['Italic']))
    story.append(Spacer(1, 18))

    for line in results.split('\n'):
        if line.strip(): story.append(Paragraph(line, styles['Normal']))
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- 2. UI & PERSISTENCE ---
st.set_page_config(page_title="ReadyAudit: Remediation Engine", layout="wide")
all_laws = list_all_laws()

if "content" not in st.session_state: st.session_state.content = ""
if "results" not in st.session_state: st.session_state.results = ""
if "hole_type" not in st.session_state: st.session_state.hole_type = "General"

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
org = st.text_input("Lead Entity", value="Synchron")
mode = st.radio("Audit Mode:", ["File Upload", "Web Sifter"])

# Step 1: Load content
if mode == "File Upload":
    f = st.file_uploader("Upload Policy PDF", type=['pdf'])
    if f:
        reader = pypdf.PdfReader(io.BytesIO(f.read()))
        st.session_state.content = "".join([p.extract_text() for p in reader.pages if p.extract_text()])
elif mode == "Web Sifter" and st.button("🔍 Sift Public Web"):
    with DDGS() as ddgs:
        results = list(ddgs.text(f"{org} AI policy 2026", max_results=1))
        if results:
            res = requests.get(results[0]['href'], timeout=10)
            st.session_state.content = BeautifulSoup(res.text, 'html.parser').get_text()
            st.success("Public Policy Loaded.")

# Step 2: Audit & Gap Identification
if st.button("🛠️ Run Remediation Audit") and st.session_state.content:
    with st.status("Analyzing Statutory Gaps..."):
        llm = get_llm()
        prompt = f"Audit {org} against {active_selections}. Find the 'Hole'. Content: {st.session_state.content[:4000]}"
        st.session_state.results = llm.invoke(prompt).content
        if "human" in st.session_state.results.lower(): st.session_state.hole_type = "Human Appeal"
        st.markdown(st.session_state.results)

# Step 3: Professional Export
if st.session_state.results:
    pdf = generate_pdf_report(st.session_state.results, org, st.session_state.hole_type, active_selections)
    st.download_button("📥 Download Remediation Report", data=pdf, file_name=f"{org}_Audit.pdf")
