import streamlit as st
import os
import tempfile
from engine import get_llm, load_selected_docs, find_and_scrape_company, EconomicImpact, create_pdf

st.set_page_config(page_title="ReadyAudit Engine", layout="wide")
st.title("⚖️ ReadyAudit: Multi-Framework Remediation")

# --- CLOUD DETECTION ---
is_cloud = st.secrets.get("GROQ_API_KEY") is not None

# --- SIDEBAR ---
with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info(f"User: Myia Hall | Mode: {'Cloud' if is_cloud else 'Local'}")
    
    st.markdown("### 📜 KNOWLEDGE BASE")
    all_pdfs = []
    if os.path.exists("Regulations"):
        for r, _, f_list in os.walk("Regulations"):
            for f in f_list:
                if f.endswith(".pdf"): all_pdfs.append(f)
    
    selected_files = []
    for f in sorted(list(set(all_pdfs))):
        if st.checkbox(f"📄 {f}", value=True, key=f"kb_{f}"):
            selected_files.append(f)
    
    st.markdown("---")
    st.markdown("### 📈 ROBOT TAX FORECASTER")
    tokens = st.number_input("Monthly Tokens (M):", value=50.0)
    replaced = st.number_input("Roles Impacted:", value=10)
    impact = EconomicImpact.calculate_liability(tokens*1000000, replaced)
    st.metric("Total Liability", f"${impact['total']:,}")

# --- MAIN INTERFACE ---
tab1, tab2 = st.tabs(["📁 Document Audit", "🤖 Autonomous Scout"])

with tab1:
    uploaded = st.file_uploader("Upload Evidence PDF", type="pdf")
    if st.button("🚀 Run Analysis"):
        if not uploaded or not selected_files:
            st.error("Select regulations and upload evidence.")
        else:
            with st.status("Analyzing...") as s:
                db = load_selected_docs(selected_files)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.getvalue())
                    t_path = tmp.name
                
                # Simple extraction & similarity search
                from langchain_community.document_loaders import PyPDFLoader
                evidence = "\n".join([p.page_content for p in PyPDFLoader(t_path).load()[:5]])
                regs = "\n".join([d.page_content for d in db.similarity_search(evidence, k=3)])
                
                prompt = f"Audit this AI project evidence against these regulations. REGS: {regs} EVIDENCE: {evidence}"
                report = get_llm(is_cloud, st.secrets).invoke(prompt).content
                st.session_state.report = report
                st.markdown(report)

with tab2:
    st.header("Lead Scout: Auto-Prospecting")
    co_name = st.text_input("Company to Audit:")
    if st.button("🔍 Scout & Pitch"):
        with st.status("Scouting Web...") as s:
            t_key = st.secrets.get("TAVILY_API_KEY")
            web_data = find_and_scrape_company(co_name, t_key)
            db = load_selected_docs(selected_files)
            regs = "\n".join([d.page_content for d in db.similarity_search("transparency", k=3)])
            
            prompt = f"Draft a cold pitch to {co_name} based on these regulations: {regs}. Web Data: {web_data}"
            report = get_llm(is_cloud, st.secrets).invoke(prompt).content
            st.session_state.report = report
            st.markdown(report)

if "report" in st.session_state:
    st.download_button("📩 Download Report", create_pdf(st.session_state.report))
