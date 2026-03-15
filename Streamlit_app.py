import streamlit as st
import os
import tempfile
from engine import get_llm, load_selected_docs, find_and_scrape_company, EconomicImpact, create_pdf

st.set_page_config(page_title="ReadyAudit Engine", layout="wide", page_icon="⚖️")
st.title("⚖️ ReadyAudit: Multi-Framework Remediation")

# --- DETECT ENVIRONMENT ---
is_cloud = st.secrets.get("GROQ_API_KEY") is not None

# --- SIDEBAR ---
with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info(f"User: Myia Hall | Mode: {'Cloud (Groq)' if is_cloud else 'Local (Ollama)'}")
    
    st.markdown("### 📜 KNOWLEDGE BASE")
    all_pdfs = [f for r, d, files in os.walk("Regulations") for f in files if f.endswith(".pdf")]
    
    selected_files = []
    if all_pdfs:
        for f in sorted(list(set(all_pdfs))):
            if st.checkbox(f"📄 {f}", value=True, key=f"kb_{f}"):
                selected_files.append(f)
    else:
        st.warning("Upload PDFs to /Regulations folder")

    st.markdown("---")
    st.markdown("### 📈 ROBOT TAX FORECASTER")
    tokens = st.number_input("Monthly Tokens (M):", value=50.0)
    replaced = st.number_input("Roles Impacted:", value=10)
    impact = EconomicImpact.calculate_liability(tokens*1000000, replaced)
    st.metric("Estimated Tax Liability", f"${impact['total']:,}")

# --- MAIN INTERFACE ---
tab1, tab2 = st.tabs(["📁 Document Audit", "🤖 Autonomous Scout"])

with tab1:
    uploaded = st.file_uploader("Upload Evidence PDF", type="pdf")
    if st.button("🚀 Run Manual Audit"):
        if not uploaded or not selected_files:
            st.error("Upload evidence and select regulations in the sidebar.")
        else:
            with st.status("🔍 Analyzing Gaps...") as s:
                db = load_selected_docs(selected_files)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.getvalue())
                    t_path = tmp.name
                
                from langchain_community.document_loaders import PyPDFLoader
                evidence = "\n".join([p.page_content for p in PyPDFLoader(t_path).load()[:10]])
                regs = "\n".join([d.page_content for d in db.similarity_search(evidence, k=3)])
                
                prompt = f"AUDIT TASK: Analyze evidence against regulations. REGS: {regs} EVIDENCE: {evidence}"
                report = get_llm(is_cloud, st.secrets).invoke(prompt).content
                st.session_state.report = report
                st.markdown(report)
                os.remove(t_path)

with tab2:
    st.header("Lead Scout: Autonomous Prospecting")
    co_name = st.text_input("Enter Company Name:")
    if st.button("🔍 Scout & Pitch"):
        if not selected_files:
            st.error("Select regulations in the sidebar first.")
        else:
            with st.status("Scouting public AI policies...") as s:
                t_key = st.secrets.get("TAVILY_API_KEY")
                web_data = find_and_scrape_company(co_name, t_key)
                db = load_selected_docs(selected_files)
                regs = "\n".join([d.page_content for d in db.similarity_search("transparency disclosure", k=3)])
                
                prompt = f"SCOUT TASK: Draft a cold pitch to {co_name} based on compliance gaps. REGS: {regs} WEB_DATA: {web_data}"
                report = get_llm(is_cloud, st.secrets).invoke(prompt).content
                st.session_state.report = report
                st.markdown(report)

# --- GLOBAL DOWNLOAD ---
if "report" in st.session_state:
    st.download_button(
        label="📩 Download Certified Audit",
        data=create_pdf(st.session_state.report),
        file_name="ReadyAudit_Report.pdf",
        mime="application/pdf"
    )
