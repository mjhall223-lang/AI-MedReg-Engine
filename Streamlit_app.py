import streamlit as st
import os
import tempfile
from engine import get_llm, load_selected_docs, find_and_scrape_company, EconomicImpact, create_pdf

# Config MUST be first
st.set_page_config(page_title="ReadyAudit Engine", page_icon="⚖️", layout="wide")
st.title("⚖️ ReadyAudit: Multi-Framework Remediation Engine")

is_cloud = "GROQ_API_KEY" in st.secrets

# --- 1. THE SIDEBAR (ALWAYS VISIBLE) ---
with st.sidebar:
    st.markdown("## 🛡️ AUDIT CONTROLS")
    st.markdown(f"**Specialist:** Myia Hall")
    st.markdown("---")

    # Dynamic File Toggles
    all_pdfs = []
    if os.path.exists("Regulations"):
        for root, _, files in os.walk("Regulations"):
            for file in files:
                if file.endswith(".pdf"): all_pdfs.append(file)

    if all_pdfs:
        st.markdown("### 📜 ACTIVE KNOWLEDGE BASE")
        selected_files = []
        for f in sorted(list(set(all_pdfs))): 
            # Use keys to prevent UI reset when switching tabs
            if st.checkbox(f"📄 {f}", value=True, key=f"toggle_{f}"):
                selected_files.append(f)
        st.session_state.selected_files = selected_files
    else:
        st.error("No PDFs found in 'Regulations/'")

    st.markdown("---")
    st.markdown("### 📈 ECONOMIC FORECASTER")
    tokens = st.number_input("Est. Monthly Tokens (M):", value=50.0)
    replaced = st.number_input("Roles Replaced:", value=50)
    impact = EconomicImpact.calculate_liability(token_usage=tokens*1000000, replaced_staff=replaced)
    st.metric("Tax Liability", f"${impact['total']:,}")
    st.session_state.impact_total = impact['total']

# --- 2. THE TABS (MAIN CONTENT) ---
tab1, tab2 = st.tabs(["📁 Manual Audit", "🤖 Autonomous Scout"])

with tab1:
    uploaded_file = st.file_uploader("Upload Project Evidence", type="pdf")
    if st.button("🚀 Run Manual Audit"):
        if not uploaded_file or not st.session_state.get('selected_files'):
            st.error("Missing Evidence or Knowledge Base Toggles.")
        else:
            with st.status("🔍 ANALYZING GAPS...") as status:
                db = load_selected_docs(st.session_state.selected_files)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                # Logic for Audit
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(tmp_path)
                evidence_text = "\n\n".join([c.page_content for c in loader.load()[:10]])
                search_docs = db.similarity_search("human oversight, bias", k=5)
                reg_context = "\n".join([d.page_content for d in search_docs])
                
                prompt = f"Perform Audit. REGS: {reg_context} EVIDENCE: {evidence_text} TAX: {st.session_state.impact_total}"
                report = get_llm(is_cloud, st.secrets).invoke(prompt).content
                st.session_state.final_report = report
                st.markdown(report)
                os.remove(tmp_path)

with tab2:
    st.header("Lead Scout: Autonomous Gap Analysis")
    target_company = st.text_input("Enter Company Name (e.g., 'Neuralink')")
    
    if st.button("🔍 Find, Scrape, & Audit"):
        if not st.session_state.get('selected_files'):
            st.error("Select at least one Regulation in the sidebar first!")
        else:
            with st.status("Searching for public AI disclosures...") as status:
                evidence = find_and_scrape_company(target_company)
                if not evidence:
                    st.error("Could not find public data.")
                else:
                    db = load_selected_docs(st.session_state.selected_files)
                    search_docs = db.similarity_search("oversight, transparency", k=5)
                    reg_context = "\n".join([d.page_content for d in search_docs])
                    
                    prompt = f"Scout Audit for {target_company}. REGS: {reg_context} POLICY: {evidence}"
                    report = get_llm(is_cloud, st.secrets).invoke(prompt).content
                    st.session_state.final_report = report
                    st.markdown(report)

# --- 3. DOWNLOAD FOOTER ---
if "final_report" in st.session_state:
    st.download_button("📄 Download Audit", create_pdf(st.session_state.final_report), file_name="ReadyAudit_Report.pdf")
