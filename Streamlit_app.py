import streamlit as st
import os
import tempfile
from engine import get_llm, load_selected_docs, find_and_scrape_company, EconomicImpact, create_pdf

st.set_page_config(page_title="ReadyAudit Engine", layout="wide", page_icon="⚖️")
st.title("⚖️ ReadyAudit: Multi-Framework Remediation")

is_cloud = st.secrets.get("GROQ_API_KEY") is not None

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info(f"User: Myia Hall | Mode: {'Cloud' if is_cloud else 'Local'}")
    
    st.markdown("### 📜 KNOWLEDGE BASE")
    all_pdfs = [f for r, d, files in os.walk("Regulations") for f in files if f.endswith(".pdf")]
    selected_files = []
    if all_pdfs:
        for f in sorted(list(set(all_pdfs))):
            if st.checkbox(f"📄 {f}", value=True, key=f"kb_{f}"):
                selected_files.append(f)
    
    st.markdown("---")
    st.markdown("### 📈 ROBOT TAX FORECASTER")
    tokens = st.number_input("Monthly Tokens (M):", value=50.0)
    replaced = st.number_input("Roles Impacted:", value=10)
    impact = EconomicImpact.calculate_liability(tokens*1000000, replaced)
    st.metric("Liability Score", f"${impact['total']:,}")

tab1, tab2 = st.tabs(["📁 Document Audit", "🤖 Autonomous Scout"])

with tab1:
    uploaded = st.file_uploader("Upload Evidence PDF", type="pdf")
    if st.button("🚀 Run Manual Audit"):
        if not uploaded or not selected_files:
            st.error("Select regulations and upload evidence.")
        else:
            with st.status("🔍 Analyzing...") as s:
                db = load_selected_docs(selected_files)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.getvalue())
                    t_path = tmp.name
                
                from langchain_community.document_loaders import PyPDFLoader
                evidence = "\n".join([p.page_content for p in PyPDFLoader(t_path).load()[:10]])
                regs = "\n".join([d.page_content for d in db.similarity_search(evidence, k=3)])
                
                prompt = f"Audit the following evidence against the provided regulations. Identify gaps and provide remediation steps.\n\nREGULATIONS:\n{regs}\n\nEVIDENCE:\n{evidence}"
                report = get_llm(is_cloud, st.secrets).invoke(prompt).content
                st.session_state.report = report
                st.markdown(report)
                os.remove(t_path)

with tab2:
    st.header("Lead Scout: Autonomous Prospecting")
    co_name = st.text_input("Company Name:")
    if st.button("🔍 Scout & Pitch"):
        if not selected_files:
            st.error("Select regulations in the sidebar.")
        else:
            with st.status("Scouting public disclosures...") as s:
                web_data = find_and_scrape_company(co_name, st.secrets.get("TAVILY_API_KEY"))
                db = load_selected_docs(selected_files)
                regs = "\n".join([d.page_content for d in db.similarity_search("transparency and liability", k=3)])
                
                # REFINED PROMPT: No technical instructions included.
                prompt = f"""
                You are a Regulatory Specialist. Draft a professional cold pitch to {co_name}. 
                Use the following web data and regulatory requirements to identify specific gaps.
                
                REGULATORY REQUIREMENTS:
                {regs}
                
                WEB DATA FOUND:
                {web_data}
                
                Focus on liability, transparency, and consumer protection. Do not mention API keys or technical settings.
                """
                report = get_llm(is_cloud, st.secrets).invoke(prompt).content
                st.session_state.report = report
                st.markdown(report)

if "report" in st.session_state:
    st.download_button("📩 Download Audit", create_pdf(st.session_state.report), file_name="Audit_Report.pdf", mime="application/pdf")
