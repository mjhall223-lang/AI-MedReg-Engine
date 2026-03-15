import streamlit as st
import os
import tempfile
from engine import get_llm, load_selected_docs, find_and_scrape_company, EconomicImpact, create_pdf

st.set_page_config(page_title="ReadyAudit Engine", layout="wide")
st.title("⚖️ ReadyAudit: Multi-Framework Remediation")

is_cloud = st.secrets.get("GROQ_API_KEY") is not None

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info(f"Specialist: Myia Hall | Mode: {'Cloud' if is_cloud else 'Local'}")
    
    st.markdown("### 📜 KNOWLEDGE BASE")
    all_pdfs = [f for r, d, files in os.walk("Regulations") for f in files if f.endswith(".pdf")]
    selected_files = [f for f in sorted(list(set(all_pdfs))) if st.checkbox(f"📄 {f}", value=True, key=f"kb_{f}")]
    
    st.markdown("---")
    st.markdown("### 📉 LIABILITY CALCULATOR")
    tokens = st.number_input("Monthly Tokens (M):", value=50.0)
    replaced = st.number_input("Roles Automated:", value=10)
    impact = EconomicImpact.calculate_liability(tokens*1000000, replaced)
    st.metric("Potential Exposure", f"${impact['total']:,}")

tab1, tab2 = st.tabs(["📁 Document Audit", "🤖 Autonomous Scout"])

with tab1:
    uploaded = st.file_uploader("Upload Evidence PDF", type="pdf")
    if st.button("🚀 Run Manual Audit"):
        if not uploaded or not selected_files:
            st.error("Select regulations and upload evidence.")
        else:
            with st.status("🔍 Analyzing Gaps...") as s:
                db = load_selected_docs(selected_files)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.getvalue())
                    t_path = tmp.name
                evidence = "\n".join([p.page_content for p in PyPDFLoader(t_path).load()[:5]])
                regs = "\n".join([d.page_content for d in db.similarity_search(evidence, k=3)])
                prompt = f"Conduct a gap analysis. Use these laws: {regs}. Evidence: {evidence}. Provide remediation steps."
                report = get_llm(is_cloud, st.secrets).invoke(prompt).content
                st.session_state.report = report
                st.markdown(report)

with tab2:
    st.header("Lead Scout: Autonomous Prospecting")
    co_name = st.text_input("Company Name:")
    if st.button("🔍 Scout & Pitch"):
        if not selected_files:
            st.error("Select regulations in the sidebar.")
        else:
            with st.status("Gathering intelligence...") as s:
                web_data = find_and_scrape_company(co_name, st.secrets.get("TAVILY_API_KEY"))
                db = load_selected_docs(selected_files)
                regs = "\n".join([d.page_content for d in db.similarity_search("neural data and liability", k=3)])
                
                # SPECIALIST PROMPT
                prompt = f"""
                You are a Regulatory Specialist. Draft a cold pitch to {co_name}. 
                1. Mention their specific 2026 products (e.g., Chiral, BCI, or AI systems) found in this data: {web_data}.
                2. Explicitly cite the June 30, 2026 Colorado AI Act deadline.
                3. Mention the new FDA QMSR transition (Feb 2026) if they are medical.
                4. Use this context from the regulations: {regs}.
                Focus on liability and the 'Affirmative Defense.' No technical meta-talk.
                """
                report = get_llm(is_cloud, st.secrets).invoke(prompt).content
                st.session_state.report = report
                st.markdown(report)

if "report" in st.session_state:
    st.download_button("📩 Download Audit", create_pdf(st.session_state.report), file_name="ReadyAudit_Report.pdf")
