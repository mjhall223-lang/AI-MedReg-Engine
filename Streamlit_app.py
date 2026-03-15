import streamlit as st
import os
import sys
import tempfile

# --- CRITICAL: FORCE LOCAL IMPORTS ---
# This tells Streamlit Cloud to look in the current folder for engine.py
sys.path.append(os.path.dirname(__file__))

try:
    from engine import (
        get_llm, 
        find_and_scrape_live_news, 
        EconomicImpact, 
        create_pdf, 
        load_selected_docs
    )
except ImportError as e:
    st.error(f"❌ Specialist Engine Load Error: {e}")
    st.info("Ensure engine.py is in the root folder on GitHub and __init__.py exists.")
    st.stop()

st.set_page_config(page_title="ReadyAudit: Specialist Lead Hunter", layout="wide", page_icon="⚖️")
st.title("⚖️ ReadyAudit: Live-News Liability Engine")

is_cloud = st.secrets.get("GROQ_API_KEY") is not None

# --- SIDEBAR: THE BEAST CALCULATOR ---
with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info(f"Today's Date: March 15, 2026")
    
    st.markdown("### 📈 LIABILITY BEAST (Statutory Risk)")
    tokens = st.number_input("Est. Monthly Tokens (M):", value=50.0)
    replaced = st.number_input("Impacted Users/Employees:", value=10)
    
    impact = EconomicImpact.calculate_liability(tokens*1000000, replaced)
    st.metric("Statutory Risk", f"${impact.get('statutory', 0):,}", delta="Per CO SB 24-205", delta_color="inverse")
    st.metric("Total Governance Debt", f"${impact.get('total', 0):,}")
    st.session_state.impact_total = impact.get('total', 0)

    st.markdown("---")
    st.markdown("### 📜 ACTIVE FRAMEWORKS")
    if not os.path.exists("Regulations"): os.makedirs("Regulations")
    all_pdfs = [f for r, d, files in os.walk("Regulations") for f in files if f.endswith(".pdf")]
    selected_files = [f for f in sorted(list(set(all_pdfs))) if st.checkbox(f"📄 {f}", value=True)]

tab1, tab2 = st.tabs(["📁 Deep Audit (Manual)", "🤖 Autonomous Hunter"])

with tab1:
    st.header("Document Audit: Gap Analysis")
    uploaded = st.file_uploader("Upload Evidence Policy (PDF)", type="pdf")
    if st.button("🚀 Run Specialist Audit"):
        if not uploaded:
            st.error("Please upload a policy to audit.")
        else:
            with st.status("Analyzing against 2026 frameworks...") as s:
                from langchain_community.document_loaders import PyPDFLoader
                db = load_selected_docs(selected_files)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.getvalue())
                    t_path = tmp.name
                
                evidence = "\n".join([p.page_content for p in PyPDFLoader(t_path).load()[:5]])
                regs = "\n".join([d.page_content for d in db.similarity_search(evidence, k=3)]) if db else "No regs found."
                
                prompt = f"Date: March 15, 2026. Gap Analysis. Regs: {regs}. Policy: {evidence}. Focus on HB 24-1058 (Neural Data) and SB 24-205 (Human Appeal)."
                report = get_llm(is_cloud, st.secrets).invoke(prompt).content
                st.session_state.report = report
                st.markdown(report)
                os.remove(t_path)

with tab2:
    st.header("Hunter Mode: Real-Time Prospecting")
    co_name = st.text_input("Enter Company (e.g., 'Synchron', 'Block')")
    if st.button("🔍 Scout Live News & Pitch"):
        with st.status("Scouting March 2026 headlines...") as s:
            news = find_and_scrape_live_news(co_name, st.secrets.get("TAVILY_API_KEY"))
            prompt = f"Regulatory Specialist (March 15, 2026). News: {news}. Draft a pitch for {co_name} using the ${st.session_state.impact_total:,.2f} liability figure. Mention the June 30th CO deadline."
            report = get_llm(is_cloud, st.secrets).invoke(prompt).content
            st.session_state.report = report
            st.markdown(report)

if "report" in st.session_state:
    st.download_button("📩 Download Professional Report", create_pdf(st.session_state.report), file_name="ReadyAudit_Report.pdf")
