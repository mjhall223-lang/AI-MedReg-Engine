import streamlit as st
import os
import sys
import tempfile

# Force Python to find local engine.py
sys.path.append(os.path.dirname(__file__))

try:
    from engine import (
        get_llm, find_and_scrape_live_news, EconomicImpact, 
        create_pdf, load_selected_docs
    )
except ImportError as e:
    st.error(f"❌ Engine Load Error: {e}")
    st.stop()

st.set_page_config(page_title="ReadyAudit: Specialist Hub", layout="wide")
st.title("⚖️ ReadyAudit: Specialist Liability Engine")

is_cloud = st.secrets.get("GROQ_API_KEY") is not None

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info("Today: March 15, 2026")
    
    st.markdown("### 📈 LIABILITY CALCULATOR")
    replaced = st.number_input("Affected Personnel/Users:", value=10, step=1)
    
    impact = EconomicImpact.calculate_liability(replaced_staff=replaced)
    st.metric("Statutory Exposure", f"${impact['statutory']:,}", delta="Per CO SB 24-205")
    st.metric("Total Governance Debt", f"${impact['total']:,}")
    st.session_state.impact_total = impact['total']

    st.markdown("---")
    st.markdown("### 📜 ACTIVE FRAMEWORKS")
    if not os.path.exists("Regulations"): os.makedirs("Regulations")
    all_pdfs = [f for f in os.listdir("Regulations") if f.endswith(".pdf")]
    selected_files = [f for f in all_pdfs if st.checkbox(f"📄 {f}", value=True)]

tab1, tab2 = st.tabs(["📁 Deep Audit", "🤖 Autonomous Hunter"])

with tab1:
    uploaded = st.file_uploader("Upload Evidence Policy", type="pdf")
    if st.button("🚀 Run Audit"):
        if not uploaded: st.error("Upload a file.")
        else:
            with st.status("Analyzing...") as s:
                db = load_selected_docs(selected_files)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.getvalue())
                    t_path = tmp.name
                from langchain_community.document_loaders import PyPDFLoader
                evidence = "\n".join([p.page_content for p in PyPDFLoader(t_path).load()[:5]])
                regs = "\n".join([d.page_content for d in db.similarity_search(evidence, k=3)]) if db else "No regs."
                
                prompt = f"Gap Analysis for March 2026. Laws: {regs}. Policy: {evidence}. Identify violations of HB 24-1058 and SB 24-205."
                report = get_llm(is_cloud, st.secrets).invoke(prompt).content
                st.session_state.report = report
                st.markdown(report)
                os.remove(t_path)

with tab2:
    st.header("Lead Hunter")
    co_name = st.text_input("Enter Company Name")
    if st.button("🔍 Scout & Pitch"):
        with st.status("Scouting March 2026 news...") as s:
            news = find_and_scrape_live_news(co_name, st.secrets.get("TAVILY_API_KEY"))
            prompt = f"""You are a Regulatory Specialist (March 15, 2026). 
            News found for {co_name}: {news}.
            Draft a pitch using the ${st.session_state.impact_total:,.2f} liability figure. 
            Cite the June 30, 2026 Colorado AI Act deadline."""
            report = get_llm(is_cloud, st.secrets).invoke(prompt).content
            st.session_state.report = report
            st.markdown(report)

if "report" in st.session_state:
    st.download_button("📩 Download PDF", create_pdf(st.session_state.report), file_name="Audit_Report.pdf")
