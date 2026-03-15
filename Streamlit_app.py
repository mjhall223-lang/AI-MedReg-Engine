import streamlit as st
import os
import tempfile
# Ensure this line is EXACTLY like this:
from engine import get_llm, find_and_scrape_live_news, EconomicImpact, create_pdf, load_selected_docs

st.set_page_config(page_title="ReadyAudit Engine", layout="wide")
st.title("⚖️ ReadyAudit: Live-News Liability Engine")

is_cloud = st.secrets.get("GROQ_API_KEY") is not None

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info(f"Today's Date: March 15, 2026")
    
    st.markdown("### 📈 LIABILITY BEAST")
    tokens = st.number_input("Est. Monthly Tokens (M):", value=50.0)
    replaced = st.number_input("Affected Personnel:", value=10)
    
    impact = EconomicImpact.calculate_liability(tokens*1000000, replaced)
    st.metric("Statutory Risk", f"${impact.get('statutory', 0):,}", delta="Per CO SB 24-205", delta_color="inverse")
    st.metric("Total Governance Debt", f"${impact.get('total', 0):,}")
    st.session_state.impact_total = impact.get('total', 0)

    st.markdown("---")
    st.markdown("### 📜 ACTIVE FRAMEWORKS")
    all_pdfs = [f for r, d, files in os.walk("Regulations") for f in files if f.endswith(".pdf")]
    selected_files = [f for f in sorted(list(set(all_pdfs))) if st.checkbox(f"📄 {f}", value=True)]

tab1, tab2 = st.tabs(["📁 Deep Audit", "🤖 Autonomous Hunter"])

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
                prompt = f"Conduct a gap analysis. Laws: {regs}. Evidence: {evidence}. Remediation steps?"
                report = get_llm(is_cloud, st.secrets).invoke(prompt).content
                st.session_state.report = report
                st.markdown(report)
                os.remove(t_path)

with tab2:
    st.header("Lead Hunter: Real-Time Prospecting")
    co_name = st.text_input("Enter Target Company (e.g., 'Block', 'Synchron')")
    if st.button("🔍 Scout Live News & Pitch"):
        if not selected_files:
            st.error("Select regulations in the sidebar.")
        else:
            with st.status("Scraping March 2026 headlines...") as s:
                news_data = find_and_scrape_live_news(co_name, st.secrets.get("TAVILY_API_KEY"))
                db = load_selected_docs(selected_files)
                regs = "\n".join([d.page_content for d in db.similarity_search("neural data privacy", k=3)])
                
                prompt = f"""You are a Regulatory Specialist. Date: March 15, 2026.
                LIVE NEWS for {co_name}: {news_data}
                LAWS: {regs}
                DRAFT PITCH: Mention a specific 2026 news event found. Use the ${st.session_state.impact_total:,.2f} liability as the hook. Focus on the June 30th Colorado deadline."""
                
                report = get_llm(is_cloud, st.secrets).invoke(prompt).content
                st.session_state.report = report
                st.markdown(report)

if "report" in st.session_state:
    st.download_button("📩 Download Professional Report", create_pdf(st.session_state.report), file_name="ReadyAudit_Report.pdf")
    
