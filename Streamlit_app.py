import streamlit as st
import os
import tempfile
from engine import get_llm, find_and_scrape_company, EconomicImpact, create_pdf, load_selected_docs

st.set_page_config(page_title="ReadyAudit Engine", layout="wide", page_icon="⚖️")
st.title("⚖️ ReadyAudit: Multi-Framework Remediation")

is_cloud = st.secrets.get("GROQ_API_KEY") is not None

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.markdown(f"**Specialist:** Myia Hall")
    
    st.markdown("### 📈 LIABILITY BEAST (Statutory Risk)")
    # Defaults set to 2026 industry averages
    tokens = st.number_input("Est. Monthly Tokens (M):", value=50.0)
    replaced = st.number_input("Employees/Applicants Impacted:", value=10)
    impact = EconomicImpact.calculate_liability(tokens*1000000, replaced)
    
    st.metric("Statutory Exposure", f"${impact['statutory']:,}", delta="Per CO SB 24-205", delta_color="inverse")
    st.metric("Total Governance Debt", f"${impact['total']:,}")
    st.session_state.impact_total = impact['total']

tab1, tab2 = st.tabs(["📁 Document Audit", "🤖 Autonomous Scout"])

# [Omitted tab1 for brevity - same as previous version]

with tab2:
    st.header("Lead Scout: Deep-Scrape & Liability Pitch")
    co_name = st.text_input("Enter Target Company (e.g., 'Synchron' or 'Neuralink')")
    
    if st.button("🔍 Run Deep Scout"):
        with st.status("Hunting for workforce impact data...") as s:
            t_key = st.secrets.get("TAVILY_API_KEY")
            web_data = find_and_scrape_company(co_name, t_key)
            
            # The AI now specifically looks for "employees" in the data to fill the calculator
            prompt = f"""
            Identify the core AI products and workforce automation plans for {co_name}.
            
            WEB DATA: {web_data}
            
            CALCULATED RISK: Current sidebar exposure is set to ${st.session_state.impact_total}.
            
            DRAFT A PITCH:
            1. Mention their specific AI systems (e.g., Chiral, BCI).
            2. Use the ${st.session_state.impact_total} figure as the 'Statutory Exposure' under the June 30th Colorado AI Act deadline.
            3. Explicitly mention the $20,000 per-violation penalty for failing to provide 'Reasonable Care' documentation.
            4. If the web data mentions job cuts or hiring AI, emphasize 'Algorithmic Discrimination' risk.
            """
            
            report = get_llm(is_cloud, st.secrets).invoke(prompt).content
            st.session_state.report = report
            st.markdown(report)

if "report" in st.session_state:
    st.download_button("📩 Download Professional Audit", create_pdf(st.session_state.report), file_name="Audit.pdf")
