import streamlit as st
import os
import tempfile
from engine import get_llm, load_selected_docs, find_and_scrape_company, EconomicImpact, create_pdf

st.set_page_config(page_title="ReadyAudit Engine", layout="wide", page_icon="⚖️")
st.title("⚖️ ReadyAudit: Multi-Framework Remediation")

is_cloud = st.secrets.get("GROQ_API_KEY") is not None

# --- SIDEBAR: THE BEAST CALCULATOR ---
with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info(f"Specialist: Myia Hall | Mode: {'Cloud' if is_cloud else 'Local'}")
    
    st.markdown("### 📈 LIABILITY BEAST (Statutory Risk)")
    tokens = st.number_input("Monthly Token Volume (M):", value=50.0)
    replaced = st.number_input("Affected Employees/Users:", value=10)
    
    # Run the Beast math
    impact = EconomicImpact.calculate_liability(tokens*1000000, replaced)
    
    # Safe key extraction to prevent KeyError crashes
    stat_val = impact.get("statutory", 0)
    total_val = impact.get("total", 0)
    
    st.metric("Statutory Exposure", f"${stat_val:,}", delta="Per CO SB 24-205", delta_color="inverse")
    st.metric("Total Governance Debt", f"${total_val:,}")
    st.session_state.impact_total = total_val

    st.markdown("---")
    st.markdown("### 📜 KNOWLEDGE BASE")
    all_pdfs = [f for r, d, files in os.walk("Regulations") for f in files if f.endswith(".pdf")]
    selected_files = [f for f in sorted(list(set(all_pdfs))) if st.checkbox(f"📄 {f}", value=True, key=f"kb_{f}")]

# --- MAIN INTERFACE ---
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
                from langchain_community.document_loaders import PyPDFLoader
                evidence = "\n".join([p.page_content for p in PyPDFLoader(t_path).load()[:5]])
                regs = "\n".join([d.page_content for d in db.similarity_search(evidence, k=3)])
                
                prompt = f"Conduct a gap analysis. Laws: {regs}. Evidence: {evidence}. Focus on remediation."
                report = get_llm(is_cloud, st.secrets).invoke(prompt).content
                st.session_state.report = report
                st.markdown(report)
                os.remove(t_path)

with tab2:
    st.header("Lead Scout: 2026 Deep-Scrape")
    co_name = st.text_input("Enter Company (e.g., 'Synchron')")
    if st.button("🔍 Scout & Pitch"):
        if not selected_files:
            st.error("Select regulations in the sidebar.")
        else:
            with st.status("Hunting for workforce automation data...") as s:
                web_data = find_and_scrape_company(co_name, st.secrets.get("TAVILY_API_KEY"))
                db = load_selected_docs(selected_files)
                regs = "\n".join([d.page_content for d in db.similarity_search("neural data and algorithmic discrimination", k=3)])
                
                # THE BEAST PROMPT: Forces AI to use the calculator results
                prompt = f"""
                You are a Regulatory Specialist. Draft a cold pitch to {co_name}.
                1. Explicitly mention the statutory exposure of ${st.session_state.impact_total:,.2f} found in our audit.
                2. Use the 'June 30, 2026' Colorado AI Act deadline as the primary hook.
                3. Mention the $20,000 per-violation penalty for workforce automation gaps.
                4. Cite specific 2026 product info from this data: {web_data}.
                5. Discuss Neural Data Privacy (HB 24-1058) if relevant.
                No technical talk. Pure sales and compliance strategy.
                """
                report = get_llm(is_cloud, st.secrets).invoke(prompt).content
                st.session_state.report = report
                st.markdown(report)

if "report" in st.session_state:
    st.download_button("📩 Download Professional Audit", create_pdf(st.session_state.report), file_name="ReadyAudit_Report.pdf")
