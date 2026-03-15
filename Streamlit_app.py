import streamlit as st
import os
import tempfile
# Absolute import from local engine file
from engine import get_llm, find_and_scrape_live_news, EconomicImpact, create_pdf, load_selected_docs

st.set_page_config(page_title="ReadyAudit: Specialist Lead Hunter", layout="wide", page_icon="⚖️")

# Ensure required folders exist
if not os.path.exists("Regulations"):
    os.makedirs("Regulations")

is_cloud = st.secrets.get("GROQ_API_KEY") is not None

# --- SIDEBAR: THE BEAST CALCULATOR ---
with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info(f"Today's Date: March 15, 2026")
    
    st.markdown("### 📈 LIABILITY BEAST (Statutory Risk)")
    tokens = st.number_input("Est. Monthly Tokens (M):", value=50.0)
    replaced = st.number_input("Impacted Users/Employees:", value=10, help="Each person impacted is a $20k violation in CO.")
    
    # Calculation
    impact = EconomicImpact.calculate_liability(tokens*1000000, replaced)
    
    st.metric("Statutory Exposure", f"${impact.get('statutory', 0):,}", delta="Per CO SB 24-205", delta_color="inverse")
    st.metric("Total Governance Debt", f"${impact.get('total', 0):,}")
    st.session_state.impact_total = impact.get('total', 0)

    st.markdown("---")
    st.markdown("### 📜 KNOWLEDGE BASE")
    all_pdfs = [f for r, d, files in os.walk("Regulations") for f in files if f.endswith(".pdf")]
    selected_files = [f for f in sorted(list(set(all_pdfs))) if st.checkbox(f"📄 {f}", value=True)]

# --- MAIN INTERFACE ---
tab1, tab2 = st.tabs(["📁 Deep Audit (Manual)", "🤖 Autonomous Lead Hunter"])

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
                regs = "\n".join([d.page_content for d in db.similarity_search(evidence, k=3)]) if db else "No local regs selected."
                
                prompt = f"""
                Conduct a specialist gap analysis. 
                Laws/Regs: {regs}
                Policy Content: {evidence}
                
                Search for 3 specific 2026 violations: 
                1. Neural Data Consent (HB 24-1058)
                2. Meaningful Human Appeal (SB 24-205)
                3. FDA QMSR (ISO 13485) alignment.
                """
                report = get_llm(is_cloud, st.secrets).invoke(prompt).content
                st.session_state.report = report
                st.markdown(report)
                os.remove(t_path)

with tab2:
    st.header("Hunter Mode: Real-Time Pitching")
    co_name = st.text_input("Enter Company (e.g., 'Synchron', 'PayPal', 'Block')")
    if st.button("🔍 Scout Live News & Pitch"):
        with st.status("Scouting March 2026 headlines...") as s:
            news = find_and_scrape_live_news(co_name, st.secrets.get("TAVILY_API_KEY"))
            
            prompt = f"""
            You are a Regulatory Specialist (Date: March 15, 2026).
            Draft a high-stakes pitch to {co_name} based on this news: {news}.
            
            HOOKS:
            - Mention a specific 2026 headline found in the news.
            - Cite the ${st.session_state.impact_total:,.2f} liability from our audit.
            - Emphasize the June 30, 2026 Colorado AI Act enforcement deadline.
            - Focus on the 'Affirmative Defense' provided by our audit framework.
            """
            report = get_llm(is_cloud, st.secrets).invoke(prompt).content
            st.session_state.report = report
            st.markdown(report)

if "report" in st.session_state:
    st.download_button("📩 Download Final Audit PDF", create_pdf(st.session_state.report), file_name="ReadyAudit_Report.pdf")
