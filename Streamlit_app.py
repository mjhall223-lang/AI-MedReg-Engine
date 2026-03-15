import streamlit as st
import os
import sys
import tempfile
sys.path.append(os.path.dirname(__file__))

from engine import (
    get_llm, find_and_scrape_live_news, EconomicImpact, 
    create_pdf, load_selected_docs, extract_headcount
)

st.set_page_config(page_title="ReadyAudit: Lead Hunter", layout="wide")
is_cloud = st.secrets.get("GROQ_API_KEY") is not None
llm = get_llm(is_cloud, st.secrets)

# --- STATE MANAGEMENT ---
if "suggested_headcount" not in st.session_state:
    st.session_state.suggested_headcount = 10

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info("Today: March 15, 2026")
    
    st.markdown("### 📈 LIABILITY CALCULATOR")
    # This input now listens to the session state
    replaced = st.number_input("Affected Personnel:", 
                               value=st.session_state.suggested_headcount, 
                               step=1, key="headcount_input")
    
    impact = EconomicImpact.calculate_liability(replaced_staff=replaced)
    st.metric("Statutory Risk", f"${impact['statutory']:,}", delta="Per CO SB 24-205")
    st.metric("Total Governance Debt", f"${impact['total']:,}")
    st.session_state.impact_total = impact['total']

    st.markdown("---")
    st.markdown("### 📜 FRAMEWORKS")
    if not os.path.exists("Regulations"): os.makedirs("Regulations")
    all_pdfs = [f for f in os.listdir("Regulations") if f.endswith(".pdf")]
    selected_files = [f for f in all_pdfs if st.checkbox(f"📄 {f}", value=True)]

tab1, tab2 = st.tabs(["📁 Deep Audit", "🤖 Autonomous Hunter"])

with tab1:
    uploaded = st.file_uploader("Upload Policy", type="pdf")
    if st.button("🚀 Run Audit"):
        if not uploaded: st.error("Upload a file.")
        else:
            with st.status("Analyzing..."):
                db = load_selected_docs(selected_files)
                from langchain_community.document_loaders import PyPDFLoader
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.getvalue())
                    t_path = tmp.name
                evidence = "\n".join([p.page_content for p in PyPDFLoader(t_path).load()[:5]])
                regs = "\n".join([d.page_content for d in db.similarity_search(evidence, k=3)]) if db else "No regs."
                prompt = f"Date: March 15, 2026. Gap Analysis. Regs: {regs}. Policy: {evidence}."
                report = llm.invoke(prompt).content
                st.session_state.report = report
                st.markdown(report)
                os.remove(t_path)

with tab2:
    st.header("Lead Hunter: Real-Time Sifting")
    co_name = st.text_input("Enter Company (e.g., 'Block', 'Synchron')")
    if st.button("🔍 Scout & Auto-Calculate"):
        with st.status("Sifting news for headcounts..."):
            news = find_and_scrape_live_news(co_name, st.secrets.get("TAVILY_API_KEY"))
            
            # THE MAGIC: Extract the number and update the sidebar
            new_count = extract_headcount(news, llm)
            st.session_state.suggested_headcount = new_count
            
            prompt = f"""
            Regulatory Specialist (March 15, 2026). News: {news}. 
            Pitch: Mention {new_count} affected people found in the news. 
            Cite the ${EconomicImpact.calculate_liability(new_count)['total']:,} total debt.
            """
            report = llm.invoke(prompt).content
            st.session_state.report = report
            st.markdown(f"**Sifted Result:** AI identified **{new_count}** affected individuals in the news.")
            st.markdown("---")
            st.markdown(report)
            st.rerun() # Refresh to show new sidebar math

if "report" in st.session_state:
    st.download_button("📩 Download PDF", create_pdf(st.session_state.report), file_name="Audit.pdf")
