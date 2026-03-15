import streamlit as st
import os
import sys
import tempfile

sys.path.append(os.path.dirname(__file__))
from engine import (
    get_llm, find_and_scrape_live_news, EconomicImpact, 
    create_pdf, load_selected_docs, extract_headcount
)

st.set_page_config(page_title="ReadyAudit: Specialist Hub", layout="wide")

# Session State Pantry
if "audit_report" not in st.session_state: st.session_state.audit_report = ""
if "scout_report" not in st.session_state: st.session_state.scout_report = ""
if "scout_news" not in st.session_state: st.session_state.scout_news = ""
if "headcount" not in st.session_state: st.session_state.headcount = 10

is_cloud = st.secrets.get("GROQ_API_KEY") is not None
llm = get_llm(is_cloud, st.secrets)

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info("Today: March 15, 2026")
    
    if st.button("♻️ Reset App State"):
        st.session_state.audit_report = ""
        st.session_state.scout_report = ""
        st.session_state.headcount = 10
        st.rerun()

    val = st.number_input("Affected Personnel:", value=st.session_state.headcount, step=1)
    st.session_state.headcount = val 
    
    impact = EconomicImpact.calculate_liability(st.session_state.headcount)
    st.metric("Statutory Risk", f"${impact['statutory']:,}", delta="Per CO SB 24-205")
    st.metric("Total Governance Debt", f"${impact['total']:,}")

    st.markdown("---")
    if not os.path.exists("Regulations"): os.makedirs("Regulations")
    all_pdfs = [f for f in os.listdir("Regulations") if f.endswith(".pdf")]
    selected_files = [f for f in all_pdfs if st.checkbox(f"📄 {f}", value=True)]

tab1, tab2 = st.tabs(["📁 Deep Audit", "🤖 Autonomous Hunter"])

with tab1:
    uploaded = st.file_uploader("Upload Policy", type="pdf")
    if st.button("🚀 Run Analysis"):
        with st.status("Analyzing..."):
            db = load_selected_docs(selected_files)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.getvalue())
                t_path = tmp.name
            from langchain_community.document_loaders import PyPDFLoader
            evidence = "\n".join([p.page_content for p in PyPDFLoader(t_path).load()[:5]])
            regs = "\n".join([d.page_content for d in db.similarity_search(evidence, k=3)]) if db else ""
            st.session_state.audit_report = llm.invoke(f"Audit Policy: {evidence} against Laws: {regs}").content
            os.remove(t_path)
    
    if st.session_state.audit_report:
        st.markdown(st.session_state.audit_report)
        st.download_button("📩 Download Audit", create_pdf(st.session_state.audit_report), "Audit.pdf")

with tab2:
    co_name = st.text_input("Enter Company Name")
    if st.button("🔍 Scout & Auto-Calculate"):
        with st.status("Sifting 2026 news..."):
            news = find_and_scrape_live_news(co_name)
            st.session_state.scout_news = news
            st.session_state.headcount = extract_headcount(news, llm)
            
            prompt = f"""
            Regulatory Specialist (March 15, 2026). News: {news}. 
            Draft a pitch for {co_name} based on {st.session_state.headcount} affected people. 
            Cite the ${EconomicImpact.calculate_liability(st.session_state.headcount)['total']:,} total debt.
            Mention the June 30, 2026 Colorado AI Act deadline.
            """
            st.session_state.scout_report = llm.invoke(prompt).content
            st.rerun()

    if st.session_state.scout_report:
        st.subheader(f"Pitch for {co_name}")
        st.markdown(st.session_state.scout_report)
        with st.expander("📊 Sourced Search Results"):
            st.write(st.session_state.scout_news)
        st.download_button("📩 Download Pitch", create_pdf(st.session_state.scout_report), "Pitch.pdf")
