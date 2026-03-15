import streamlit as st
import sys
import os
import tempfile

# Force cloud to find local engine.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine import (
    get_llm, find_and_scrape_live_news, EconomicImpact, 
    create_pdf, load_selected_docs, extract_headcount
)

st.set_page_config(page_title="ReadyAudit: Specialist Hub", layout="wide")

# Persistent State
if "headcount" not in st.session_state: st.session_state.headcount = 10
if "scout_report" not in st.session_state: st.session_state.scout_report = ""

is_cloud = st.secrets.get("GROQ_API_KEY") is not None
llm = get_llm(is_cloud, st.secrets)

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info("Today: March 15, 2026")
    
    # LOCK: This widget is now the source of truth for the app
    st.number_input("Affected Personnel:", key="headcount", step=1)
    
    impact = EconomicImpact.calculate_liability(st.session_state.headcount)
    st.metric("Statutory Risk", f"${impact['statutory']:,}")
    st.metric("Total Governance Debt", f"${impact['total']:,}")

tab1, tab2 = st.tabs(["📁 Deep Audit", "🤖 Autonomous Hunter"])

with tab2:
    co_name = st.text_input("Enter Company (e.g., 'Block', 'Synchron')")
    if st.button("🔍 Scout & Auto-Calculate"):
        with st.status("Sifting 2026 news..."):
            news = find_and_scrape_live_news(co_name)
            # OVERWRITE: The AI's finding replaces the default 10
            found_count = extract_headcount(news, llm)
            st.session_state.headcount = found_count 
            
            # Pitch Construction
            total_debt = EconomicImpact.calculate_liability(found_count)['total']
            prompt = f"March 15, 2026. Lead: {co_name}. Headcount: {found_count}. Risk: ${total_debt:,}. News: {news}. Draft Specialist Pitch citing June 30 CO deadline."
            st.session_state.scout_report = llm.invoke(prompt).content
            st.rerun() # Forces Sidebar metric to update immediately

    if st.session_state.scout_report:
        st.markdown(st.session_state.scout_report)
        st.download_button("📩 Download Pitch", create_pdf(st.session_state.scout_report), "Pitch.pdf")
