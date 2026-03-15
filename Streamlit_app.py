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

# Initialize Session State
if "headcount" not in st.session_state: st.session_state.headcount = 10
if "scout_report" not in st.session_state: st.session_state.scout_report = ""

is_cloud = st.secrets.get("GROQ_API_KEY") is not None
llm = get_llm(is_cloud, st.secrets)

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info("Today: March 15, 2026")
    
    # BINDING: The widget is now locked to the session memory
    st.number_input("Affected Personnel:", key="headcount", step=1)
    
    # Calculate using the LIVE state value
    impact = EconomicImpact.calculate_liability(st.session_state.headcount)
    st.metric("Statutory Risk", f"${impact['statutory']:,}")
    st.metric("Total Governance Debt", f"${impact['total']:,}")

tab1, tab2 = st.tabs(["📁 Deep Audit", "🤖 Autonomous Hunter"])

with tab2:
    co_name = st.text_input("Enter Company (e.g., 'Block', 'Synchron')")
    if st.button("🔍 Scout & Auto-Calculate"):
        with st.status("Sifting 2026 news..."):
            news = find_and_scrape_live_news(co_name)
            
            # THE OVERRIDE: Update the state so the sidebar and pitch match
            found_count = extract_headcount(news, llm)
            st.session_state.headcount = found_count 
            
            # Specialist Prompt to stop hallucinations
            total_debt = EconomicImpact.calculate_liability(found_count)['total']
            prompt = f"""
            March 15, 2026. News: {news}. Target: {co_name}.
            Specialist Task: Draft a liability-focused pitch for {found_count} affected people. 
            Highlight the ${total_debt:,} statutory debt under CO SB 24-205.
            Deadline: June 30, 2026. Pitch an Affirmative Defense audit.
            """
            st.session_state.scout_report = llm.invoke(prompt).content
            st.rerun() # Refresh to update the Sidebar math instantly

    if st.session_state.scout_report:
        st.markdown(st.session_state.scout_report)
        st.download_button("📩 Download Pitch", create_pdf(st.session_state.scout_report), "Pitch.pdf")
