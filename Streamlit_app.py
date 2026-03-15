import streamlit as st
import sys
import os
import tempfile

# FORCE CLOUD COMPATIBILITY: Look in the root folder for engine.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine import (
    get_llm, find_and_scrape_live_news, EconomicImpact, 
    create_pdf, extract_headcount
)

st.set_page_config(page_title="ReadyAudit: Specialist Hub", layout="wide")

# Persistent State Initialization
if "headcount" not in st.session_state: st.session_state.headcount = 10
if "scout_report" not in st.session_state: st.session_state.scout_report = ""

is_cloud = st.secrets.get("GROQ_API_KEY") is not None
llm = get_llm(is_cloud, st.secrets)

# RENDER SIDEBAR (Updates after logic rerun)
with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info("Today: March 15, 2026")
    
    # Use 'value' instead of 'key' to allow manual overrides during Scout
    temp_count = st.number_input("Affected Personnel:", value=st.session_state.headcount, step=1)
    st.session_state.headcount = temp_count 
    
    impact = EconomicImpact.calculate_liability(st.session_state.headcount)
    st.metric("Statutory Risk", f"${impact['statutory']:,}", delta="Per CO SB 24-205")
    st.metric("Total Governance Debt", f"${impact['total']:,}")

tab1, tab2 = st.tabs(["📁 Deep Audit", "🤖 Autonomous Hunter"])

with tab2:
    co_name = st.text_input("Enter Company (e.g., 'Block', 'Synchron')")
    if st.button("🔍 Scout & Auto-Calculate"):
        with st.status("Sifting 2026 news..."):
            news_text = find_and_scrape_live_news(co_name)
            found_count = extract_headcount(news_text, llm)
            
            # OVERWRITE STATE: Sidebar picks this up on rerun
            st.session_state.headcount = found_count 
            
            total_debt = EconomicImpact.calculate_liability(found_count)['total']
            # Specialist Pitch Framing
            prompt = f"""
            March 15, 2026. Lead: {co_name}. Count: {found_count}. Risk: ${total_debt:,}. News: {news_text}. 
            Draft Specialist Pitch citing June 30 CO deadline (SB 25B-004 extension). 
            Focus on Affirmative Defense through NIST RMF.
            """
            st.session_state.scout_report = llm.invoke(prompt).content
            st.rerun() 

    if st.session_state.scout_report:
        st.markdown(st.session_state.scout_report)
        # BYTES FIX applied here
        pdf_data = create_pdf(st.session_state.scout_report)
        st.download_button("📩 Download Pitch PDF", pdf_data, f"{co_name}_Pitch.pdf", mime="application/pdf")
        
