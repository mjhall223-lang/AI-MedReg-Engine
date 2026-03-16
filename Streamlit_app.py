import streamlit as st
import sys
import os

# Cloud compatibility path fix
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine import (
    get_llm, find_live_news, EconomicImpact, 
    create_pdf, extract_headcount
)

st.set_page_config(page_title="ReadyAudit: Specialist Hub", layout="wide")

# Persistent State Management
if "headcount" not in st.session_state: st.session_state.headcount = 10
if "scout_report" not in st.session_state: st.session_state.scout_report = ""

llm = get_llm(st.secrets)

tab1, tab2 = st.tabs(["📁 Deep Audit", "🤖 Autonomous Hunter"])

with tab2:
    co_name = st.text_input("Enter Company (e.g., 'Block', 'Synchron')")
    if st.button("🔍 Scout & Sift Liability"):
        with st.status("Hunting March 2026 news for 'Beast' triggers..."):
            news_text = find_live_news(co_name)
            found_count = extract_headcount(news_text, llm)
            
            # This makes the sidebar metric jump immediately
            st.session_state.headcount = found_count 
            
            total_debt = EconomicImpact.calculate_liability(found_count)['total']
            
            # THE REAL THING PROMPT: Focuses on SB 25B-004 and the $80M risk
            prompt = f"""
            Today is March 16, 2026. Lead: {co_name}. Headcount: {found_count}. Risk: ${total_debt:,}. News: {news_text}.
            Draft a liability-focused Specialist Pitch for the June 30 Colorado AI Act deadline. 
            CITE SB 25B-004 (the extension). 
            Highlight the 'Affirmative Defense' through NIST AI RMF. 
            Go straight to the financial crisis. No fluff.
            """
            st.session_state.scout_report = llm.invoke(prompt).content
            st.rerun() 

    if st.session_state.scout_report:
        st.markdown(st.session_state.scout_report)
        pdf_data = create_pdf(st.session_state.scout_report)
        st.download_button("📩 Download Pitch PDF", pdf_data, f"{co_name}_Pitch.pdf", mime="application/pdf")

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info("Today: March 16, 2026")
    
    # Input is tied to state so the 'Scout' logic can update it
    manual_count = st.number_input("Affected Personnel:", value=st.session_state.headcount, step=1)
    st.session_state.headcount = manual_count 
    
    impact = EconomicImpact.calculate_liability(st.session_state.headcount)
    st.metric("Statutory Risk (SB 24-205)", f"${impact['statutory']:,}")
    st.metric("Total Governance Debt", f"${impact['total']:,}")
