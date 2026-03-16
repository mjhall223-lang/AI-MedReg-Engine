import streamlit as st
import sys
import os

# Cloud compatibility path fix
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine import (
    get_llm, find_live_news, EconomicImpact, 
    create_pdf_bytes, extract_headcount
)

st.set_page_config(page_title="ReadyAudit: Specialist Hub", layout="wide")

# Persistent State Management
if "headcount" not in st.session_state: st.session_state.headcount = 10
if "scout_report" not in st.session_state: st.session_state.scout_report = ""

llm = get_llm(st.secrets)

tab1, tab2 = st.tabs(["📁 Deep Audit", "🤖 Autonomous Hunter"])

with tab2:
    co_name = st.text_input("Enter Company Lead (e.g., 'Block', 'Synchron')")
    if st.button("🔍 Scout & Sift Liability"):
        with st.status("Hunting March 2026 news for 'Beast' triggers..."):
            news_text = find_live_news(co_name)
            found_count = extract_headcount(news_text, llm)
            
            # THE FIX: This forces the sidebar metric to jump immediately
            st.session_state.headcount = found_count 
            
            risk_data = EconomicImpact.calculate_liability(found_count)
            
            # THE REAL THING PROMPT: Directly targets the $80M debt
            prompt = f"""
            March 16, 2026. Lead: {co_name}. Count: {found_count}. Risk: ${risk_data['statutory']:,}.
            News Summary: {news_text}.
            Target: June 30 Colorado AI Act Deadline (SB 25B-004).
            Draft a Specialist Pitch. Focus on 'Affirmative Defense' via NIST AI RMF. 
            No fluff. Go straight to the financial exposure.
            """
            st.session_state.scout_report = llm.invoke(prompt).content
            st.rerun() 

    if st.session_state.scout_report:
        st.markdown(st.session_state.scout_report)
        pdf_data = create_pdf_bytes(st.session_state.scout_report)
        st.download_button("📩 Download Pitch PDF", pdf_data, f"{co_name}_Pitch.pdf", mime="application/pdf")

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info("Today: March 16, 2026")
    
    # Linked to state so the 'Scout' logic can overwrite it
    manual_count = st.number_input("Affected Personnel:", value=st.session_state.headcount, step=1)
    st.session_state.headcount = manual_count 
    
    impact = EconomicImpact.calculate_liability(st.session_state.headcount)
    st.metric("Statutory Risk (SB 24-205)", f"${impact['statutory']:,}")
    st.metric("Total Governance Debt", f"${impact['total']:,}")
