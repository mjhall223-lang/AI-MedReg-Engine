import sys
import os
import streamlit as st

# PATH FIX: Ensures engine.py is found on Streamlit Cloud
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from engine import (
    get_llm, find_live_news, EconomicImpact, 
    create_pdf, extract_headcount
)

st.set_page_config(page_title="ReadyAudit: Specialist Hub", layout="wide")

# Persistent State Initialization
if "headcount" not in st.session_state: st.session_state.headcount = 10
if "scout_report" not in st.session_state: st.session_state.scout_report = ""

# Security check
if "GROQ_API_KEY" not in st.secrets:
    st.error("Missing GROQ_API_KEY in Secrets.")
    st.stop()

llm = get_llm(st.secrets)

tab1, tab2 = st.tabs(["📁 Deep Audit", "🤖 Autonomous Hunter"])

with tab2:
    co_name = st.text_input("Enter Company (e.g., 'Block', 'Synchron')")
    if st.button("🔍 Scout & Sift Liability"):
        with st.status("Hunting March 2026 news for 'Beast' triggers..."):
            news_text = find_live_news(co_name)
            found_count = extract_headcount(news_text, llm)
            
            # THE FIX: This forces the sidebar metric to jump immediately
            st.session_state.headcount = found_count 
            
            total_debt = EconomicImpact.calculate_liability(found_count)['total']
            
            # THE REAL THING PROMPT: No fluff. Directly hits the $84M debt.
            prompt = f"""
            Context: March 16, 2026. Lead: {co_name}. Headcount: {found_count}. Risk: ${total_debt:,}.
            News Summary: {news_text}.
            Target: June 30 Colorado AI Act Deadline (SB 25B-004).
            Draft a Specialist Pitch. Focus on 'Affirmative Defense' via NIST AI RMF. 
            Highlight the $20,000 per violation penalty. No fluff.
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
    
    # Linked to state so the 'Scout' logic can overwrite it
    st.session_state.headcount = st.number_input("Affected Personnel:", value=st.session_state.headcount)
    
    impact = EconomicImpact.calculate_liability(st.session_state.headcount)
    st.metric("Statutory Risk (SB 24-205)", f"${impact['statutory']:,}")
    st.metric("Total Governance Debt", f"${impact['total']:,}")
