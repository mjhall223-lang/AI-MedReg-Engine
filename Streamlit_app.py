import streamlit as st
import sys
import os

# PATH FIX for Streamlit Cloud
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
    co_name = st.text_input("Enter Lead (e.g., 'Block', 'Synchron')")
    if st.button("🔍 Scout & Sift Liability"):
        with st.status("Hunting March 2026 news for 'Beast' triggers..."):
            news_text = find_live_news(co_name)
            found_count = extract_headcount(news_text, llm)
            
            # THE FIX: Force session state update before anything else
            st.session_state.headcount = found_count 
            
            total_debt = EconomicImpact.calculate_liability(found_count)['total']
            
            # THE KILL SHOT PROMPT: No fluff. Directly hits the $84M debt.
            prompt = f"""
            March 16, 2026. Lead: {co_name}. Headcount: {found_count}. Risk: ${total_debt:,}.
            Sifted News: {news_text}.
            Deadline: June 30 Colorado AI Act (SB 24-205).
            Draft a Specialist Pitch. CITE SB 25B-004 (the extension). 
            Focus on the 'Affirmative Defense' via NIST AI RMF. 
            Highlight that failure to conduct a written impact assessment by June 30 makes the $20,000 penalty uncurable.
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
    
    # Linked directly to state to allow the 'Scout' to overwrite the input
    st.session_state.headcount = st.number_input("Affected Personnel:", value=st.session_state.headcount, step=1)
    
    impact = EconomicImpact.calculate_liability(st.session_state.headcount)
    st.metric("Statutory Risk (SB 24-205)", f"${impact['statutory']:,}")
    st.metric("Total Governance Debt", f"${impact['total']:,}")
