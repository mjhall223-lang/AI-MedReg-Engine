import streamlit as st
from engine import get_llm, find_and_scrape_live_news, EconomicImpact, create_pdf, extract_headcount

# 1. State Sync
if "headcount" not in st.session_state: st.session_state.headcount = 10

# 2. Sidebar Calculator
with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    # This value now follows the 'Hunter' logic
    val = st.number_input("Affected Personnel:", value=st.session_state.headcount)
    st.session_state.headcount = val
    impact = EconomicImpact.calculate_liability(val)
    st.metric("Statutory Risk", f"${impact['statutory']:,}")

# 3. The Hunter Tab
with st.expander("🤖 Autonomous Hunter", expanded=True):
    co_name = st.text_input("Enter Company (e.g., 'Block')")
    if st.button("🔍 Scout & Auto-Calculate"):
        news = find_and_scrape_live_news(co_name)
        found_count = extract_headcount(news, llm)
        st.session_state.headcount = found_count # THIS makes the calculator jump
        st.rerun()
