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

# 1. INITIALIZE: Set the default if it doesn't exist
if "headcount" not in st.session_state:
    st.session_state.headcount = 10
if "scout_report" not in st.session_state:
    st.session_state.scout_report = ""

is_cloud = st.secrets.get("GROQ_API_KEY") is not None
llm = get_llm(is_cloud, st.secrets)

# 2. THE TABS (Logic happens here first)
tab1, tab2 = st.tabs(["📁 Deep Audit", "🤖 Autonomous Hunter"])

with tab2:
    co_name = st.text_input("Enter Company (e.g., 'Block', 'Synchron')")
    if st.button("🔍 Scout & Auto-Calculate"):
        with st.status("Sifting 2026 news..."):
            news = find_and_scrape_live_news(co_name)
            found_count = extract_headcount(news, llm)
            
            # SUCCESS: We update the state HERE. 
            # Because the Sidebar hasn't "rendered" the widget yet in the rerun, 
            # this is allowed.
            st.session_state.headcount = found_count 
            
            total_debt = EconomicImpact.calculate_liability(found_count)['total']
            prompt = f"March 15, 2026. Lead: {co_name}. Headcount: {found_count}. Risk: ${total_debt:,}. News: {news}. Draft Specialist Pitch."
            st.session_state.scout_report = llm.invoke(prompt).content
            st.rerun() 

    if st.session_state.scout_report:
        st.markdown(st.session_state.scout_report)

# 3. THE SIDEBAR (Renders AFTER the logic updates the state)
with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info("Today: March 15, 2026")
    
    # We use st.session_state.headcount as the value, but NO KEY.
    # This prevents the "Locked Key" error while still letting the code update it.
    new_val = st.number_input("Affected Personnel:", value=st.session_state.headcount, step=1)
    st.session_state.headcount = new_val # Manual update
    
    impact = EconomicImpact.calculate_liability(st.session_state.headcount)
    st.metric("Statutory Risk", f"${impact['statutory']:,}")
    st.metric("Total Governance Debt", f"${impact['total']:,}")

# ... (Rest of Tab 1 code)
