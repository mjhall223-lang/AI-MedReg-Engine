import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine import (
    get_llm, find_and_scrape_live_news, EconomicImpact, 
    create_pdf, load_selected_docs, extract_headcount
)

st.set_page_config(page_title="ReadyAudit: Specialist Hub", layout="wide")

# Persistent State Initialization
if "headcount" not in st.session_state:
    st.session_state.headcount = 10
if "scout_report" not in st.session_state:
    st.session_state.scout_report = ""

is_cloud = st.secrets.get("GROQ_API_KEY") is not None
llm = get_llm(is_cloud, st.secrets)

# SIDEBAR: Renders at the start using current state
with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info("Today: March 15, 2026")
    
    # Use 'value' instead of 'key' to avoid locking the session_state
    # This allows the scout button to manually overwrite it later
    selected_count = st.number_input("Affected Personnel:", value=st.session_state.headcount, step=1)
    st.session_state.headcount = selected_count
    
    impact = EconomicImpact.calculate_liability(st.session_state.headcount)
    st.metric("Statutory Risk", f"${impact['statutory']:,}")
    st.metric("Total Governance Debt", f"${impact['total']:,}")

tab1, tab2 = st.tabs(["📁 Deep Audit", "🤖 Autonomous Hunter"])

with tab2:
    co_name = st.text_input("Enter Company (e.g., 'Block', 'Synchron')")
    if st.button("🔍 Scout & Auto-Calculate"):
        with st.status("Sifting 2026 news..."):
            news = find_and_scrape_live_news(co_name)
            found_count = extract_headcount(news, llm)
            
            # THE FIX: This updates the state BEFORE the rerun
            st.session_state.headcount = found_count 
            
            total_risk = EconomicImpact.calculate_liability(found_count)['total']
            prompt = f"Lead: {co_name}. Headcount: {found_count}. Risk: ${total_risk:,}. Draft a liability pitch for June 30 deadline."
            st.session_state.scout_report = llm.invoke(prompt).content
            st.rerun() # Refresh sidebar and pitch instantly

    if st.session_state.scout_report:
        st.markdown(st.session_state.scout_report)
        # Bytes fix applied here
        st.download_button("📩 Download PDF", create_pdf(st.session_state.scout_report), f"{co_name}_Pitch.pdf")
