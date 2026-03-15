import streamlit as st
import sys
import os

# Ensure the app finds your engine.py in the root folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine import (
    get_llm, find_and_scrape_live_news, EconomicImpact, 
    create_pdf, extract_headcount
)

st.set_page_config(page_title="ReadyAudit: Specialist Hub", layout="wide")

# 1. INITIALIZE: Setup state without locking widgets
if "headcount" not in st.session_state: st.session_state.headcount = 10
if "scout_report" not in st.session_state: st.session_state.scout_report = ""

is_cloud = st.secrets.get("GROQ_API_KEY") is not None
llm = get_llm(is_cloud, st.secrets)

# 2. LOGIC TABS (Renders first to process search)
tab1, tab2 = st.tabs(["📁 Deep Audit", "🤖 Autonomous Hunter"])

with tab2:
    co_name = st.text_input("Enter Company (e.g., 'Block', 'Synchron')")
    if st.button("🔍 Scout & Auto-Calculate"):
        with st.status("Sifting 2026 news..."):
            news = find_and_scrape_live_news(co_name)
            found_count = extract_headcount(news, llm)
            
            # FORCE UPDATE: Overwrite the 10 with the real number (e.g., 4000)
            st.session_state.headcount = found_count 
            
            risk = EconomicImpact.calculate_liability(found_count)['total']
            prompt = f"March 15, 2026. Lead: {co_name}. Count: {found_count}. Risk: ${risk:,}. Draft Specialist Pitch for June 30 CO deadline."
            st.session_state.scout_report = llm.invoke(prompt).content
            st.rerun() # Forces sidebar metrics to update immediately

    if st.session_state.scout_report:
        st.markdown(st.session_state.scout_report)
        # BYTES FIX: Ensuring data is compatible for download
        pdf_bytes = create_pdf(st.session_state.scout_report)
        st.download_button("📩 Download Pitch", pdf_bytes, f"{co_name}_Pitch.pdf", mime="application/pdf")

# 3. SIDEBAR: Renders last so it sees the updated state
with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info("Today: March 15, 2026")
    
    # Use 'value' instead of 'key' to avoid the StreamlitAPIException crash
    new_val = st.number_input("Affected Personnel:", value=st.session_state.headcount, step=1)
    st.session_state.headcount = new_val 
    
    impact = EconomicImpact.calculate_liability(st.session_state.headcount)
    st.metric("Statutory Risk", f"${impact['statutory']:,}")
    st.metric("Total Governance Debt", f"${impact['total']:,}")
