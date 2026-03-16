import streamlit as st
from engine import get_llm, find_live_news, EconomicImpact, create_pdf_bytes, extract_headcount

st.set_page_config(page_title="ReadyAudit: Specialist Hub", layout="wide")

# Persistent State Management
if "headcount" not in st.session_state: st.session_state.headcount = 10
if "scout_report" not in st.session_state: st.session_state.scout_report = ""

llm = get_llm(st.secrets)

tab1, tab2 = st.tabs(["📁 Deep Audit", "🤖 Autonomous Hunter"])

with tab2:
    co_name = st.text_input("Enter Lead (e.g., 'Block', 'Synchron')")
    if st.button("🔍 Scout & Sift Liability"):
        with st.status("Sifting March 2026 news for 'Beast' triggers..."):
            news = find_live_news(co_name)
            found_count = extract_headcount(news, llm)
            
            # This ensures the sidebar metric jumps the moment the news is found
            st.session_state.headcount = found_count 
            
            risk_data = EconomicImpact.calculate_liability(found_count)
            # High-stakes prompt targeting the June 30 deadline
            prompt = f"""
            March 15, 2026 context. Lead: {co_name}. Count: {found_count}. Statutory Risk: ${risk_data['statutory']:,}. 
            News found: {news}.
            Draft a Specialist Pitch for the June 30 CO deadline. 
            Focus on: 'Affirmative Defense' via SB 25B-004 and NIST AI RMF alignment. 
            No fluff. Go straight to the financial crisis.
            """
            st.session_state.scout_report = llm.invoke(prompt).content
            st.rerun() 

    if st.session_state.scout_report:
        st.markdown(st.session_state.scout_report)
        pdf_data = create_pdf_bytes(st.session_state.scout_report)
        st.download_button("📩 Download Pitch PDF", pdf_data, f"{co_name}_Pitch.pdf", mime="application/pdf")

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info("Today: March 15, 2026")
    
    # We use st.session_state.headcount as the value to allow the Scout to update it
    st.session_state.headcount = st.number_input("Affected Personnel:", value=st.session_state.headcount)
    
    impact = EconomicImpact.calculate_liability(st.session_state.headcount)
    st.metric("Statutory Risk (SB 24-205)", f"${impact['statutory']:,}")
    st.metric("Total Governance Debt", f"${impact['total']:,}")
