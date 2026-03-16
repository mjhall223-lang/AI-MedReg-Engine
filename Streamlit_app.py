import sys
import os
import streamlit as st

# FORCE PATH FIX: Tells Streamlit exactly where engine.py is
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Now try the import
try:
    from engine import (
        get_llm, find_live_news, LiabilityEngine, 
        create_pdf_bytes, extract_headcount
    )
except ImportError as e:
    st.error(f"Critical System Error: Could not load 'engine.py'. Details: {e}")
    st.stop()

st.set_page_config(page_title="ReadyAudit: Specialist Hub", layout="wide")

# Persistent State
if "headcount" not in st.session_state: st.session_state.headcount = 10
if "report" not in st.session_state: st.session_state.report = ""

# Security check for Cloud
if "GROQ_API_KEY" not in st.secrets:
    st.error("Missing GROQ_API_KEY in Streamlit Secrets.")
    st.stop()

llm = get_llm(st.secrets)

tab1, tab2 = st.tabs(["📁 Deep Audit", "🤖 Autonomous Hunter"])

with tab2:
    co_name = st.text_input("Enter Company Lead (e.g., 'Block', 'Synchron')")
    if st.button("🔍 Scout & Sift Liability"):
        with st.status("Hunting March 2026 'Beast' triggers..."):
            news = find_live_news(co_name)
            count = extract_headcount(news, llm)
            
            st.session_state.headcount = count 
            math = LiabilityEngine.run_math(count)
            
            # THE REAL THING PROMPT: Directly targets the $80M debt
            prompt = f"""
            Context: March 16, 2026. Lead: {co_name}. Headcount: {count}. Risk: ${math['statutory']:,}.
            News Summary: {news}.
            Target: June 30 Colorado AI Act Deadline (SB 25B-004).
            Draft a Specialist Pitch. Focus on 'Affirmative Defense' via NIST AI RMF. 
            No fluff. Go straight to the financial exposure.
            """
            st.session_state.report = llm.invoke(prompt).content
            st.rerun() 

    if st.session_state.report:
        st.markdown(st.session_state.report)
        pdf = create_pdf_bytes(st.session_state.report)
        st.download_button("📩 Download Pitch PDF", pdf, f"{co_name}_Pitch.pdf", mime="application/pdf")

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info("Today: March 16, 2026")
    st.session_state.headcount = st.number_input("Affected Personnel:", value=st.session_state.headcount)
    
    impact = LiabilityEngine.run_math(st.session_state.headcount)
    st.metric("Statutory Risk (SB 24-205)", f"${impact['statutory']:,}")
    st.metric("Total Governance Debt", f"${impact['total']:,}")
