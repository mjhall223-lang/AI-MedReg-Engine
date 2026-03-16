import sys
import os
import re 
import streamlit as st
from engine import get_llm, scout_organization, SpecialistMath, create_pdf

# Path fix for deployment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="ReadyAudit: Specialist Hub", layout="wide")

# Persistent State Management
if "count" not in st.session_state: st.session_state.count = 10
if "report" not in st.session_state: st.session_state.report = ""
if "hole" not in st.session_state: st.session_state.hole = "Governance Gap"

llm = get_llm(st.secrets)

tab1, tab2 = st.tabs(["📁 Deep Audit", "🤖 Autonomous Hunter"])

with tab2:
    org_name = st.text_input("Enter Lead (e.g., 'Neuralink', 'Block')")
    if st.button("🔍 Scout & Sift"):
        # CLEAN SLATE: Prevents cross-pollination of reports
        st.session_state.report = "" 
        
        with st.status(f"Hunting {org_name} 'Beast' triggers..."):
            news, analysis = scout_organization(org_name, llm)
            
            try:
                parts = analysis.split("|")
                industry = parts[0].strip()
                # Extracts the '21' or '4,237'
                count = int(re.sub(r"\D", "", parts[1])) 
                hole = parts[2].strip()
                
                # UPDATE STATE: This populates the sidebar metrics
                st.session_state.count = count
                st.session_state.hole = hole
                
                # PITCH DRAFTING
                pitch_prompt = f"""
                Draft a Specialist Pitch for {org_name}.
                Industry: {industry}. Count: {count}. Hole: {hole}.
                Target the $20,000 violation risk under SB 25B-004.
                Focus on 'Affirmative Defense' via NIST AI RMF.
                Deadline: June 30, 2026.
                """
                st.session_state.report = llm.invoke(pitch_prompt).content
                
                # THE SYNC FIX: Forces sidebar to refresh with new math
                st.rerun() 
                
            except Exception as e:
                st.error(f"Sift failed: {e}")

if st.session_state.report:
    st.markdown(f"### 🛡️ Specialized Pitch for {org_name}")
    st.markdown(st.session_state.report)
    pdf = create_pdf(st.session_state.report)
    st.download_button("📩 Download Pitch PDF", pdf, f"{org_name}_Pitch.pdf")

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info(f"Target Hole: {st.session_state.hole}")
    
    # Input follows the 'Scout' results but allows manual override
    st.session_state.count = st.number_input("Affected Personnel/Subjects:", value=st.session_state.count)
    
    impact = SpecialistMath.calculate(st.session_state.count)
    st.metric("Statutory Risk (SB 24-205)", f"${impact['statutory']:,}")
    st.metric("Total Governance Debt", f"${impact['total']:,}")
