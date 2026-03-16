import sys
import os
import re 
import streamlit as st
from engine import get_llm, scout_organization, SpecialistMath, create_pdf

# PATH FIX for Streamlit Cloud
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="ReadyAudit: Specialist Hub", layout="wide")

# Persistent State Management
if "count" not in st.session_state: st.session_state.count = 10
if "report" not in st.session_state: st.session_state.report = ""
if "hole" not in st.session_state: st.session_state.hole = "Governance Gap"

llm = get_llm(st.secrets)

tab1, tab2 = st.tabs(["📁 Deep Audit", "🤖 Autonomous Hunter"])

with tab2:
    org_name = st.text_input("Enter Organization (e.g., 'Block', 'Neuralink')")
    if st.button("🔍 Scout & Auto-Tailor"):
        with st.status(f"Hunting {org_name} 'Beast' triggers..."):
            news, analysis = scout_organization(org_name, llm)
            
            try:
                parts = analysis.split("|")
                industry = parts[0].strip()
                count = int(re.sub(r"\D", "", parts[1]))
                hole = parts[2].strip()
            except:
                industry, count, hole = "Enterprise", 10, "Impact Assessment Missing"
            
            st.session_state.count = count
            st.session_state.hole = hole
            
            # THE KILL SHOT PROMPT: Targets the specific Hole
            pitch_prompt = f"""
            March 16, 2026. Lead: {org_name}. Count: {count}. Industry: {industry}.
            The Hole: {hole}. 
            Draft a Specialist Pitch targeting the $20,000 violation risk under SB 25B-004.
            Highlight the 'Affirmative Defense' Safe Harbor via NIST AI RMF. 
            Crucial: Cite the June 30, 2026 Hard Start. 
            For Block: Focus on 'Missing Human Appeal Path' for the 4,000 workers.
            For Neuralink: Focus on 'Substantial Modification' for the 21-person trial.
            """
            st.session_state.report = llm.invoke(pitch_prompt).content
            st.rerun() 

if st.session_state.report:
    st.markdown(f"### 🛡️ Specialized Pitch for {org_name}")
    st.markdown(st.session_state.report)
    pdf_data = create_pdf(st.session_state.report)
    st.download_button("📩 Download Pitch PDF", pdf_data, f"{org_name}_Pitch.pdf")

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info(f"Target Hole: {st.session_state.hole}")
    st.session_state.count = st.number_input("Affected Personnel/Subjects:", value=st.session_state.count)
    
    impact = SpecialistMath.calculate(st.session_state.count)
    st.metric("Statutory Risk (SB 24-205)", f"${impact['statutory']:,}")
    st.metric("Total Governance Debt", f"${impact['total']:,}")
