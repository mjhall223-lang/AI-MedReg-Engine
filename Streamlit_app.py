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
            
            # PARSING: Industry | Number | Hole
            try:
                parts = analysis.split("|")
                industry = parts[0].strip()
                count = int(re.sub(r"\D", "", parts[1]))
                hole = parts[2].strip()
            except:
                industry, count, hole = "Enterprise", 10, "Written Impact Assessment Missing"
            
            st.session_state.count = count
            st.session_state.hole = hole
            
            # THE TAILORED PITCH PROMPT
            pitch_prompt = f"""
            March 16, 2026. Lead: {org_name}. Count: {count}. Industry: {industry}.
            The Hole: {hole}. News Context: {news[:500]}.
            Draft a Specialist Pitch targeting the $20,000 violation risk.
            CITE SB 25B-004 (the extension). 
            Focus on the 'Affirmative Defense' Safe Harbor via NIST AI RMF. 
            Highlight the June 30, 2026 'Hard Start' deadline.
            If MedTech: Focus on 'Substantial Modification Audit' for the human subjects.
            If Fintech: Focus on 'Human Appeal/Appeal Path' for the affected staff.
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
    
    # Value tied to state so 'Scout' results overwrite manual input
    st.session_state.count = st.number_input("Affected Subjects/Personnel:", value=st.session_state.count)
    
    impact = SpecialistMath.calculate(st.session_state.count)
    st.metric("Statutory Risk (SB 24-205)", f"${impact['statutory']:,}")
    st.metric("Total Governance Debt", f"${impact['total']:,}")
