import sys
import os
import re  # FIX: Crucial for Line 34 to work
import streamlit as st

# FORCE PATH FIX: For Streamlit Cloud compatibility
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from engine import (
    get_llm, scout_organization, EconomicImpact, 
    create_pdf
)

st.set_page_config(page_title="ReadyAudit: Specialist Hub", layout="wide")

# Persistent State Management
if "count" not in st.session_state: st.session_state.count = 10
if "scout_report" not in st.session_state: st.session_state.scout_report = ""
if "trigger" not in st.session_state: st.session_state.trigger = "General Governance"

llm = get_llm(st.secrets)

tab1, tab2 = st.tabs(["📁 Deep Audit", "🤖 Autonomous Hunter"])

with tab2:
    org_name = st.text_input("Enter Organization (e.g., 'Block', 'Neuralink')")
    if st.button("🔍 Scout & Auto-Tailor"):
        with st.status(f"Hunting {org_name} triggers..."):
            news, analysis = scout_organization(org_name, llm)
            
            # SAFE PARSING: Industry | Number | Trigger
            try:
                parts = analysis.split("|")
                industry = parts[0].strip()
                count_str = re.sub(r"\D", "", parts[1])
                count = int(count_str) if count_str else 10
                trigger = parts[2].strip()
            except:
                industry, count, trigger = "Enterprise", 10, "General Governance"
            
            # Injecting findings into state
            st.session_state.count = count
            st.session_state.trigger = trigger
            math = EconomicImpact.calculate(count)
            
            # THE TAILORED PITCH PROMPT
            pitch_prompt = f"""
            Context: March 16, 2026. Lead: {org_name}. Count: {count}. Industry: {industry}.
            Trigger: {trigger}. News: {news[:500]}.
            Deadline: June 30 Colorado AI Act (SB 24-205 & SB 25B-004).
            Draft a Specialist Pitch targeting the $20,000 violation risk. 
            Highlight the 'Affirmative Defense' Safe Harbor via NIST AI RMF. 
            If MedTech: Focus on 'Patient Safety & Clinical Trial Integrity'.
            If Fintech: Focus on 'Algorithmic Discrimination in Hiring/Lending'.
            No fluff. Go straight to the financial crisis.
            """
            st.session_state.scout_report = llm.invoke(pitch_prompt).content
            st.rerun() 

    if st.session_state.scout_report:
        st.markdown(st.session_state.scout_report)
        pdf = create_pdf(st.session_state.scout_report)
        st.download_button("📩 Download Pitch PDF", pdf, f"{org_name}_Pitch.pdf")

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info(f"Detected Trigger: {st.session_state.trigger}")
    
    # Linked to state so the 'Scout' results overwrite the manual input
    st.session_state.count = st.number_input("Affected Subjects/Staff:", value=st.session_state.count)
    
    impact = EconomicImpact.calculate(st.session_state.count)
    st.metric("Statutory Risk (SB 24-205)", f"${impact['statutory']:,}")
    st.metric("Total Governance Debt", f"${impact['total']:,}")
