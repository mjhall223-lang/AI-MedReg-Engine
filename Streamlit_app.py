import sys
import os
import re  # FIX: Crucial for line 24 to work
import streamlit as st

# FORCE PATH FIX
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine import (
    get_llm, scout_organization, EconomicImpact, 
    create_pdf
)

st.set_page_config(page_title="ReadyAudit: Specialist Hub", layout="wide")

# Persistent State
if "count" not in st.session_state: st.session_state.count = 10
if "report" not in st.session_state: st.session_state.report = ""
if "trigger" not in st.session_state: st.session_state.trigger = "General Risk"

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
                industry, count, trigger = "Enterprise", 10, "General Risk"
            
            st.session_state.count = count
            st.session_state.trigger = trigger
            math = EconomicImpact.calculate(count)
            
            # THE TAILORED PITCH PROMPT
            pitch_prompt = f"""
            March 16, 2026. Lead: {org_name}. Count: {count}. Industry: {industry}.
            Trigger: {trigger}. News Summary: {news[:500]}.
            Target: June 30 Colorado AI Act (SB 24-205 & SB 25B-004).
            Draft a Specialist Pitch. 
            If MedTech: Focus on 'Patient Safety & Clinical Integrity'.
            If Fintech: Focus on 'Algorithmic Discrimination & Hiring'.
            Highlight the $20,000 penalty and 'Affirmative Defense' Safe Harbor via NIST AI RMF.
            """
            st.session_state.scout_report = llm.invoke(pitch_prompt).content
            st.rerun() 

    if "scout_report" in st.session_state and st.session_state.scout_report:
        st.markdown(st.session_state.scout_report)
        pdf = create_pdf(st.session_state.scout_report)
        st.download_button("📩 Download Pitch PDF", pdf, f"{org_name}_Pitch.pdf")

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info(f"Trigger: {st.session_state.trigger}")
    
    # Value tied to session state so 'Scout' can overwrite the input
    st.session_state.count = st.number_input("Affected Personnel/Subjects:", value=st.session_state.count)
    
    impact = EconomicImpact.calculate(st.session_state.count)
    st.metric("Statutory Risk (SB 24-205)", f"${impact['statutory']:,}")
    st.metric("Total Governance Debt", f"${impact['total']:,}")
