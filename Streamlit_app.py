import streamlit as st
import re
from engine import get_llm, scout_organization, SpecialistMath

st.set_page_config(page_title="ReadyAudit: Specialist Hub", layout="wide")

# Persistent State
if "count" not in st.session_state: st.session_state.count = 10
if "report" not in st.session_state: st.session_state.report = ""
if "hole" not in st.session_state: st.session_state.hole = "General Governance"

llm = get_llm(st.secrets)

tab1, tab2 = st.tabs(["📁 Deep Audit", "🤖 Autonomous Hunter"])

with tab2:
    org_name = st.text_input("Enter Organization (e.g., 'Block', 'Neuralink')")
    if st.button("🔍 Scout & Auto-Tailor"):
        with st.status(f"Hunting {org_name} triggers..."):
            news, analysis = scout_organization(org_name, llm)
            
            # PARSING: Industry | Number | Hole
            try:
                parts = analysis.split("|")
                industry = parts[0].strip()
                count = int(re.sub(r"\D", "", parts[1]))
                hole = parts[2].strip()
            except:
                industry, count, hole = "Enterprise", 10, "Impact Assessment Missing"
            
            st.session_state.count = count
            st.session_state.hole = hole
            math = SpecialistMath.calculate(count)
            
            # TAILORED PITCH PROMPT
            pitch_prompt = f"""
            March 16, 2026. Lead: {org_name}. Count: {count}. Industry: {industry}.
            The Hole: {hole}. 
            Draft a Specialist Pitch targeting the $20,000 violation risk. 
            CITE SB 25B-004 (the extension). 
            Focus on the 'Affirmative Defense' via NIST AI RMF. 
            Highlight the June 30, 2026 deadline.
            """
            st.session_state.report = llm.invoke(pitch_prompt).content
            st.rerun() 

if st.session_state.report:
    st.markdown(f"### 🛡️ Specialized Pitch for {org_name}")
    st.markdown(st.session_state.report)

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info(f"Target Hole: {st.session_state.hole}")
    st.metric("Statutory Risk", f"${SpecialistMath.calculate(st.session_state.count)['statutory']:,}")
