import sys, os, re, streamlit as st
from engine import get_llm, scout_organization, SpecialistMath

# Path fix for deployment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
st.set_page_config(page_title="ReadyAudit: Specialist Hub", layout="wide")

# Persistent State: Initialize with March 2026 context
if "count" not in st.session_state: st.session_state.count = 4000
if "report" not in st.session_state: st.session_state.report = ""
if "hole" not in st.session_state: st.session_state.hole = "Human Appeal Path"

llm = get_llm(st.secrets)
tab1, tab2 = st.tabs(["📁 Deep Audit", "🤖 Autonomous Hunter"])

with tab2:
    org_name = st.text_input("Enter Lead", value="Block")
    if st.button("🔍 Scout & Sift"):
        st.session_state.report = "" 
        with st.status(f"Hunting {org_name} 2026 triggers..."):
            news, analysis = scout_organization(org_name, llm)
            try:
                parts = [p.strip() for p in analysis.split("|")]
                if len(parts) >= 3:
                    count_match = re.search(r'\d+', parts[1].replace(',', ''))
                    count = int(count_match.group()) if count_match else 4000
                    
                    # SYNC: Forces the sidebar to update immediately
                    st.session_state.count = count
                    st.session_state.hole = parts[2]
                    
                    # AUDIT PROPOSAL PROMPT
                    audit_prompt = f"Draft a HIGH-STAKES COMPLIANCE AUDIT PROPOSAL for {org_name}. Statutory Risk: ${count * 20000:,}. Hole: {parts[2]}. Deadline: June 30, 2026 (SB 25B-004)."
                    st.session_state.report = llm.invoke(audit_prompt).content
                    st.rerun() 
            except Exception: st.error("Sift failed.")

if st.session_state.report:
    st.markdown(st.session_state.report)

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info(f"Target Hole: {st.session_state.hole}")
    st.session_state.count = st.number_input("Affected Personnel/Subjects:", value=st.session_state.count)
    impact = SpecialistMath.calculate(st.session_state.count)
    st.metric("Statutory Risk (SB 24-205)", f"${impact['statutory']:,}")
    st.metric("Total Governance Debt", f"${impact['total']:,}")
    st.caption("Enforcement Cliff: June 30, 2026")
