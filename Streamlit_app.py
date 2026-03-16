import streamlit as st
import re
from engine import get_llm, scout_organization, perform_remediation

st.set_page_config(page_title="ReadyAudit: 2026 Specialist Hub", layout="wide")

# Persistent State
if "count" not in st.session_state: st.session_state.count = 4000
if "hole" not in st.session_state: st.session_state.hole = "Human Appeal Path"
if "pitch" not in st.session_state: st.session_state.pitch = ""
if "audit_report" not in st.session_state: st.session_state.audit_report = ""

llm = get_llm(st.secrets)

tab1, tab2 = st.tabs(["📁 Remediation Audit", "🤖 Scout & Pitch"])

with tab2:
    st.header("The Hunter: Scout & Pitch")
    org_name = st.text_input("Enter Lead Name", value="Block")
    if st.button("🔍 Scout 2026 Triggers"):
        with st.status("Sifting Regulations/Regulations/..."):
            analysis = scout_organization(org_name, llm)
            parts = [p.strip() for p in analysis.split("|")]
            if len(parts) >= 3:
                st.session_state.count = int(re.sub(r"\D", "", parts[1]))
                st.session_state.hole = parts[2]
                st.session_state.pitch = llm.invoke(f"Draft a board-level proposal for {org_name} to fill the ${st.session_state.count*20000:,} hole via a {parts[2]}.").content
                st.rerun()
    st.markdown(st.session_state.pitch)

with tab1:
    st.header("The Mechanic: Remediation Audit")
    if not st.session_state.pitch:
        st.warning("Run a Scout first to identify the statutory risk.")
    else:
        if st.button("🛠️ Generate Affirmative Defense Artifacts"):
            with st.spinner("Building NIST-aligned protocols..."):
                st.session_state.audit_report = perform_remediation(org_name, st.session_state.hole, st.session_state.count, llm)
        st.markdown(st.session_state.audit_report)

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info(f"Target Hole: {st.session_state.hole}")
    st.session_state.count = st.number_input("Affected Count:", value=st.session_state.count)
    statutory = st.session_state.count * 20000
    st.metric("Statutory Risk", f"${statutory:,}")
    st.metric("Total Governance Debt", f"${round(statutory * 1.25, 2):,}")
    st.caption("Enforcement Cliff: June 30, 2026")
