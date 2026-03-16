import streamlit as st
import re
from engine import get_llm, scout_organization, perform_remediation_audit

st.set_page_config(page_title="ReadyAudit", layout="wide")

# Persistent State Management
if "count" not in st.session_state: st.session_state.count = 4000
if "hole" not in st.session_state: st.session_state.hole = "Human Appeal Path"
if "pitch" not in st.session_state: st.session_state.pitch = ""
if "audit_report" not in st.session_state: st.session_state.audit_report = ""

llm = get_llm(st.secrets)

# THE TWO-TAB ENGINE
tab1, tab2 = st.tabs(["📁 Remediation Audit", "🤖 Scout & Pitch"])

with tab2:
    st.header("The Hunter: Scout & Pitch")
    org_name = st.text_input("Enter Lead Name", value="Block")
    if st.button("🔍 Scout Live Wires"):
        with st.status("Sifting 2026 Data..."):
            analysis = scout_organization(org_name, llm)
            parts = [p.strip() for p in analysis.split("|")]
            if len(parts) >= 3:
                st.session_state.count = int(re.sub(r"\D", "", parts[1]))
                st.session_state.hole = parts[2]
                st.session_state.pitch = llm.invoke(f"Draft a board-level pitch for {org_name} for the {parts[2]} hole.").content
                st.rerun()
    st.markdown(st.session_state.pitch)

with tab1:
    st.header("The Mechanic: Remediation Audit")
    if st.session_state.pitch == "":
        st.warning("Run a Scout first to identify a target.")
    else:
        st.info(f"Auditing {st.session_state.hole} for {org_name}...")
        if st.button("🛠️ Generate Remediation Plan"):
            with st.spinner("Building Affirmative Defense artifacts..."):
                st.session_state.audit_report = perform_remediation_audit(org_name, st.session_state.hole, llm)
        st.markdown(st.session_state.audit_report)

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info(f"Focus: {st.session_state.hole}")
    st.session_state.count = st.number_input("Count:", value=st.session_state.count)
    statutory = st.session_state.count * 20000
    st.metric("Statutory Risk", f"${statutory:,}")
    st.caption("Enforcement Cliff: June 30, 2026")
