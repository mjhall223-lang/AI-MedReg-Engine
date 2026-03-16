import streamlit as st
import re
from engine import get_llm, scout_organization

st.set_page_config(page_title="ReadyAudit", layout="wide")

if "count" not in st.session_state: st.session_state.count = 4000
if "hole" not in st.session_state: st.session_state.hole = "Human Appeal Path"
if "report" not in st.session_state: st.session_state.report = ""

llm = get_llm(st.secrets)

org_name = st.text_input("Enter Lead", value="Block")
if st.button("🔍 Scout & Sift"):
    with st.status(f"Hunting {org_name} in Regulations/Regulations/..."):
        news, analysis = scout_organization(org_name, llm)
        parts = [p.strip() for p in analysis.split("|")]
        if len(parts) >= 3:
            st.session_state.count = int(re.sub(r"\D", "", parts[1]))
            st.session_state.hole = parts[2]
            
            # THE FIX: No more data breach talk. This is an AI Compliance Audit.
            audit_prompt = f"""
            Draft a HIGH-STAKES AI COMPLIANCE AUDIT for {org_name}. 
            Context: They face an ${st.session_state.count * 20000:,} risk for {st.session_state.count} AI decisions.
            Requirement: Implement a {st.session_state.hole} by June 30, 2026 (SB 25B-004).
            Defense: Propose a NIST AI RMF-aligned Affirmative Defense.
            """
            st.session_state.report = llm.invoke(audit_prompt).content
            st.rerun()

st.markdown(st.session_state.report)

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info(f"Target Hole: {st.session_state.hole}")
    st.session_state.count = st.number_input("Affected Count:", value=st.session_state.count)
    statutory = st.session_state.count * 20000
    st.metric("Statutory Risk", f"${statutory:,}")
    st.caption("Enforcement Cliff: June 30, 2026")
