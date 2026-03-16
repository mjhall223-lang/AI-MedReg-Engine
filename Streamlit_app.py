import streamlit as st
import re
from engine import get_llm, scout_organization

st.set_page_config(page_title="ReadyAudit", layout="wide")

if "count" not in st.session_state: st.session_state.count = 1
if "report" not in st.session_state: st.session_state.report = ""
if "hole" not in st.session_state: st.session_state.hole = "Governance Gap"

llm = get_llm(st.secrets)

org_name = st.text_input("Enter Lead", value="Block")
if st.button("🔍 Scout & Sift"):
    with st.status(f"Hunting {org_name}..."):
        news, analysis = scout_organization(org_name, llm)
        parts = [p.strip() for p in analysis.split("|")]
        if len(parts) >= 3:
            # SAFETY NET: Rips digits, but handles empty strings
            digits = re.sub(r"\D", "", parts[1])
            st.session_state.count = int(digits) if digits else 1
            st.session_state.hole = parts[2]
            
            # Update Pitch
            st.session_state.report = llm.invoke(f"Draft 2026 pitch for {org_name} regarding {parts[2]}").content
            st.rerun()

st.markdown(st.session_state.report)

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info(f"Hole: {st.session_state.hole}")
    st.session_state.count = st.number_input("Count:", value=st.session_state.count)
    statutory = st.session_state.count * 20000
    st.metric("Statutory Risk", f"${statutory:,}")
    st.caption("Deadline: June 30, 2026")
