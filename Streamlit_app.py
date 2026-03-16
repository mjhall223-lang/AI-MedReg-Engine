import streamlit as st
import re
from engine import get_llm, scout_organization

st.set_page_config(page_title="ReadyAudit", layout="wide")

# Persistent State
if "count" not in st.session_state: st.session_state.count = 1
if "report" not in st.session_state: st.session_state.report = ""
if "hole" not in st.session_state: st.session_state.hole = "Governance Gap"

llm = get_llm(st.secrets)

org_name = st.text_input("Enter Lead", value="Block")
if st.button("🔍 Scout & Sift"):
    with st.status(f"Hunting {org_name} triggers..."):
        news, analysis = scout_organization(org_name, llm)
        parts = [p.strip() for p in analysis.split("|")]
        if len(parts) >= 3:
            # SAFETY NET: Rips digits, but defaults to 1 instead of 0
            digits = re.sub(r"\D", "", parts[1])
            st.session_state.count = int(digits) if (digits and int(digits) > 0) else 1
            st.session_state.hole = parts[2]
            
            # Cites the June 30, 2026 Enforcement Cliff (SB 25B-004)
            st.session_state.report = llm.invoke(f"Pitch for {org_name}. Count: {st.session_state.count}. Hole: {parts[2]}. Deadline: June 30, 2026.").content
            st.rerun()

st.markdown(st.session_state.report)

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info(f"Target Hole: {st.session_state.hole}")
    st.session_state.count = st.number_input("Count:", value=st.session_state.count)
    
    # Statutory Calculation: $20,000 per violation
    statutory = st.session_state.count * 20000
    st.metric("Statutory Risk", f"${statutory:,}")
    st.metric("Governance Debt", f"${round(statutory * 1.25, 2):,}")
    st.caption("Enforcement Cliff: June 30, 2026")
