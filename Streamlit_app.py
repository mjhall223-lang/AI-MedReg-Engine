import streamlit as st
import re
from engine import get_llm, scout_organization, SpecialistMath

st.set_page_config(page_title="ReadyAudit", layout="wide")

# STATE LOCK: Prevents the $200k reset
if "count" not in st.session_state: st.session_state.count = 4000
if "hole" not in st.session_state: st.session_state.hole = "Human Appeal Path"
if "report" not in st.session_state: st.session_state.report = ""

llm = get_llm(st.secrets)

org_name = st.text_input("Enter Lead", value="Block")
if st.button("🔍 Scout & Sift"):
    with st.status("Hunting 2026 Triggers..."):
        news, analysis = scout_organization(org_name, llm)
        parts = [p.strip() for p in analysis.split("|")]
        if len(parts) >= 3:
            # Extracts 4000 or 21
            st.session_state.count = int(re.sub(r"\D", "", parts[1]))
            st.session_state.hole = parts[2]
            st.session_state.report = llm.invoke(f"Pitch for {org_name}. Hole: {parts[2]}. Deadline: June 30, 2026.").content
            st.rerun()

st.markdown(st.session_state.report)

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info(f"Target Hole: {st.session_state.hole}")
    
    # Input is now LOCKED to the scout results
    st.session_state.count = st.number_input("Count:", value=st.session_state.count)
    impact = SpecialistMath.calculate(st.session_state.count)
    
    st.metric("Statutory Risk", f"${impact['statutory']:,}")
    st.metric("Total Governance Debt", f"${impact['total']:,}")
    st.caption("Cliff: June 30, 2026")
