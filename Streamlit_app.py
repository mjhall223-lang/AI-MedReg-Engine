import streamlit as st
import re
from engine import get_llm, scout_organization

st.set_page_config(page_title="ReadyAudit: Specialist Hub", layout="wide")

# STATE PERSISTENCE: Initialize with 2026 ground truth
if "count" not in st.session_state: st.session_state.count = 4000
if "report" not in st.session_state: st.session_state.report = ""
if "hole" not in st.session_state: st.session_state.hole = "Human Appeal Path"

llm = get_llm(st.secrets)

org_name = st.text_input("Enter Lead", value="Block")
if st.button("🔍 Scout & Sift"):
    with st.status(f"Hunting {org_name} in Regulations/Regulations/..."):
        news, analysis = scout_organization(org_name, llm)
        try:
            parts = [p.strip() for p in analysis.split("|")]
            if len(parts) >= 3:
                # Capture the "Beast Number" (4000 or 21)
                count_match = re.search(r'\d+', parts[1].replace(',', ''))
                st.session_state.count = int(count_match.group()) if count_match else 4000
                st.session_state.hole = parts[2]
                
                # AUDIT PROPOSAL: Cites the June 30, 2026 Cliff (SB 25B-004)
                audit_prompt = f"""
                Draft a HIGH-STAKES COMPLIANCE AUDIT PROPOSAL for {org_name}. 
                Risk: ${st.session_state.count * 20000:,}. 
                Hole: {parts[2]}. 
                Deadline: June 30, 2026 (SB 25B-004).
                """
                st.session_state.report = llm.invoke(audit_prompt).content
                st.rerun() 
        except: st.error("Sift failed.")

if st.session_state.report:
    st.markdown(st.session_state.report)

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info(f"Target Hole: {st.session_state.hole}")
    st.session_state.count = st.number_input("Affected Count:", value=st.session_state.count)
    
    # Statutory math: $20,000 per violation
    statutory = st.session_state.count * 20000
    st.metric("Statutory Risk", f"${statutory:,}")
    st.metric("Total Governance Debt", f"${round(statutory * 1.25, 2):,}")
    st.caption("Enforcement Cliff: June 30, 2026")
