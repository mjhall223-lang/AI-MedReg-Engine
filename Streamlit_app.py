import streamlit as st
import re
from engine import get_llm, scout_organization

st.set_page_config(page_title="ReadyAudit: Specialist Hub", layout="wide")

# Persistent State Management
if "count" not in st.session_state: st.session_state.count = 4000
if "report" not in st.session_state: st.session_state.report = ""
if "hole" not in st.session_state: st.session_state.hole = "Human Appeal Path"

llm = get_llm(st.secrets)

org_name = st.text_input("Enter Lead", value="Block")
if st.button("🔍 Scout & Sift"):
    with st.status(f"Hunting {org_name} 2026 triggers..."):
        news, analysis = scout_organization(org_name, llm)
        try:
            parts = [p.strip() for p in analysis.split("|")]
            if len(parts) >= 3:
                # REGEX FIX: Rips digits out even if LLM adds text
                count_match = re.search(r'\d+', parts[1].replace(',', ''))
                st.session_state.count = int(count_match.group()) if count_match else 4000
                st.session_state.hole = parts[2]
                
                # PITCH: Cites June 30, 2026 Enforcement Cliff (SB 25B-004)
                st.session_state.report = llm.invoke(f"Pitch for {org_name}. Count: {st.session_state.count}. Hole: {parts[2]}. Deadline: June 30, 2026.").content
                st.rerun() 
        except: st.error("Sift failed.")

st.markdown(st.session_state.report)

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info(f"Target Hole: {st.session_state.hole}")
    st.session_state.count = st.number_input("Affected Count:", value=st.session_state.count)
    
    statutory = st.session_state.count * 20000
    st.metric("Statutory Risk", f"${statutory:,}")
    st.metric("Total Governance Debt", f"${round(statutory * 1.25, 2):,}")
    st.caption("Enforcement Cliff: June 30, 2026")
