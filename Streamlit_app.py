import sys, os, re, streamlit as st
from engine import get_llm, scout_organization, SpecialistMath

# Force path persistence for Streamlit Cloud
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
st.set_page_config(page_title="ReadyAudit: Specialist Hub", layout="wide")

# Persistent State Management
if "count" not in st.session_state: st.session_state.count = 10
if "report" not in st.session_state: st.session_state.report = ""
if "hole" not in st.session_state: st.session_state.hole = "Governance Gap"

llm = get_llm(st.secrets)
tab1, tab2 = st.tabs(["📁 Deep Audit", "🤖 Autonomous Hunter"])

with tab2:
    org_name = st.text_input("Enter Lead (e.g., 'Block', 'Neuralink')")
    if st.button("🔍 Scout & Sift"):
        with st.status(f"Hunting {org_name} triggers..."):
            news, analysis = scout_organization(org_name, llm)
            try:
                parts = [p.strip() for p in analysis.split("|")]
                if len(parts) >= 3:
                    # Capture '4000' or '21'
                    count_match = re.search(r'\d+', parts[1].replace(',', ''))
                    count = int(count_match.group()) if count_match else 10
                    
                    # SYNC: Lock to session state
                    st.session_state.count = count
                    st.session_state.hole = parts[2]
                    
                    pitch_prompt = f"Specialist Pitch for {org_name}. Count: {count}. Hole: {st.session_state.hole}. Deadline: June 30, 2026."
                    st.session_state.report = llm.invoke(pitch_prompt).content
                    st.rerun() 
            except Exception: st.error("Sift failed.")

if st.session_state.report:
    st.markdown(st.session_state.report)

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info(f"Target Hole: {st.session_state.hole}")
    st.session_state.count = st.number_input("Affected Count:", value=st.session_state.count)
    impact = SpecialistMath.calculate(st.session_state.count)
    st.metric("Statutory Risk", f"${impact['statutory']:,}")
    st.metric("Total Governance Debt", f"${impact['total']:,}")
    st.caption("Enforcement Cliff: June 30, 2026")
