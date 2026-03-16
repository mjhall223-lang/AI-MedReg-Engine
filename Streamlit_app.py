import sys, os, re, streamlit as st
from engine import get_llm, scout_organization

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
st.set_page_config(page_title="ReadyAudit: Specialist Hub", layout="wide")

if "count" not in st.session_state: st.session_state.count = 1
if "report" not in st.session_state: st.session_state.report = ""
if "hole" not in st.session_state: st.session_state.hole = "Governance Gap"

llm = get_llm(st.secrets)
tab1, tab2 = st.tabs(["📁 Deep Audit", "🤖 Autonomous Hunter"])

with tab2:
    org_name = st.text_input("Enter Lead (e.g., 'Block', 'Neuralink')")
    if st.button("🔍 Scout & Sift"):
        with st.status(f"Hunting {org_name} 2026 triggers..."):
            news, analysis = scout_organization(org_name, llm)
            try:
                parts = [p.strip() for p in analysis.split("|")]
                if len(parts) >= 3:
                    count_match = re.search(r'\d+', parts[1].replace(',', ''))
                    st.session_state.count = int(count_match.group()) if count_match else 1
                    st.session_state.hole = parts[2]
                    
                    # SPECIALIST PROMPT: No more marketing fluff.
                    audit_prompt = f"""
                    Draft a HIGH-STAKES COMPLIANCE AUDIT PROPOSAL for {org_name}.
                    Context: They are facing a ${st.session_state.count * 20000:,} statutory risk.
                    Focus: Implementing a {st.session_state.hole} for the June 30, 2026 deadline.
                    Requirement: CITE SB 25B-004 and the NIST AI RMF Affirmative Defense.
                    """
                    st.session_state.report = llm.invoke(audit_prompt).content
                    st.rerun() 
            except: st.error("Sift failed.")

if st.session_state.report:
    st.markdown(f"### 🛡️ Compliance Audit Proposal: {org_name}")
    st.markdown(st.session_state.report)

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info(f"Hole: {st.session_state.hole}")
    st.session_state.count = st.number_input("Affected Count:", value=st.session_state.count)
    
    statutory = st.session_state.count * 20000
    st.metric("Statutory Risk (SB 24-205)", f"${statutory:,}")
    st.metric("Total Governance Debt", f"${round(statutory * 1.25, 2):,}")
    st.caption("Enforcement Cliff: June 30, 2026")
