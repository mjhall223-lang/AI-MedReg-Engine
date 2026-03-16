import sys, os, re, streamlit as st
from engine import get_llm, scout_organization, SpecialistMath

# Force path for Streamlit Cloud
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
st.set_page_config(page_title="ReadyAudit: Specialist Hub", layout="wide")

# 1. PERSISTENT STATE MANAGEMENT
if "count" not in st.session_state: st.session_state.count = 10
if "report" not in st.session_state: st.session_state.report = ""
if "hole" not in st.session_state: st.session_state.hole = "Governance Gap"

llm = get_llm(st.secrets)
tab1, tab2 = st.tabs(["📁 Deep Audit", "🤖 Autonomous Hunter"])

with tab2:
    org_name = st.text_input("Enter Lead (e.g., 'Block', 'Neuralink')")
    if st.button("🔍 Scout & Sift"):
        st.session_state.report = "" 
        with st.status(f"Hunting {org_name} 2026 triggers..."):
            news, analysis = scout_organization(org_name, llm)
            try:
                # PARSE: Handles "Industry | Number | Hole"
                parts = [p.strip() for p in analysis.split("|")]
                if len(parts) >= 3:
                    # REGEX FIX: Pulls digits out even if LLM adds text
                    count_match = re.search(r'\d+', parts[1].replace(',', ''))
                    count = int(count_match.group()) if count_match else 10
                    
                    # 2. LOCK THE STATE: Sidebar pulls from here
                    st.session_state.count = count
                    st.session_state.hole = parts[2]
                    
                    # 3. PITCH: Targets the June 30, 2026 Deadline
                    pitch_prompt = f"Draft a Specialist Pitch for {org_name}. Count: {count}. Hole: {st.session_state.hole}. Target $20k violation risk under SB 25B-004. Deadline: June 30, 2026."
                    st.session_state.report = llm.invoke(pitch_prompt).content
                    
                    # 4. THE SYNC FIX: Forces UI refresh to update the Sidebar math
                    st.rerun() 
            except Exception as e:
                st.error(f"Sift failed. Error: {e}")

if st.session_state.report:
    st.markdown(f"### 🛡️ Specialized Pitch for {org_name}")
    st.markdown(st.session_state.report)

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info(f"Target Hole: {st.session_state.hole}")
    
    # Value is now SLAVED to the scout results (21 or 4237)
    st.session_state.count = st.number_input("Affected Personnel/Subjects:", value=st.session_state.count)
    
    impact = SpecialistMath.calculate(st.session_state.count)
    st.metric("Statutory Risk (SB 24-205)", f"${impact['statutory']:,}")
    st.metric("Total Governance Debt", f"${impact['total']:,}")
    st.caption("Enforcement Cliff: June 30, 2026")
