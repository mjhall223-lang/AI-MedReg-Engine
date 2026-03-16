import sys, os, re, streamlit as st
from engine import get_llm, scout_organization, SpecialistMath, create_pdf

# Path fix for deployment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
st.set_page_config(page_title="ReadyAudit: Specialist Hub", layout="wide")

# Persistent State Management
if "count" not in st.session_state: st.session_state.count = 10
if "report" not in st.session_state: st.session_state.report = ""
if "hole" not in st.session_state: st.session_state.hole = "Governance Gap"

llm = get_llm(st.secrets)
tab1, tab2 = st.tabs(["📁 Deep Audit", "🤖 Autonomous Hunter"])

with tab2:
    org_name = st.text_input("Enter Lead (e.g., 'Neuralink', 'Block')")
    if st.button("🔍 Scout & Sift"):
        st.session_state.report = "" # Clean old ghosts
        with st.status(f"Hunting {org_name} triggers..."):
            news, analysis = scout_organization(org_name, llm)
            try:
                parts = analysis.split("|")
                industry = parts[0].strip()
                
                # FIX: Data-Safe Number Extraction
                count_str = re.sub(r"\D", "", parts[1])
                count = int(count_str) if count_str else 10
                hole = parts[2].strip()
                
                # Force State Update
                st.session_state.count = count
                st.session_state.hole = hole
                
                # Pitch Generation
                pitch_prompt = f"""
                Draft a Specialist Pitch for {org_name}. Count: {count}. Hole: {hole}.
                Target $20,000/violation risk under SB 25B-004.
                Focus: NIST AI RMF Affirmative Defense. Deadline: June 30, 2026.
                BANNED: Mentioning other companies (no 'Block' in 'Neuralink' pitch).
                """
                st.session_state.report = llm.invoke(pitch_prompt).content
                
                # THE SYNC FIX: Forces sidebar to catch up to the new count
                st.rerun() 
            except Exception as e:
                st.error(f"Sift failed to parse count. Using default. Error: {e}")

if st.session_state.report:
    st.markdown(f"### 🛡️ Specialized Pitch for {org_name}")
    st.markdown(st.session_state.report)

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info(f"Target Hole: {st.session_state.hole}")
    
    # Linked to state: automatically updates when 'Scout' finds 21 or 4,237
    st.session_state.count = st.number_input("Affected Personnel/Subjects:", value=st.session_state.count)
    
    impact = SpecialistMath.calculate(st.session_state.count)
    st.metric("Statutory Risk (SB 24-205)", f"${impact['statutory']:,}")
    st.metric("Total Governance Debt", f"${impact['total']:,}")
