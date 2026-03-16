import sys, os, re, streamlit as st
from engine import get_llm, scout_organization, SpecialistMath

# Path fix for deployment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
st.set_page_config(page_title="ReadyAudit: Specialist Hub", layout="wide")

# Persistent State
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
                # 1. PARSE: Handles "Industry | Number | Hole"
                parts = [p.strip() for p in analysis.split("|")]
                if len(parts) >= 3:
                    industry = parts[0]
                    # REGEX FIX: Pulls digits from strings like "4,000 people" or "21 subjects"
                    count_match = re.search(r'\d+', parts[1].replace(',', ''))
                    count = int(count_match.group()) if count_match else 10
                    hole = parts[2]
                    
                    # 2. STATE SYNC: Forces the sidebar to update
                    st.session_state.count = count
                    st.session_state.hole = hole
                    
                    pitch_prompt = f"""
                    Draft a Specialist Pitch for {org_name}. Count: {count}. Hole: {hole}.
                    Target the $20,000 violation risk under Colorado SB 25B-004.
                    Focus: NIST AI RMF Affirmative Defense. Deadline: June 30, 2026.
                    """
                    st.session_state.report = llm.invoke(pitch_prompt).content
                    
                    # 3. THE TRIGGER: Forces the UI to refresh with the new math
                    st.rerun() 
                else:
                    st.error("Format error in sifting data.")
            except Exception as e:
                st.error(f"Sift failed. Error: {e}")

if st.session_state.report:
    st.markdown(f"### 🛡️ Specialized Pitch for {org_name}")
    st.markdown(st.session_state.report)

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info(f"Target Hole: {st.session_state.hole}")
    
    # Updated value to reflect the actual March 2026 data (21 or 4000)
    st.session_state.count = st.number_input("Affected Subjects/Staff:", value=st.session_state.count)
    
    impact = SpecialistMath.calculate(st.session_state.count)
    st.metric("Statutory Risk (SB 24-205)", f"${impact['statutory']:,}")
    st.metric("Total Governance Debt", f"${impact['total']:,}")
