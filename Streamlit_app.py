import streamlit as st
from engine import get_llm, scout_organization, SpecialistMath, create_pdf

st.set_page_config(page_title="ReadyAudit: Specialist Hub", layout="wide")

if "count" not in st.session_state: st.session_state.count = 10
if "report" not in st.session_state: st.session_state.report = ""
if "analysis" not in st.session_state: st.session_state.analysis = "Enterprise | 10 | General Risk"

llm = get_llm(st.secrets)

tab1, tab2 = st.tabs(["📁 Deep Audit", "🤖 Autonomous Hunter"])

with tab2:
    org_name = st.text_input("Enter Organization (e.g., 'Block', 'Neuralink', 'Goldman Sachs')")
    if st.button("🔍 Scout & Auto-Tailor"):
        with st.status(f"Hunting {org_name} triggers..."):
            news, analysis = scout_organization(org_name, llm)
            st.session_state.analysis = analysis
            
            # Parse the analysis: Industry | Number | Trigger
            parts = analysis.split("|")
            industry = parts[0].strip()
            count = int(re.sub(r"\D", "", parts[1]))
            trigger = parts[2].strip()
            
            st.session_state.count = count
            math = SpecialistMath.calculate(industry, count)
            
            # THE TAILORED PITCH PROMPT
            pitch_prompt = f"""
            March 16, 2026. Lead: {org_name}. Count: {count}. Industry: {industry}.
            Trigger: {trigger}. News: {news[:500]}.
            Deadline: June 30 Colorado AI Act (SB 24-205 & SB 25B-004).
            Draft a Specialist Pitch targeting the $20,000 violation risk. 
            If MedTech: Focus on 'Patient Safety & Clinical Integrity'.
            If Fintech: Focus on 'Algorithmic Discrimination & Hiring'.
            CITE the 'Affirmative Defense' Safe Harbor via NIST AI RMF.
            """
            st.session_state.report = llm.invoke(pitch_prompt).content
            st.rerun() 

    if st.session_state.report:
        st.markdown(st.session_state.report)
        pdf = create_pdf(st.session_state.report)
        st.download_button("📩 Download Pitch PDF", pdf, f"{org_name}_Pitch.pdf")

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info(f"Trigger: {st.session_state.analysis.split('|')[-1]}")
    st.session_state.count = st.number_input("Affected Personnel/Subjects:", value=st.session_state.count)
    
    impact = SpecialistMath.calculate("General", st.session_state.count)
    st.metric("Statutory Risk (SB 24-205)", f"${impact['statutory']:,}")
    st.metric("Total Governance Debt", f"${impact['total']:,}")
