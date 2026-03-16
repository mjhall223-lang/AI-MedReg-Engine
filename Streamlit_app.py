import streamlit as st
from engine import get_llm, perform_gap_analysis

st.set_page_config(page_title="ReadyAudit: Remediation Engine", layout="wide")

llm = get_llm(st.secrets)

st.header("📁 Remediation & Gap Analysis Engine")
st.write("Upload a lead's technical architecture or AI policy to find the holes in their 2026 compliance.")

col1, col2 = st.columns([2, 1])

with col1:
    org_name = st.text_input("Entity Name (e.g., Block)", value="Block")
    count = st.number_input("Affected Personnel (e.g., 4000 layoffs)", value=4000)
    uploaded_file = st.file_uploader("Upload Tech/Policy File", type=['txt', 'pdf', 'md'])
    
    if st.button("🛠️ Run Remediation Audit"):
        if uploaded_file:
            content = uploaded_file.read().decode("utf-8")
            with st.spinner("Sifting Regulations/Regulations folder..."):
                results = perform_gap_analysis(content, org_name, count, llm)
                st.markdown(results)
        else:
            st.warning("Please upload a file to begin the audit.")

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    statutory = count * 20000
    st.metric("Statutory Risk", f"${statutory:,}")
    st.metric("Total Governance Debt", f"${round(statutory * 1.25, 2):,}")
    st.error("Enforcement Cliff: June 30, 2026")
    st.caption("Tracking: SB 24-205, SB 25B-004, NIST AI RMF")
