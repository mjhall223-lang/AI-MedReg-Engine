import streamlit as st
from engine import get_llm, perform_gap_analysis

st.set_page_config(page_title="ReadyAudit: Remediation Engine", layout="wide")
llm = get_llm(st.secrets)

st.header("📁 Remediation & Gap Analysis Engine")
st.write("Upload docs to identify 'Holes' for the June 30, 2026 deadline.")

org_name = st.text_input("Lead Entity Name", value="Block")
uploaded_file = st.file_uploader("Upload Tech/Policy File", type=['txt', 'pdf', 'md'])

if st.button("🛠️ Run Remediation Audit"):
    if uploaded_file:
        content = uploaded_file.read().decode("utf-8")
        with st.status(f"Sifting Regulations for {org_name}..."):
            results = perform_gap_analysis(content, org_name, llm)
            st.markdown(results)
    else:
        st.warning("Please upload a file to begin.")

with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.success("Target: Affirmative Defense")
    st.error("Enforcement Cliff: June 30, 2026")
    st.caption("Tracking: SB 24-205, SB 25B-004, FDA PCCP, NIST AI RMF")
