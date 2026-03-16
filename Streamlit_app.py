import streamlit as st
from engine import get_llm, list_all_laws, perform_gap_analysis

st.set_page_config(page_title="ReadyAudit: Remediation Engine", layout="wide")
llm = get_llm(st.secrets)

# 1. Fetch all laws from your nested Regulations/Regulations/... folders
all_available_laws = list_all_laws()

with st.sidebar:
    st.header("🛡️ LAW DATABASE")
    
    # Toggle All Logic
    select_all = st.checkbox("Select All Laws", value=True)
    
    if select_all:
        selected_laws = st.multiselect(
            "Active Audit Laws:", 
            all_available_laws, 
            default=all_available_laws
        )
    else:
        selected_laws = st.multiselect(
            "Active Audit Laws:", 
            all_available_laws
        )
        
    st.divider()
    st.error("Enforcement Cliff: June 30, 2026")

st.header("📁 Remediation & Gap Analysis Engine")
uploaded_file = st.file_uploader("Upload Tech/Policy File", type=['txt', 'pdf', 'md'])

if st.button("🛠️ Run Remediation Audit"):
    if not selected_laws:
        st.warning("Please toggle at least one law in the sidebar.")
    elif uploaded_file:
        content = uploaded_file.read().decode("utf-8")
        with st.status("Analyzing against selected database..."):
            results = perform_gap_analysis(content, selected_laws, llm)
            st.markdown(results)
    else:
        st.warning("Please upload a file to begin the audit.")
