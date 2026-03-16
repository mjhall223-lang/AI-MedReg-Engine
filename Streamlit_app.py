import streamlit as st
from engine import get_llm, list_all_laws, extract_pdf_text, perform_gap_analysis

st.set_page_config(page_title="ReadyAudit: Remediation Engine", layout="wide")
llm = get_llm(st.secrets)

all_available_laws = list_all_laws()

with st.sidebar:
    st.header("🛡️ LAW DATABASE")
    select_all = st.checkbox("Select All Laws", value=True)
    selected_laws = st.multiselect("Active Audit Laws:", all_available_laws, 
                                   default=all_available_laws if select_all else None)
    st.divider()
    st.error("Enforcement Cliff: June 30, 2026")

st.header("📁 Remediation & Gap Analysis Engine")
uploaded_file = st.file_uploader("Upload Tech/Policy File", type=['pdf', 'txt'])

if st.button("🛠️ Run Remediation Audit"):
    if uploaded_file and selected_laws:
        with st.status("Analyzing binary content against database..."):
            # Handles the PDF or Text branching
            content = extract_pdf_text(uploaded_file) if uploaded_file.name.endswith('.pdf') else uploaded_file.read().decode("utf-8")
            results = perform_gap_analysis(content, selected_laws, "Target_Entity", llm)
            st.markdown(results)
