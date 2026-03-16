import streamlit as st
import sys
import os

# --- PATH ALIGNMENT: Forces Streamlit Cloud to see engine.py ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import from your separate engine file
from engine import get_llm, list_all_laws, web_sifter, perform_gap_analysis, generate_pdf_report

st.set_page_config(page_title="ReadyAudit: Remediation Engine", layout="wide")
llm = get_llm(st.secrets)

# Persistent Memory (Session State)
if "audit_content" not in st.session_state: st.session_state.audit_content = ""
if "audit_results" not in st.session_state: st.session_state.audit_results = ""
if "hole_type" not in st.session_state: st.session_state.hole_type = "General"

all_laws = list_all_laws()

def update_selections():
    if st.session_state.select_all_check:
        st.session_state.selected_laws = all_laws
    else:
        st.session_state.selected_laws = []

with st.sidebar:
    st.header("🛡️ LAW DATABASE")
    st.checkbox("Select All Laws", value=True, key="select_all_check", on_change=update_selections)
    current_laws = st.multiselect("Active Audit Laws:", options=all_laws, key="selected_laws")
    st.divider()
    st.error("Enforcement Cliff: June 30, 2026")

st.header("📁 Remediation & Gap Analysis Engine")
org_name = st.text_input("Lead Entity", value="Synchron")
mode = st.radio("Mode:", ["Web Sifter", "File Upload"])

# Content Sifting Logic
if mode == "Web Sifter":
    if st.button("🔍 Sift Public Web"):
        with st.spinner(f"Sifting for {org_name}..."):
            st.session_state.audit_content = web_sifter(org_name)
            if "Error:" not in st.session_state.audit_content:
                st.success("Web Content Loaded.")
            else:
                st.error(st.session_state.audit_content)
else:
    f = st.file_uploader("Upload Policy PDF", type=['pdf'])
    if f:
        st.session_state.audit_content = extract_pdf_text(f) # Add this to engine.py

# Run Audit
if st.button("🛠️ Run Remediation Audit"):
    if st.session_state.audit_content and current_laws:
        with st.status("Analyzing Statutory Gaps..."):
            st.session_state.audit_results = perform_gap_analysis(st.session_state.audit_content, current_laws, org_name, llm)
            if "human" in st.session_state.audit_results.lower():
                st.session_state.hole_type = "Human Appeal"
            st.markdown(st.session_state.audit_results)

# Export
if st.session_state.audit_results:
    pdf = generate_pdf_report(st.session_state.audit_results, org_name, st.session_state.hole_type, current_laws)
    st.download_button("📥 Download PDF", data=pdf, file_name=f"{org_name}_Audit.pdf")
