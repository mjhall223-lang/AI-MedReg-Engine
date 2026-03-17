import streamlit as st
import sys, os

# Path Alignment
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: sys.path.insert(0, current_dir)

from engine import get_llm, list_all_laws, smart_web_sifter, perform_gap_analysis, generate_pdf_report, extract_pdf_text

st.set_page_config(page_title="ReadyAudit 2026", layout="wide")

# Persistent Memory
if "audit_content" not in st.session_state: st.session_state.audit_content = ""
if "audit_results" not in st.session_state: st.session_state.audit_results = ""
if "hole_type" not in st.session_state: st.session_state.hole_type = "General"

llm = get_llm(st.secrets)
all_laws = list_all_laws()

with st.sidebar:
    st.header("🛡️ LAW DATABASE")
    current_laws = st.multiselect("Active Audit Laws:", options=all_laws, default=all_laws)
    st.error("Enforcement Cliff: June 30, 2026")

st.header("📁 Remediation & Gap Analysis Engine")
org_name = st.text_input("Lead Entity", value="Synchron")
mode = st.radio("Mode:", ["Web Sifter", "Manual Paste", "File Upload"])

# Loading Logic
if mode == "Web Sifter":
    if st.button("🔍 Sift Public Web"):
        with st.spinner("Sifting..."):
            st.session_state.audit_content = smart_web_sifter(org_name)
elif mode == "Manual Paste":
    pasted = st.text_area("Paste Press Release/Policy Text:", height=300)
    if pasted: st.session_state.audit_content = pasted
elif mode == "File Upload":
    f = st.file_uploader("Upload PDF", type=['pdf'])
    if f: st.session_state.audit_content = extract_pdf_text(f)

# The Audit Trigger
if st.button("🛠️ Run Remediation Audit") and st.session_state.audit_content:
    with st.status("🚀 Running Strict Statutory Audit...", expanded=True) as status:
        st.write("🔍 Identifying decoding-to-action pathways...")
        st.write("⚖️ Calculating statutory debt (SB 24-205)...")
        st.session_state.audit_results = perform_gap_analysis(st.session_state.audit_content, current_laws, org_name, llm)
        status.update(label="✅ Audit Complete!", state="complete")
    
    st.markdown(st.session_state.audit_results)

# Export
if st.session_state.audit_results:
    pdf = generate_pdf_report(st.session_state.audit_results, org_name, st.session_state.hole_type, current_laws)
    st.download_button("📥 Download PDF Report", data=pdf, file_name=f"{org_name}_Audit.pdf")
