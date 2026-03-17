import streamlit as st
import sys
import os

# --- PATH ALIGNMENT: Prevents Redacted Import Errors on Cloud ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import modular functions
try:
    from engine import (
        get_llm, 
        list_all_laws, 
        smart_web_sifter, 
        perform_gap_analysis, 
        generate_pdf_report,
        extract_pdf_text
    )
except ImportError as e:
    st.error(f"Critical Error: Could not find engine.py. Ensure both files are in the same folder. {e}")
    st.stop()

st.set_page_config(page_title="AI-MedReg-Engine | 2026 Compliance", layout="wide")

# Persistent Session Memory
if "audit_content" not in st.session_state: st.session_state.audit_content = ""
if "audit_results" not in st.session_state: st.session_state.audit_results = ""

llm = get_llm(st.secrets)
all_laws = list_all_laws()

with st.sidebar:
    st.header("🛡️ REGULATORY STACK")
    current_laws = st.multiselect("Selected Regulations:", options=all_laws, default=all_laws)
    st.divider()
    st.warning("⏱️ Enforcement Cliff: June 30, 2026")

st.header("🔬 Remediation & Statutory Audit Engine")
org_name = st.text_input("Lead Entity", value="Synchron")
mode = st.radio("Sourcing Mode:", ["Web Sifter", "Manual Paste", "File Upload"])

# --- STEP 1: LOAD CONTENT ---
if mode == "Web Sifter":
    if st.button("🔍 Sift Public Web"):
        with st.spinner(f"Hunting for {org_name} Chiral™ governance..."):
            st.session_state.audit_content = smart_web_sifter(org_name)
            if "Error:" not in st.session_state.audit_content:
                st.success("Web Content Successfully Ingested.")
            else:
                st.error(st.session_state.audit_content)
                st.info("Try 'Manual Paste' if bot-shields are active.")

elif mode == "Manual Paste":
    pasted_text = st.text_area("Paste News/Policy Text Here:", height=300)
    if pasted_text:
        st.session_state.audit_content = pasted_text
        st.toast("Manual data loaded.", icon="📥")

elif mode == "File Upload":
    f = st.file_uploader("Upload Policy PDF", type=['pdf'])
    if f:
        st.session_state.audit_content = extract_pdf_text(f)

# --- STEP 2: AUDIT & REMEDIATE ---
if st.button("🛠️ Run Statutory Audit"):
    if st.session_state.audit_content and current_laws:
        with st.status("🔍 Cross-referencing 2026 Statutory Cliff...", expanded=True) as status:
            st.session_state.audit_results = perform_gap_analysis(st.session_state.audit_content, current_laws, org_name, llm)
            status.update(label="✅ Audit Complete!", state="complete")
        st.markdown(st.session_state.audit_results)
    else:
        st.warning("Please load content and select regulations first.")

# --- STEP 3: EXPORT ---
if st.session_state.audit_results:
    pdf = generate_pdf_report(st.session_state.audit_results, org_name, current_laws)
    st.download_button("📥 Download Remediation PDF", data=pdf, file_name=f"{org_name}_Audit_2026.pdf")
