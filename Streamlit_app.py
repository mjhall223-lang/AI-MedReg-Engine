import streamlit as st
import sys, os

# --- PATH ALIGNMENT: Forces Streamlit Cloud to see engine.py ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from engine import get_llm, list_all_laws, smart_web_sifter, perform_gap_analysis, generate_pdf_report, extract_pdf_text

st.set_page_config(page_title="ReadyAudit 2026", layout="wide")

# Persistent State
if "audit_content" not in st.session_state: st.session_state.audit_content = ""
if "audit_results" not in st.session_state: st.session_state.audit_results = ""
if "hole_type" not in st.session_state: st.session_state.hole_type = "General"

llm = get_llm(st.secrets)
all_laws = list_all_laws()

with st.sidebar:
    st.header("🛡️ LAW DATABASE")
    st.checkbox("Select All Laws", value=True, key="select_all_check")
    # Logic for Select All
    if st.session_state.select_all_check:
        current_laws = st.multiselect("Active Audit Laws:", options=all_laws, default=all_laws)
    else:
        current_laws = st.multiselect("Active Audit Laws:", options=all_laws)
    st.divider()
    st.error("Enforcement Cliff: June 30, 2026")

st.header("📁 Remediation & Gap Analysis Engine")
org_name = st.text_input("Lead Entity", value="Synchron")
mode = st.radio("Mode:", ["Web Sifter", "Manual Paste", "File Upload"])

# --- CONTENT LOADING ---
if mode == "Web Sifter":
    if st.button("🔍 Sift Public Web"):
        with st.spinner(f"Sifting for {org_name}..."):
            st.session_state.audit_content = smart_web_sifter(org_name)
            if "Error:" in st.session_state.audit_content:
                st.error(st.session_state.audit_content)
                st.info("💡 Try 'Manual Paste' mode if the search is blocked by bot-shields.")
            else:
                st.success("Web Content Loaded.")

elif mode == "Manual Paste":
    pasted_text = st.text_area("Paste Company Policy/Press Release Text Here:", height=300)
    if pasted_text:
        st.session_state.audit_content = pasted_text
        st.success("Manual Content Loaded.")

elif mode == "File Upload":
    f = st.file_uploader("Upload PDF", type=['pdf'])
    if f: st.session_state.audit_content = extract_pdf_text(f)

# --- AUDIT RUN ---
if st.button("🛠️ Run Remediation Audit") and st.session_state.audit_content:
    with st.status("Analyzing Statutory Gaps..."):
        st.session_state.audit_results = perform_gap_analysis(st.session_state.audit_content, current_laws, org_name, llm)
        if "human" in st.session_state.audit_results.lower():
            st.session_state.hole_type = "Human Appeal"
        st.markdown(st.session_state.audit_results)

# --- EXPORT ---
if st.session_state.audit_results:
    pdf = generate_pdf_report(st.session_state.audit_results, org_name, st.session_state.hole_type, current_laws)
    st.download_button("📥 Download PDF Report", data=pdf, file_name=f"{org_name}_Audit.pdf")
