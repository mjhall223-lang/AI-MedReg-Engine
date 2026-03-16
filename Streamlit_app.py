import streamlit as st
from engine import get_llm, list_all_laws, web_sifter, perform_gap_analysis, generate_pdf_report

st.set_page_config(page_title="ReadyAudit: Remediation Engine", layout="wide")
llm = get_llm(st.secrets)
all_laws = list_all_laws()

# CALLBACK: Ensures the multiselect updates with the checkbox instantly
def update_selections():
    if st.session_state.select_all_check:
        st.session_state.selected_laws = all_laws
    else:
        st.session_state.selected_laws = []

if "audit_results" not in st.session_state: st.session_state.audit_results = ""

with st.sidebar:
    st.header("🛡️ LAW DATABASE")
    # THE TOGGLE
    st.checkbox("Select All Laws", value=True, key="select_all_check", on_change=update_selections)
    current_selections = st.multiselect("Active Audit Laws:", options=all_laws, key="selected_laws")
    st.divider()
    st.error("Enforcement Cliff: June 30, 2026")

st.header("📁 Remediation & Gap Analysis Engine")
mode = st.radio("Audit Mode:", ["Web Sifter (Public Policy)", "File Upload (Private Docs)"])
org_name = st.text_input("Lead Entity", value="Synchron")

# LOGIC FOR LOADING CONTENT (Web vs File)
content = ""
if mode == "Web Sifter (Public Policy)":
    if st.button("🔍 Sift Public Web"):
        content = web_sifter(org_name)
        st.success("Public Policy found and loaded.")
else:
    f = st.file_uploader("Upload Policy PDF", type=['pdf'])
    if f:
        content = extract_pdf_text(f)

# THE AUDIT
if st.button("🛠️ Run Remediation Audit") and content:
    with st.status(f"Auditing {org_name}..."):
        st.session_state.audit_results = perform_gap_analysis(content, current_selections, org_name, llm)
        st.markdown(st.session_state.audit_results)

# PDF EXPORT
if st.session_state.audit_results:
    pdf = generate_pdf_report(st.session_state.audit_results, org_name, "Human", current_selections)
    st.download_button("📥 Download PDF Report", data=pdf, file_name=f"{org_name}_Audit.pdf")
