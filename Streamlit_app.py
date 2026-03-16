import streamlit as st
from engine import get_llm, list_all_laws, extract_pdf_text, web_sifter, generate_pdf_report

st.set_page_config(page_title="ReadyAudit: Remediation Engine", layout="wide")
llm = get_llm(st.secrets)
all_laws = list_all_laws()

# CALLBACK for the Toggle All logic
def update_selections():
    if st.session_state.select_all_check:
        st.session_state.selected_laws = all_laws
    else:
        st.session_state.selected_laws = []

if "audit_results" not in st.session_state: st.session_state.audit_results = ""
if "hole_type" not in st.session_state: st.session_state.hole_type = "General"

with st.sidebar:
    st.header("🛡️ LAW DATABASE")
    st.checkbox("Select All Laws", value=True, key="select_all_check", on_change=update_selections)
    current_selections = st.multiselect("Active Audit Laws:", options=all_laws, key="selected_laws")
    st.divider()
    st.error("Enforcement Cliff: June 30, 2026")

st.header("📁 Remediation & Gap Analysis Engine")
mode = st.radio("Audit Mode:", ["Web Sifter (Public Policy)", "File Upload (Private Docs)"])
org_name = st.text_input("Lead Entity", value="Synchron")

# CONTENT HANDLING
content = ""
if mode == "Web Sifter (Public Policy)":
    if st.button("🔍 Sift Public Web"):
        content = web_sifter(org_name)
        st.success("Public Policy found.")
else:
    f = st.file_uploader("Upload Policy PDF", type=['pdf'])
    if f: content = extract_pdf_text(f)

# THE AUDIT
if st.button("🛠️ Run Remediation Audit") and content:
    with st.status(f"Auditing {org_name}..."):
        # Audit logic here
        st.session_state.audit_results = llm.invoke(f"Audit {org_name} against {current_selections}. Content: {content[:4000]}").content
        if "human" in st.session_state.audit_results.lower(): st.session_state.hole_type = "Human Appeal"
        st.markdown(st.session_state.audit_results)

# PDF EXPORT
if st.session_state.audit_results:
    pdf = generate_pdf_report(st.session_state.audit_results, org_name, st.session_state.hole_type, current_selections)
    st.download_button("📥 Download PDF Report", data=pdf, file_name=f"{org_name}_Audit.pdf")
