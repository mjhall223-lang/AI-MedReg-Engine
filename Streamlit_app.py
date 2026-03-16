import streamlit as st
from engine import get_llm, list_all_laws, extract_pdf_text, web_sifter, perform_gap_analysis, generate_pdf_report

st.set_page_config(page_title="ReadyAudit: Remediation Engine", layout="wide")
llm = get_llm(st.secrets)
all_laws = list_all_laws()

# PERSISTENT TOGGLE LOGIC
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
org_name = st.text_input("Lead Entity", value="Synchron")
uploaded_file = st.file_uploader("Upload Tech/Policy File", type=['pdf', 'txt'])

if st.button("🛠️ Run Remediation Audit"):
    if uploaded_file and current_selections:
        with st.status(f"Auditing {org_name}..."):
            content = extract_pdf_text(uploaded_file) if uploaded_file.name.endswith('.pdf') else uploaded_file.read().decode("utf-8")
            st.session_state.audit_results = perform_gap_analysis(content, current_selections, org_name, llm)
            
            # Risk Coster logic
            if "human" in st.session_state.audit_results.lower(): st.session_state.hole_type = "Human Appeal"
            
            st.markdown(st.session_state.audit_results)
    else:
        st.warning("Upload file and toggle laws to begin.")

# EXPORT
if st.session_state.audit_results:
    pdf_buffer = generate_pdf_report(st.session_state.audit_results, org_name, st.session_state.hole_type, current_selections)
    st.download_button("📥 Download Professional Remediation Report", data=pdf_buffer, file_name=f"{org_name}_Audit_2026.pdf", mime="application/pdf")
