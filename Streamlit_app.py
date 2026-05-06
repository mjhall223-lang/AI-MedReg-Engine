import streamlit as st
from engine import (
    get_llm, list_all_laws, smart_web_sifter, scrape_url,
    perform_gap_analysis, generate_pdf_report, 
    extract_pdf_text, ask_regulatory_chat
)

st.set_page_config(page_title="AI-MedReg-Engine | 2026 Compliance", layout="wide")

# --- SESSION STATE ---
if "audit_content" not in st.session_state: st.session_state.audit_content = ""
if "audit_results" not in st.session_state: st.session_state.audit_results = ""
if "chat_history" not in st.session_state: st.session_state.chat_history = []

llm = get_llm(st.secrets)
all_laws = list_all_laws()

with st.sidebar:
    st.header("🛡️ REGULATORY STACK")
    current_laws = st.multiselect("Selected Regulations:", options=all_laws, default=all_laws)
    st.divider()
    debug_mode = st.checkbox("Enable Developer Debug", value=False)
    if st.button("🗑️ Wipe All Data"):
        st.session_state.clear()
        st.rerun()

st.header("🔬 Remediation & Statutory Audit Engine")
org_name = st.text_input("Lead Entity", value="Synchron")
mode = st.radio("Sourcing Mode:", ["Web Sifter", "Link Paste", "Manual Paste", "File Upload"], horizontal=True)

# --- INGESTION LOGIC ---
if mode == "Web Sifter" and st.button("🔍 Sift Web"):
    with st.spinner("Searching for entity data..."):
        st.session_state.audit_content = smart_web_sifter(org_name)
    st.toast("Data Ingested.")

elif mode == "Link Paste":
    target_url = st.text_input("Paste Regulatory/Policy URL:")
    if st.button("🔗 Fetch Content") and target_url:
        with st.spinner("Scraping target URL..."):
            st.session_state.audit_content = scrape_url(target_url)
        st.success("Content Loaded.")

elif mode == "Manual Paste":
    st.session_state.audit_content = st.text_area("Paste News/Policy Text:", height=200)

elif mode == "File Upload":
    f = st.file_uploader("Upload Policy PDF", type=['pdf'])
    if f: st.session_state.audit_content = extract_pdf_text(f)

if debug_mode and st.session_state.audit_content:
    with st.expander("🛠️ DEBUG: Ingested Content"): st.text(st.session_state.audit_content)

# --- AUDIT EXECUTION ---
if st.button("🛠️ Run Statutory Audit"):
    if st.session_state.audit_content:
        with st.status("🔍 Analyzing against Regulatory Stack...") as status:
            st.session_state.audit_results = perform_gap_analysis(st.session_state.audit_content, current_laws, org_name, llm)
            status.update(label="✅ Audit Complete!", state="complete")
    else:
        st.warning("Please load content using one of the sourcing modes above.")

if st.session_state.audit_results:
    st.markdown("### 📋 Audit Findings")
    st.info(st.session_state.audit_results)
    pdf = generate_pdf_report(st.session_state.audit_results, org_name, current_laws)
    st.download_button("📥 Download PDF Report & Remediation Plan", data=pdf, file_name=f"{org_name}_Audit_2026.pdf")

# --- CONSULTATION ---
st.divider()
st.subheader("💬 Regulatory Consultation")
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if user_query := st.chat_input("Ask a follow-up about these findings..."):
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"): st.markdown(user_query)
    with st.chat_message("assistant"):
        response = ask_regulatory_chat(user_query, st.session_state.audit_results, llm) if st.session_state.audit_results else "Run audit first."
        st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
