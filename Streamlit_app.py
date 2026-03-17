import streamlit as st
import sys
import os
from engine import (
    get_llm, list_all_laws, smart_web_sifter, 
    perform_gap_analysis, generate_pdf_report, 
    extract_pdf_text, ask_regulatory_chat
)

st.set_page_config(page_title="AI-MedReg-Engine | 2026 Compliance", layout="wide")

# --- INITIALIZE SESSION STATE ---
if "audit_content" not in st.session_state: st.session_state.audit_content = ""
if "audit_results" not in st.session_state: st.session_state.audit_results = ""
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# Load core assets
llm = get_llm(st.secrets)
all_laws = list_all_laws()

# --- SIDEBAR & GLOBAL CONTROLS ---
with st.sidebar:
    st.header("🛡️ REGULATORY STACK")
    current_laws = st.multiselect("Selected Regulations:", options=all_laws, default=all_laws)
    
    st.divider()
    st.subheader("⚙️ System Controls")
    # NEW: The Debug Toggle
    debug_mode = st.checkbox("Enable Developer Debug", value=False)
    
    if st.button("🗑️ Wipe All Data"):
        st.session_state.audit_content = ""
        st.session_state.audit_results = ""
        st.session_state.chat_history = []
        st.rerun()

# --- MAIN UI ---
st.header("🔬 Remediation & Statutory Audit Engine")
org_name = st.text_input("Lead Entity", value="Synchron")
mode = st.radio("Sourcing Mode:", ["Web Sifter", "Manual Paste", "File Upload"], horizontal=True)

# --- STEP 1: CONTENT INGESTION ---
if mode == "Web Sifter":
    if st.button("🔍 Sift Public Web"):
        with st.spinner(f"Sifting web for {org_name}..."):
            st.session_state.audit_content = smart_web_sifter(org_name)
            st.toast("Web data ingested successfully!", icon="🌐")

elif mode == "Manual Paste":
    st.session_state.audit_content = st.text_area("Paste News/Policy Text:", height=200)

elif mode == "File Upload":
    f = st.file_uploader("Upload Policy PDF", type=['pdf'])
    if f: st.session_state.audit_content = extract_pdf_text(f)

# --- DEBUG BLOCK: CONTENT CHECK ---
if debug_mode:
    with st.expander("🛠️ DEBUG: Raw Ingested Content", expanded=False):
        if st.session_state.audit_content:
            st.text(st.session_state.audit_content)
        else:
            st.info("No content ingested yet.")

# --- STEP 2: THE AUDIT ---
if st.button("🛠️ Run Statutory Audit"):
    if st.session_state.audit_content and current_laws:
        with st.status("🔍 Analyzing against 2026 Statutory Cliff...") as status:
            st.session_state.audit_results = perform_gap_analysis(
                st.session_state.audit_content, current_laws, org_name, llm
            )
            status.update(label="✅ Audit Complete!", state="complete")
    else:
        st.warning("Please ensure content is loaded and laws are selected.")

# Display Results
if st.session_state.audit_results:
    st.markdown("### 📋 Audit Findings")
    st.info(st.session_state.audit_results)
    
    pdf = generate_pdf_report(st.session_state.audit_results, org_name, current_laws)
    st.download_button("📥 Download Remediation PDF", data=pdf, file_name=f"{org_name}_Audit_2026.pdf")

# --- DEBUG BLOCK: RESULTS CHECK ---
if debug_mode:
    with st.expander("🛠️ DEBUG: Session State Keys", expanded=False):
        st.write(st.session_state.to_dict())

# --- STEP 3: REGULATORY Q&A ---
st.divider()
st.subheader("💬 Regulatory Consultation")

# Render history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
if user_query := st.chat_input("How do we resolve the Colorado SB 24-205 gaps?"):
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"): st.markdown(user_query)

    with st.chat_message("assistant"):
        if not st.session_state.audit_results:
            response = "⚠️ Please **Run the Statutory Audit** first so I have the necessary context to assist you."
        else:
            with st.spinner("Consulting 2026 database..."):
                response = ask_regulatory_chat(user_query, st.session_state.audit_results, llm)
        
        st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
