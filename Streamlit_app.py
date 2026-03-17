import streamlit as st
import sys
import os
from engine import (
    get_llm, list_all_laws, smart_web_sifter, 
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
    if st.button("🗑️ Reset Chat"):
        st.session_state.chat_history = []
        st.rerun()

st.header("🔬 Remediation & Statutory Audit Engine")
org_name = st.text_input("Lead Entity", value="Synchron")
mode = st.radio("Sourcing Mode:", ["Web Sifter", "Manual Paste", "File Upload"], horizontal=True)

# --- INGESTION ---
if mode == "Web Sifter" and st.button("🔍 Sift Web"):
    st.session_state.audit_content = smart_web_sifter(org_name)
    st.success("Ingested.")
elif mode == "Manual Paste":
    st.session_state.audit_content = st.text_area("Paste Text:", height=200)
elif mode == "File Upload":
    f = st.file_uploader("Upload PDF", type=['pdf'])
    if f: st.session_state.audit_content = extract_pdf_text(f)

# --- AUDIT ---
if st.button("🛠️ Run Statutory Audit"):
    if st.session_state.audit_content:
        with st.spinner("Analyzing..."):
            st.session_state.audit_results = perform_gap_analysis(st.session_state.audit_content, current_laws, org_name, llm)
    else:
        st.warning("Load content first.")

if st.session_state.audit_results:
    st.markdown("### 📋 Audit Findings")
    st.info(st.session_state.audit_results)
    pdf = generate_pdf_report(st.session_state.audit_results, org_name, current_laws)
    st.download_button("📥 Download PDF", data=pdf, file_name=f"{org_name}_Audit.pdf")

# --- NEW: REGULATORY Q&A ---
st.divider()
st.subheader("💬 Regulatory Consultation")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Ask about specific 2026 compliance steps..."):
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"): st.markdown(user_query)

    with st.chat_message("assistant"):
        if not st.session_state.audit_results:
            response = "Please run the audit above first so I have context."
        else:
            response = ask_regulatory_chat(user_query, st.session_state.audit_results, current_laws, llm)
        st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
