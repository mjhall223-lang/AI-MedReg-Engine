import streamlit as st
from engine import list_all_laws, get_llm

st.set_page_config(page_title="ReadyAudit 2026", layout="wide")

# Boot the laws
all_laws = list_all_laws()

# Safety Check: If laws are empty, show a warning instead of a blank screen
if not all_laws:
    st.error("⚠️ No PDFs found in /Regulations. Check your folder nesting.")
    st.stop()

# CALLBACK for Toggle logic
def update_selections():
    if st.session_state.select_all_check:
        st.session_state.selected_laws = all_laws
    else:
        st.session_state.selected_laws = []

# Sidebar UI
with st.sidebar:
    st.header("🛡️ LAW DATABASE")
    st.checkbox("Select All Laws", value=True, key="select_all_check", on_change=update_selections)
    current_selections = st.multiselect("Active Audit Laws:", options=all_laws, key="selected_laws")
