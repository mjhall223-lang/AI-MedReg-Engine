import streamlit as st
import os
import tempfile
from engine import get_llm, find_and_scrape_live_news, EconomicImpact, create_pdf, load_selected_docs

st.set_page_config(page_title="ReadyAudit Engine", layout="wide", page_icon="⚖️")
st.title("⚖️ ReadyAudit: Live-News Liability Engine")

is_cloud = st.secrets.get("GROQ_API_KEY") is not None

# --- SIDEBAR: THE LIVE CALCULATOR ---
with st.sidebar:
    st.header("🛡️ SPECIALIST PANEL")
    st.info(f"Today's Date: March 15, 2026")
    
    st.markdown("### 📈 LIABILITY BEAST")
    tokens = st.number_input("Est. Monthly Tokens (M):", value=50.0)
    # If news shows 4,000 layoffs (like Block), enter that here!
    replaced = st.number_input("Affected Personnel:", value=10)
    
    impact = EconomicImpact.calculate_liability(tokens*1000000, replaced)
    st.metric("Statutory Risk", f"${impact['statutory']:,}", delta="Per CO SB 24-205", delta_color="inverse")
    st.metric("Total Governance Debt", f"${impact['total']:,}")
    st.session_state.impact_total = impact['total']

    st.markdown("---")
    st.markdown("### 📜 ACTIVE FRAMEWORKS")
    all_pdfs = [f for r, d, files in os.walk("Regulations") for f in files if f.endswith(".pdf")]
    selected_files = [f for f in sorted(list(set(all_pdfs))) if st.checkbox(f"📄 {f}", value=True)]

tab1, tab2 = st.tabs(["📁 Deep Audit", "🤖 Autonomous Hunter"])

# [Tab 1 handles manual PDF uploads as before]

with tab2:
    st.header("Lead Hunter: Real-Time Prospecting")
    co_name = st.text_input("Enter Target Company (e.g., 'Block', 'Synchron', 'PayPal')")
    
    if st.button("🔍 Scout Live News & Pitch"):
        with st.status("Scraping March 2026 headlines...") as s:
            news_data = find_and_scrape_live_news(co_name, st.secrets.get("TAVILY_API_KEY"))
            
            # The prompt now FORCES the AI to mention real news it just found
            prompt = f"""
            You are a Regulatory Specialist. Date: March 15, 2026.
            
            1. Analyze this LIVE NEWS for {co_name}: {news_data}
            2. Link a specific headline (like a layoff, product launch, or partnership) to the June 30, 2026 Colorado AI Act deadline.
            3. Use the calculated liability of ${st.session_state.impact_total:,.2f} as the hook.
            4. Mention 'Neural Data' risk if the news mentions BCI or biometrics.
            5. Draft a cold pitch that sounds like you just read the news this morning.
            """
            
            report = get_llm(is_cloud, st.secrets).invoke(prompt).content
            st.session_state.report = report
            st.markdown(report)

if "report" in st.session_state:
    st.download_button("📩 Download Professional Report", create_pdf(st.session_state.report), file_name="Audit.pdf")
