import streamlit as st
from engine import get_llm, load_selected_docs, find_and_scrape_company

# Add a New Tab for Autonomy
tab1, tab2 = st.tabs(["📁 Manual Audit", "🤖 Autonomous Scout"])

with tab2:
    st.header("Lead Scout: Autonomous Gap Analysis")
    target_company = st.text_input("Enter Company Name (e.g., 'Neuralink' or 'Palantir')")
    
    if st.button("🔍 Find, Scrape, & Audit"):
        with st.status("Searching for public AI disclosures...") as status:
            # 1. FIND & SCRAPE
            evidence = find_and_scrape_company(target_company)
            if not evidence:
                st.error("Could not find sufficient public data for this company.")
            else:
                st.info(f"Scraped {len(evidence)} characters of public policy data.")
                
                # 2. RUN GAP ANALYSIS (Against your toggled Regulations)
                db = load_selected_docs(st.session_state.get('selected_files', []))
                search_docs = db.similarity_search("human oversight, data bias", k=5)
                reg_context = "\n".join([d.page_content for d in search_docs])
                
                prompt = f"""
                Analyze the PUBLIC POLICY of {target_company} against the REGULATORY DATABASE.
                
                REGS: {reg_context}
                PUBLIC POLICY: {evidence}
                
                FIND THE GAPS:
                1. What are they claiming that violates the EU/Colorado laws?
                2. Where is their 'Reasonable Care' missing?
                3. Draft a 'Cold Pitch' based on these specific failures.
                """
                
                report = get_llm(is_cloud, st.secrets).invoke(prompt).content
                st.markdown(report)
