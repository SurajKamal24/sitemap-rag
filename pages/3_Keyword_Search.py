from vectorstore.chroma import ChromaVectorStore
from main import *
import streamlit as st
import os, json

CONFIG_FILE = "rag_config.json"

st.set_page_config(
    page_title="AI Powered Keyword Search",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.header('Keyword-Based Search Engine')

st.markdown("""
    Welcome to the **Keyword-Based Search Engine**, designed to help you find content containing specific keywords from your dataset.
    Unlike semantic search, this tool focuses on **exact or partial keyword matches**, making it ideal for quick document lookups or filtering content based on terms of interest.
""")
st.markdown("---")

if not os.path.exists(CONFIG_FILE):
    st.error("Configuration not found! Please configure settings first.")
    st.stop()

with open(CONFIG_FILE, 'r') as f:
    config = json.load(f)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("##### Enter Your Keywords")
    st.markdown("""
        Provide one or more keywords to search.
    """)

    sample_keywords = [
        "Acquisition Excellence",
        "Small Business Programs",
        "CAAC Letters",
        "Encore III Contract",
        "Category Management"
    ]

    keyword_input = st.text_input(
        label="Keywords",
        value=st.session_state.get("keyword_search", ""),
        placeholder="Type keywords here...",
        key="keyword_search"
    )

    selected_sample = st.selectbox(
        "Or choose a sample keyword set:",
        options=["Select sample"] + sample_keywords,
        index=0
    )

    if selected_sample != "Select sample":
        keyword_input = selected_sample

    st.markdown("\n")

with col2:
    st.markdown("##### Search Configuration")
    st.markdown("""
        Apply content filters or adjust the number of results to refine your search.
    """)

    regulations = st.multiselect(
        "Select Optional Content Filters (if any):",
        ["content"]
    )

    top_k_results = st.selectbox("Number of Results to Retrieve:", [5, 10, 15, 20], index=1)

    st.markdown("\n")

st.markdown("<br>", unsafe_allow_html=True)
center_col = st.columns([2, 1, 2])
with center_col[1]:
    search_clicked = st.button("Start Keyword Search", use_container_width=True)

if search_clicked:
    with st.spinner(f"Performing keyword search for: **{keyword_input}**"):
        embedding_model = initialize_embedding_model(config['embedding_model']).model
        vector_store = initialize_vector_store(
            config['vector_store'], 
            config['collection_name'], 
            config['persist_dir'], 
            embedding_model
        )
        response = keyword_search(
            vector_store,
            text=keyword_input,
            filter=regulations,
            top_k=top_k_results
        )

        st.markdown("##### Search Results")
        for idx, item in enumerate(response, 1):
            st.markdown(f"**{idx}. [{item['title']}]({item['url']})**")
            st.write(item['short_content'])
            st.write("---")
