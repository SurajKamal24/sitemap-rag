from embedding.nomic import NomicEmbeddingGenerator
from embedding.gemini import GeminiEmbeddingGenerator
from vectorstore.chroma import ChromaVectorStore
import streamlit as st
import os, time, json
from main import *

CONFIG_FILE = "rag_config.json"

st.set_page_config(
    page_title="AI Powered Search Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.header("Sitemap RAG Pipeline Configuration")

with st.form("rag_settings_form"):
    sitemap_url = st.text_input(
        "Enter your website sitemap URL:",
        placeholder="https://yourwebsite.com/sitemap.xml"
    )

    enable_filter = st.checkbox("Enable URL Filtering from Sitemap?", value=False)

    filter_pattern = st.text_input(
        "Enter regex pattern to filter URLs (required if filtering enabled):",
        placeholder="Example: .*category/innovation.*"
    )

    st.markdown("##### Embedding & Vector Store Settings")
    embedding_model = st.selectbox(
        "Select Embedding Model:",
        ["nomic", "gemini"]
    )

    vector_store = st.selectbox(
        "Select Vector Store:",
        ["chroma"]
    )

    persist_dir = st.text_input(
        "Enter Vector Store Persist Directory Path:",
        placeholder="./vector_store"
    )

    collection_name = st.text_input(
        "Enter Collection Name to Store Embeddings:",
        placeholder="my_collection"
    )

    # Submit button
    submitted = st.form_submit_button("Save Settings")

if submitted:
    # Example: You can store this config in session_state or file
    config_data = {
        "sitemap_url": sitemap_url,
        "filter_enabled": enable_filter,
        "filter_pattern": filter_pattern,
        "embedding_model": embedding_model,
        "vector_store": vector_store,
        "persist_dir": persist_dir,
        "collection_name": collection_name
    }

    with open(CONFIG_FILE, 'w') as f:
        json.dump(config_data, f, indent=4)

    st.success("Settings saved and persisted to rag_config.json!")
    st.json(config_data)


if os.path.exists(CONFIG_FILE):
    if st.button("Start Processing"):
        with st.spinner("Processing... Please wait while we load your data."):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)

            embedding_model = initialize_embedding_model(config['embedding_model']).model
            vector_store = initialize_vector_store(
                config['vector_store'], 
                config['collection_name'], 
                config['persist_dir'], 
                embedding_model
            )
            load_data(
                config['sitemap_url'],
                vector_store, 
                Config.BATCH_SIZE, 
                config['filter_enabled'],
                config['filter_pattern'],
            )

        st.success("Data processing completed successfully!")

