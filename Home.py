import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="AI Powered Search Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.header("AI Powered Search Engine for Your Websites")

st.markdown("""
Welcome to your **Intelligent AI Search Assistant** powered by **Retrieval-Augmented Generation (RAG)**.  
This application transforms your website data into a powerful AI-driven search and chat experience, enabling:
- **Semantic Search**: Understands meaning, not just keywords.
- **Keyword Search**: Retrieves exact term matches swiftly.
- **AI Chat Assistant**: Engages in human-like conversation based on your website knowledge base.

Explore, extract insights, and offer an AI-powered search experience to your users.
""")

st.markdown("---")

with st.expander("**How Does This Work?**", expanded=True):
    st.write("""
**RAG Architecture** ensures contextual, accurate responses by combining retrieval from your data and generative AI models:
- **Data Crawling**: We process your sitemap, extracting website content (pages, articles, policies).
- **Embedding Generation**: Using models like **Gemini** or **Nomic**, your content is converted into semantic vectors.
- **Vector Database**: Stored in **ChromaDB** for high-speed similarity searches.
- **AI Models**: LLMs like **Gemini** or **Llama** generate human-like, contextually accurate answers.
- **Streamlit Interface**: Easy-to-use UI for performing searches and AI chat.
""")

with st.expander("**High-Level System Design**", expanded=True):
    st.write("""
Our modular system includes:
- **Data Ingestion Layer**: Extracts and preprocesses data.
- **Embedding Layer**: Converts text into high-dimensional vectors.
- **Vector Storage (ChromaDB)**: Enables semantic similarity search.
- **Chat Assistant Layer**: Handles user interaction with LLM-powered responses.
- **Frontend**: Provides Semantic Search, Keyword Search, and Chat modes.
""")

st.markdown("---")
st.markdown("### **Start Exploring:**")

# Navigation Buttons (Streamlit Native or use st.page_link if multipage)
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button(" Chat Assistant"):
        st.switch_page("pages/1_Chat_Assistant.py")

with col2:
    if st.button(" Semantic Search"):
        st.switch_page("pages/2_Semantic_Search.py")

with col3:
    if st.button(" Keyword Search"):
        st.switch_page("pages/3_Keyword_Search.py")

with col4:
    if st.button(" Settings"):
        st.switch_page("pages/4_Settings.py")

st.markdown("""
---  
Each module is designed to help you navigate your website's data:
- **Chat Assistant**: Conversational AI driven by your content.
- **Semantic Search**: Retrieves contextually relevant answers, beyond keyword matching.
- **Keyword Search**: Classic search for fast, direct results.
- **Settings**: Customize AI model behavior, control tokens, and adjust creativity.
""")
