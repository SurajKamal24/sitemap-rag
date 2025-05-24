from vectorstore.chroma import ChromaVectorStore
from langchain_core.documents import Document
from chat.llama import LlamaChat
from chat.gemini import GeminiChat
from main import *
import streamlit as st
import os, json

CONFIG_FILE = "rag_config.json"

st.set_page_config(
    page_title="AI Powered Search Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.header('Semantic Search Engine')

st.markdown("""
    Welcome to the **Semantic Search Engine**, designed to help you retrieve highly relevant content from your data.
    Unlike traditional keyword search, **semantic search** understands the intent behind your query and fetches contextually accurate information.
    This is particularly powerful for navigating large knowledge bases, regulations, or policy documents.
""")
st.markdown("---")

if not os.path.exists(CONFIG_FILE):
    st.error("Configuration not found! Please configure settings first.")
    st.stop()

with open(CONFIG_FILE, 'r') as f:
    config = json.load(f)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("##### Enter Your Search Query")
    st.markdown("""
        Provide a natural language question to search. Semantic Search retrieves the most meaningful responses even if specific keywords are missing.
    """)

    sample_questions = [
        "What is the purpose of the Acquisition Excellence and Small Business Excellence Awards?",
        "Who is eligible for the Acquisition Excellence Award?",
        "How can a user view the content of a specific CAAC letter?",
        "Which organization received the CAOC Small Business Excellence Award?"
    ]

    query_text = st.text_input(
        label="Search Query",
        value=st.session_state.get("search_question", ""),
        placeholder="Type your question here",
        key="search_question"
    )

    selected_sample = st.selectbox(
        "Or choose a sample question:",
        options=["Select a question"] + sample_questions,
        index=0
    )

    if selected_sample != "Select a question":
        query_text = selected_sample

    st.markdown("\n")


with col2:
    st.markdown("##### Search Configuration")
    st.markdown("""
        Apply filters or set the number of results to refine your search and control the depth of retrieved information.
    """)

    regulations = st.multiselect(
        "Select Optional Content Filters (if any):", 
        ["content"]
    )

    top_k_results = st.selectbox("Number of Results to Retrieve:", [5, 10, 15, 20], index=1)

    st.markdown("\n")


st.markdown("##### Language Model Settings")
st.markdown("""
    Tune how the AI responds to your query. You can select the model and tweak its behavior:
        - **Max Tokens** controls the response length.
        -  **Temperature** adjusts creativity vs factuality. Lower is more factual.
    """
)

c_col1, c_col2, c_col3 = st.columns([1, 1, 1])

with c_col1:
    model_choice = st.selectbox("Choose Chat Model:", ["Llama", "Gemini"])
    st.caption("Choose between available language models. "
           "**Llama** is fast and efficient for general queries. "
           "**Gemini** provides more detailed, context-aware responses ideal for complex questions.")

with c_col2:
    max_tokens = st.number_input(
        "Max Output Tokens",
        min_value=50,
        max_value=500,
        value=250,
        step=10
    )
    st.caption("The maximum number of tokens to generate. Increase for longer responses.")

with c_col3:
    creativity = st.slider(
        "Creativity (Temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1
    )
    st.caption("Higher values allow more creativity. Lower values make answers more deterministic.")

st.markdown("<br>", unsafe_allow_html=True)
center_col = st.columns([2, 1, 2])
with center_col[1]:
    search_clicked = st.button("Start Semantic Search", use_container_width=True)


if search_clicked:
    with st.spinner(f"Performing semantic search for: **{query_text}**"):
        embedding_model = initialize_embedding_model(config['embedding_model']).model
        vector_store = initialize_vector_store(
            config['vector_store'], 
            config['collection_name'], 
            config['persist_dir'], 
            embedding_model
        )
        chat_model = initialize_chat_model(model_choice.lower(), temperature=creativity, max_tokens=max_tokens)
        response = semantic_search(
            vector_store,
            chat_model,
            query=query_text,
            filter=None,
            session_id=None,
            mode="search"
        )


        st.markdown("##### Semantic Search Results")

        st.caption("Below are the most relevant results retrieved based on your query. "
           "Each result includes a title (clickable link) and a brief content preview for context.")
        
        st.markdown("**Content:**")
        st.markdown(response["response_text"])

        if response["references"]:
            st.markdown("**References:**")
            for idx, ref in enumerate(response["references"], 1):
                st.markdown(f"**{idx}. [{ref['title']}]({ref['url']})** (Score: {ref['score']:.4f})")
        else:
            st.info("No references found.")

        st.caption("Click on the titles to view the full content. The short description provides context from the matched document.")

