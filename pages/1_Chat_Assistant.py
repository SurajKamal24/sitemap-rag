import uuid
import streamlit as st
from chat.gemini import GeminiChat
from config.config import Config
from chat.gemini import GeminiChat
from main import *
import os, json

CONFIG_FILE = "rag_config.json"

st.set_page_config(
    page_title="AI Powered Chat Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.header(' AI Chat Assistant')
st.markdown("""
    Welcome to the **AI Chat Assistant** powered by large language models. 
    This assistant can answer your questions, guide you through complex topics, or help you search knowledge bases.
    You can customize the model behavior and create new chat sessions as needed.
""")

if not os.path.exists(CONFIG_FILE):
    st.error("Configuration not found! Please configure settings first.")
    st.stop()

with open(CONFIG_FILE, 'r') as f:
    config = json.load(f)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

if not st.session_state.messages:
    greeting_text = "**Hello!** I'm your AI Assistant, here to support your search and answer your questions. How can I help?"
    st.session_state.messages.append({"role": "assistant", "content": greeting_text})

col_left, col_right = st.columns([3, 1])


with col_left:
    with st.expander("Language Model Settings", expanded=True):
        model_choice = st.selectbox("Choose Chat Model:", ["Gemini"], index=0)
        st.caption("Choose between available language models. "
                   "**Gemini** provides detailed, context-aware responses ideal for complex questions.")

        max_tokens = st.number_input("Max Output Tokens", min_value=50, max_value=500, value=Config.DEFAULT_MAX_TOKENS, step=10)
        st.caption("The maximum number of tokens to generate. Increase for longer, more detailed responses.")

        temperature = st.slider("Creativity (Temperature)", min_value=0.0, max_value=1.0, value=Config.DEFAULT_TEMPERATURE, step=0.1)
        st.caption("Higher values allow more creativity and variation. Lower values make answers more focused and deterministic.")

with col_right:
    if st.button("Start New Chat Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state["session_id"] = str(uuid.uuid4())
        st.success("Started a new chat session!")

        greeting_text = "*Welcome!** I am your AI-powered assistant, ready to help you with your queries. How can I assist you today?"
        st.session_state.messages.append({"role": "assistant", "content": greeting_text})

    st.caption("""
        Starting a new chat session clears the current conversation and creates a new session ID.
        Useful for starting fresh with new context.
    """)

st.markdown("---")

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input("Ask your question or start chatting..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    embedding_model = initialize_embedding_model(config['embedding_model']).model
    vector_store = initialize_vector_store(
        config['vector_store'], 
        config['collection_name'], 
        config['persist_dir'], 
        embedding_model
    )
    chat_model = initialize_chat_model(model_choice.lower(), temperature=temperature, max_tokens=max_tokens)

    with st.spinner("Thinking..."):
        response = semantic_search(
            vector_store=vector_store, 
            chat_model=chat_model,
            query=user_input,
            filter=None,
            session_id=st.session_state["session_id"],
            mode="chat"
        )
    st.session_state.messages.append({"role": "assistant", "content": response["response_text"]})
    st.chat_message("assistant").write(response["response_text"])