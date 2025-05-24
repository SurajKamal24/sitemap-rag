from chat.base import BaseChat
from langchain_ollama import ChatOllama
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from prompts.prompts import Templates
from config.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

class LlamaChat(BaseChat):
    """Llama-powered chat implementation using ChatOllama."""

    def __init__(self, temperature=Config.DEFAULT_TEMPERATURE, max_tokens=Config.DEFAULT_MAX_TOKENS):

        super().__init__(model_name=Config.LLAMA_CHAT_MODEL)
        logger.info(self.model_name)
        self.model = ChatOllama(model=self.model_name, temperature=temperature, num_predict=max_tokens)

    def generate_response(self, query, context):
        """Generate a structured response """

        logger.info("Generating response using Llama chat model")

        formatted_prompt = Templates.QA_PROMPT.format(context=context, question=query)
        raw_response = self.model.invoke(formatted_prompt)

        logger.debug(f"Raw response from Llama chat model: \n{raw_response}")

        extracted_text = (
            raw_response.content.strip() if hasattr(raw_response, "content") and isinstance(raw_response.content, str)
            else "I'm sorry, but I couldn't generate a response for this query."
        )

        return extracted_text

    def generate_response_with_history(self, query, context, session_id):
        """Generate a structured response """

        logger.info("Generating response with chat session history using Llama chat model")
        
        selected_prompt = self.select_prompt(context)
        inputs = {"question": query}

        logger.info(f"Using chat history for session: {session_id}")

        chat_history = self.get_message_history(session_id)
            
        logger.debug(f"Chat history for session: {chat_history}")

        full_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                MessagesPlaceholder(variable_name="history"),
                ("human", selected_prompt)
            ]
        )

        chain_with_history = RunnableWithMessageHistory(
            full_prompt_template | self.model,
            lambda _: chat_history,
            input_messages_key="question",
            history_messages_key="history",
        )

        raw_response = chain_with_history.invoke(inputs, {"configurable": {"session_id": session_id}})

        logger.debug(f"Raw response from Llama chat model: \n{raw_response}")

        extracted_text = (
            raw_response.content.strip() if hasattr(raw_response, "content") and isinstance(raw_response.content, str)
            else "I'm sorry, but I couldn't generate a response for this query."
        )

        return extracted_text

if __name__ == "__main__":
    gemini_chat = LlamaChat(temperature=0.7, max_tokens=100)
    query = "What is Machine Learning?"
    context = None
    response = gemini_chat.generate_response(query, context)

    if response:
        print("\nGemini chat response generated successfully!")
        print(response)
    else:
        print("Failed to generate Gemini chat response.")