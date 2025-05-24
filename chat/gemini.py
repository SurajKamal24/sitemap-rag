from chat.base import BaseChat
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from prompts.prompts import Templates
from config.config import Config
import os
from utils.logger import get_logger

logger = get_logger(__name__)

class GeminiChat(BaseChat):
    """Gemini-powered chat implementation using ChatGoogleGenerativeAI."""

    def __init__(self, temperature=Config.DEFAULT_TEMPERATURE, max_tokens=Config.DEFAULT_MAX_TOKENS):

        super().__init__(model_name=Config.GEMINI_CHAT_MODEL) 

        if "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = Config.GEMINI_API_KEY
        self.model = ChatGoogleGenerativeAI(model=self.model_name, temperature=temperature, max_tokens=max_tokens)

    def generate_response(self, query, context):
        """Generate a structured response """

        logger.info("Generating response using Gemini chat model")

        formatted_prompt = Templates.QA_PROMPT.format(context=context, question=query)
        raw_response = self.model.invoke(formatted_prompt)

        logger.debug(f"Raw response from Gemini chat model: \n{raw_response}")

        extracted_text = (
            raw_response.content.strip() if hasattr(raw_response, "content") and isinstance(raw_response.content, str)
            else "I'm sorry, but I couldn't generate a response for this query."
        )

        return extracted_text
    
    def generate_response_with_history(self, query, context, session_id):
        """Generate a structured response with chat session history"""

        logger.info("Generating response with chat session history using Gemini chat model")
        
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

        logger.debug(f"Raw response from Gemini chat model: \n{raw_response}")

        extracted_text = (
            raw_response.content.strip() if hasattr(raw_response, "content") and isinstance(raw_response.content, str)
            else "I'm sorry, but I couldn't generate a response for this query."
        )

        return extracted_text

    def refine_query_with_history(self, user_query, session_id):
    
        logger.info(f"Refining query using chat history for session: {session_id}")

        chat_history = self.get_message_history(session_id).messages
        if not chat_history:
            logger.info("No chat history available. Using original query.")
            return user_query 
        
        formatted_history = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history[-5:]])

        formatted_prompt = Templates.REFINEMENT_PROMPT.format(
            history=formatted_history,
            query=user_query
        )

        refined_query = self.model.invoke(formatted_prompt).content.strip()

        logger.info(f"Refined Query: {refined_query}")
        return refined_query

if __name__ == "__main__":
    gemini_chat = GeminiChat(temperature=0.7, max_tokens=100)
    query = "What is Machine Learning?"
    context = None
    response = gemini_chat.generate_response(query, context)

    if response:
        print("\nGemini chat response generated successfully!")
        print(response)
    else:
        print("Failed to generate Gemini chat response.")