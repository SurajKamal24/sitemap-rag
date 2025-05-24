from abc import ABC, abstractmethod
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory
from prompts.prompts import Templates
from config.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

class BaseChat(ABC):
    """Abstract base class for chat models."""

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    def get_message_history(self, session_id):
        return SQLChatMessageHistory(
            session_id=session_id, connection_string=Config.CHAT_HISTORY_DB_URI
        )
    
    def select_prompt(self, context):
        if context:
            logger.info("Using context-based prompt.")
            return Templates.CONTEXTUAL_PROMPT.format(context=context, question="{question}")
        else:
            logger.info("Using general knowledge prompt.")
            return Templates.GENERAL_PROMPT.format(question="{question}")
        
    @abstractmethod
    def generate_response(self, query, context):
        """
        Generate a response based on the query and context.
        This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def generate_response_with_history(self, query, context, session_id):
        """
        Generate a response based on the query and context.
        This method must be implemented by subclasses.
        """
        pass