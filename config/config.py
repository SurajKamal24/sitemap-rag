from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    """Configuration class for loading environment variables."""
    SITEMAP_URL = os.getenv("SITEMAP_URL")
    BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
    FILTER_URLS = bool(os.getenv("FILTER_URLS"))
    FILTER_PATTERN = os.getenv("FILTER_PATTERN")

    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")
    CHROMA_DB_COLLECTION = os.getenv("CHROMA_DB_COLLECTION")

    NOMIC_EMBEDDING_MODEL = os.getenv("NOMIC_EMBEDDING_MODEL")
    NOMIC_API_KEY = os.getenv("NOMIC_API_KEY", "")

    GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL")
    LLAMA_CHAT_MODEL = os.getenv("LLAMA_CHAT_MODEL")

    DEFAULT_VECTOR_STORE = os.getenv("DEFAULT_VECTOR_STORE")
    DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL")
    DEFAULT_CHAT_MODEL = os.getenv("DEFAULT_CHAT_MODEL")
    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE"))
    DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS"))
    
    CHAT_HISTORY_DB_URI = os.getenv("CHAT_HISTORY_DB_URI")

    SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD"))

    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS"))

    @staticmethod
    def display_config():
        """Display the current configuration for debugging purposes."""
        print(f" SITEMAP_URL: {Config.SITEMAP_URL}")
        print(f" BATCH_SIZE: {Config.BATCH_SIZE}")
        print(f" FILTER_URLS: {Config.FILTER_URLS}")
        print(f" FILTER_PATTERN: {Config.FILTER_PATTERN}")

        print(f" CHROMA_DB_PATH: {Config.CHROMA_DB_PATH}")
        print(f" CHROMA_DB_COLLECTION: {Config.CHROMA_DB_COLLECTION}")

        print(f" NOMIC_EMBEDDING_MODEL: {Config.NOMIC_EMBEDDING_MODEL}")
        print(f" NOMIC_API_KEY Set: {'Yes' if Config.NOMIC_API_KEY else 'No'}")

        print(f" GEMINI_EMBEDDING_MODEL: {Config.GEMINI_EMBEDDING_MODEL}")
        print(f" GEMINI_API_KEY Set: {'Yes' if Config.GEMINI_API_KEY else 'No'}")

        print(f" GEMINI_CHAT_MODEL: {Config.GEMINI_CHAT_MODEL}")
        print(f" LLAMA_CHAT_MODEL: {Config.LLAMA_CHAT_MODEL}")

        print(f" DEFAULT_VECTOR_STORE: {Config.DEFAULT_VECTOR_STORE}")
        print(f" DEFAULT_EMBEDDING_MODEL: {Config.DEFAULT_EMBEDDING_MODEL}")
        print(f" DEFAULT_CHAT_MODEL: {Config.DEFAULT_CHAT_MODEL}")
        print(f" DEFAULT_TEMPERATURE: {Config.DEFAULT_TEMPERATURE}")
        print(f" DEFAULT_MAX_TOKENS: {Config.DEFAULT_MAX_TOKENS}")

        print(f" CHAT_HISTORY_DB_URI: {Config.CHAT_HISTORY_DB_URI}")

        print(f" SCORE_THRESHOLD: {Config.SCORE_THRESHOLD}")

        print(f" TOP_K_RESULTS: {Config.TOP_K_RESULTS}")

if __name__ == "__main__":
    Config.display_config()
