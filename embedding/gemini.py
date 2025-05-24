
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from embedding.base import BaseEmbedding
from config.config import Config
from utils.logger import get_logger
import os

logger = get_logger(__name__)

class GeminiEmbeddingGenerator(BaseEmbedding):
    """Embedding generator using Gemini AI model."""
    
    def __init__(self):
        super().__init__(Config.GEMINI_EMBEDDING_MODEL)
        if "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = Config.GEMINI_API_KEY
        try:
            self.model = GoogleGenerativeAIEmbeddings(model=f"models/{Config.GEMINI_EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Error initializing GeminiEmbeddingGenerator: {e}")
        
if __name__ == "__main__":
    gemini_generator = GeminiEmbeddingGenerator()
    sample_text = "Machine Learning enables computers to learn from data."
    embedding = gemini_generator.generate_embedding(sample_text)

    if embedding:
        print("\nGemini Embedding generated successfully!")
        print(f"Embedding Length: {len(embedding)}")
        print(f"First 5 Values: {embedding[:5]}")
    else:
        print("Failed to generate Gemini embedding.")