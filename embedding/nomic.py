from langchain_ollama import OllamaEmbeddings
from config.config import Config
from embedding.base import BaseEmbedding
from utils.logger import get_logger

logger = get_logger(__name__)

class NomicEmbeddingGenerator(BaseEmbedding):
    """Embedding generator using Nomic (Ollama) model."""
    
    def __init__(self):
        super().__init__(Config.NOMIC_EMBEDDING_MODEL)
        try:
            self.model = OllamaEmbeddings(model=self.model_name)
            logger.info(f"Nomic embedding model initialized successfully '{self.model}'.")
        except Exception as e:
            logger.error(f"Error initializing NomicEmbeddingGenerator: {e}")
        
if __name__ == "__main__":
    nomic_generator = NomicEmbeddingGenerator()
    sample_text = "Artificial Intelligence is transforming the world."
    embedding = nomic_generator.generate_embedding(sample_text)

    if embedding:
        print("\nNomic Embedding generated successfully!")
        print(f"Embedding Length: {len(embedding)}")
        print(f"First 5 Values: {embedding[:5]}")
    else:
        print("Failed to generate Nomic embedding.")