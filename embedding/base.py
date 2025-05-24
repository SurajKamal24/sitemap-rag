from utils.logger import get_logger

logger = get_logger(__name__)

class BaseEmbedding:
    """Base class for embedding generators."""

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
    
    def generate_embedding(self, text):
        """Generate an embedding for a single piece of text."""
        try:
            if not self.model:
                self.initialize_model()
            embedding = self.model.embed_documents([text])
            return embedding[0] if embedding else None
        except Exception as e:
            logger.error(f"Error generating embedding with {self.model_name}: {e}")
            return None