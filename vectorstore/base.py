from abc import abstractmethod

class BaseVectorStore:
    """Abstract base class for vector store management."""

    def __init__(self, collection_name, persist_directory, embedding_model):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.vectorstore = None

    @abstractmethod
    def store_documents(self, documents):
        """Store documents in the vector store."""
        pass

    @abstractmethod
    def query_similar(self, query_text, top_k=5):
        """Retrieve similar documents from the vector store."""
        pass

    @abstractmethod
    def delete_collection(self):
        """Delete the vector store collection."""
        pass

    @abstractmethod
    def list_collections(self):
        """List all collections in the vector store."""
        pass

    @abstractmethod
    def iterate_over_collection(self, collection_name):
        """Iterate over all documents in a collection."""
        pass