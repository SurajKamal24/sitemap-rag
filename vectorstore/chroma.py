from vectorstore.base import BaseVectorStore
from langchain_core.documents import Document
from langchain_chroma import Chroma
from config.config import Config
import chromadb
from utils.logger import get_logger

logger = get_logger(__name__)

class ChromaVectorStore(BaseVectorStore):
    """Implementation of BaseVectorStore using ChromaDB."""

    def __init__(self, collection_name, persist_directory, embedding_model):
        super().__init__(collection_name, persist_directory, embedding_model)
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory,
        )
        logger.info(f"ChromaDB initialized for collection '{self.collection_name}'.")

    def store_documents(self, documents):
        """Add documents to ChromaDB."""
        try:
            self.vectorstore.add_documents(documents=documents)
            logger.info(f"Successfully stored {len(documents)} documents in ChromaDB.")
        except Exception as e:
            logger.error(f"Error loading data into ChromaDB: {e}")

    def query_similar(self, query_text, top_k=5, filter=None):
        """Retrieve similar documents from ChromaDB."""
        try:

            # Similarity search with filter
            # results = self.vectorstore.similarity_search(
            #     query=query_text,
            #     k=top_k, 
            #     filter=filter
            # )
            
            # Similarity search without filter
            # results = self.vectorstore.similarity_search(
            #     query=query_text,
            #     k=top_k, 
            # )
            
            # Similarity search by vector
            # query_embedding = self.embedding_model.embed_query(query_text)
            # results = self.vectorstore.similarity_search_by_vector(query_embedding, k=top_k)

            # Similarity search with score - Convert List[Tuple[Document, float]] to List[Document] with score in metadata
            tuple_output = self.vectorstore.similarity_search_with_score(query=query_text, k=top_k, filter=filter)
            results = [
                Document(
                    id=doc.id,
                    metadata={**doc.metadata, "score": score},
                    page_content=doc.page_content
                )
                for doc, score in tuple_output
            ]

            return results
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            return []

    def list_collections(self):
        """List all collections in ChromaDB."""
        try:
            client = chromadb.PersistentClient(path=self.persist_directory)
            collection_names = client.list_collections()

            if not collection_names:
                logger.info("No collections found in ChromaDB.")
                return []

            logger.info("Collections in ChromaDB:")
            for name in collection_names:
                collection = client.get_collection(name)
                total_docs = len(collection.get()["ids"])
                logger.info(f"- {name} (Documents: {total_docs})")
            return collection_names
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
        
    def get_collections(self):
        """Get collection in ChromaDB."""
        try:
            client = chromadb.PersistentClient(path=self.persist_directory)
            collection_name = client.get_collection(name=self.collection_name)
            logger.info(type(collection_name.get(include=["metadatas"])))
            logger.info((collection_name.get(
                where={"regulation": "transfars"}
            )))

            if not collection_name:
                logger.info("Collection not found in ChromaDB.")
                return []
            logger.info(f"Database name: {collection_name.database}")
            logger.info(f"Collection name: {collection_name.name}")
            logger.info(f"Embedding name: {collection_name._embedding_function}")
        except Exception as e:
            logger.error(f"Error getting collections: {e}")
            return []
    
    def get_documents(self, where: dict = None, where_document: str = None, top_k=5):
        """Get collection in ChromaDB."""
        try:
            client = chromadb.PersistentClient(path=self.persist_directory)
            collection = client.get_collection(name=self.collection_name)
        
            if not collection:
                logger.info("Collection not found in ChromaDB.")
                return []
            query_params = {}
            if where:
                query_params["where"] = where
            if where_document:
                query_params["where_document"] = where_document

            results = collection.get(**query_params, include=["metadatas", "documents"], limit=top_k)
            return results
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
        return []
        
    def iterate_over_collection(self, collection_name):
        """Iterate and display all documents in the specified collection."""
        try:
            client = chromadb.PersistentClient(path=self.persist_directory)
            collection = client.get_collection(collection_name)

            if collection is None:
                logger.info(f"Collection '{collection_name}' not found.")
                return

            documents = collection.get(include=['documents', 'embeddings', 'metadatas'])
            total_docs = len(documents['ids'])
            logger.info(f"Total documents in collection '{collection_name}': {total_docs}")

            for i in range(total_docs):
                doc_id = documents['ids'][i]
                page_content = documents['documents'][i]
                metadata = documents['metadatas'][i]
                logger.info(f"\nDocument ID: {doc_id}")
                logger.info(f"Page Content: {page_content[:100]}...")
                logger.info(f"Metadata: {metadata}")
        except Exception as e:
            logger.info(f"Error iterating over collection '{collection_name}': {e}")
        
    def delete_collection(self):
        """Delete the ChromaDB collection."""
        try:
            self.vectorstore.delete()
            logger.info(f"Collection '{self.collection_name}' deleted successfully.")
        except Exception as e:
            logger.info(f"Error deleting collection '{self.collection_name}': {e}")

    def delete_specific_collection(self, collection_name):
        """Delete a specific collection by name."""
        try:
            client = chromadb.PersistentClient(path=self.persist_directory)
            client.delete_collection(collection_name)
            logger.info(f"Collection '{collection_name}' deleted successfully.")
        except Exception as e:
            logger.info(f"Error deleting collection '{collection_name}': {e}")

if __name__ == "__main__":
    from embedding.nomic import NomicEmbeddingGenerator

    print("\nInitializing ChromaDBManager with Nomic Embeddings...")
    
    embedding_model = NomicEmbeddingGenerator().model
    db_manager = ChromaVectorStore(collection_name=Config.CHROMA_DB_COLLECTION, persist_directory=Config.CHROMA_DB_PATH, embedding_model=embedding_model)

    # # Get collection
    # # db_manager.get_collections()
    # metadata_filter = {"regulation": "transfars"} 
    # document_filter = {"$contains": "The Two Councils"}

    # # Get Documents
    # # documents = db_manager.get_documents(Config.CHROMA_DB_COLLECTION, where=metadata_filter, where_document=document_filter)
    # documents = db_manager.get_documents(Config.CHROMA_DB_COLLECTION, where=metadata_filter)
    # print(len(documents["ids"]))
    # print(documents)

    # Iterate over collection
    # db_manager.iterate_over_collection(Config.CHROMA_DB_COLLECTION)
    
    # Delete collection
    # db_manager.delete_specific_collection("drupal_db_collection")
    # db_manager.get_collections()

    # Store and query documents
    
    documents = [
        Document(page_content="AI is evolving rapidly.", metadata={"source": "AI Paper"}),
        Document(page_content="Machine learning is part of AI.", metadata={"source": "ML Journal"})
    ]
    db_manager.store_documents(documents)
    query_text = "What is AI?"
    results = db_manager.query_similar(query_text)
    print("\nQuery Results:")
    for doc in results:
        print(f"- {doc.page_content}")
    
    