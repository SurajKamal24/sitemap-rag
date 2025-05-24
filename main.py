from embedding.nomic import NomicEmbeddingGenerator
from embedding.gemini import GeminiEmbeddingGenerator
from vectorstore.chroma import ChromaVectorStore
from langchain_core.documents import Document
from loader.sitemap import Sitemap
from chat.llama import LlamaChat
from chat.gemini import GeminiChat
from config.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

def initialize_embedding_model(embedding_type):
    if embedding_type.lower() == "nomic":
        logger.info("Initializing Nomic Embedding Model...")
        return NomicEmbeddingGenerator()
    elif embedding_type.lower() == "gemini":
        logger.info("Initializing Gemini Embedding Model...")
        return GeminiEmbeddingGenerator()
    else:
        logger.error(f"Unknown embedding model: {embedding_type}")
        raise ValueError(f"Unsupported embedding type: {embedding_type}")
    
def initialize_vector_store(store_type, collection_name, persist_directory, embedding_model):
    if store_type.lower() == "chroma":
        logger.info("Initializing Chroma Vector Store...")
        return ChromaVectorStore(
            collection_name=collection_name, 
            persist_directory=persist_directory, 
            embedding_model=embedding_model
        )
    else:
        logger.error(f"Unknown vector store: {store_type}")
        raise ValueError(f"Unsupported vector store: {store_type}")

def initialize_chat_model(chat_type, temperature, max_tokens):
    if chat_type.lower() == "gemini":
        logger.info("Initializing Gemini Chat Model...")
        return GeminiChat(temperature, max_tokens)
    elif chat_type.lower() == "llama":
        logger.info("Initializing Llama Chat Model...")
        return LlamaChat(temperature, max_tokens)
    else:
        logger.error(f"Unknown chat model: {chat_type}")
        raise ValueError(f"Unsupported chat model: {chat_type}")

def load_data(sitemap_url, vector_store, block_size, filter_urls, filter_pattern):
    logger.info("Starting Sitemap RAG data loading job...")
    try:
        loader = Sitemap(
            sitemap_url=sitemap_url,
            vector_store=vector_store,
            block_size=block_size,
            filter_urls=filter_urls,
            filter_pattern=filter_pattern
        )
        loader.load_records()
        logger.info("Sitemap RAG data loading job completed!")
    except Exception as e:
        logger.error(f"Error during data loading: {e}")
    
def semantic_search(vector_store, chat_model, query, filter, session_id, mode="search"):
        
    if not query.strip():
        logger.warning("Empty query provided for semantic search.")
        return "Empty query provided for semantic search."

    logger.info(f"Semantic Search Query: {query}")

    try:
        results = vector_store.query_similar(query_text=query, top_k=Config.TOP_K_RESULTS, filter=filter)
        logger.info(f"Retrieved {len(results)} results from ChromaDB.")
        #logger.debug(f"Results with scores: {results}")

        if not results:
            logger.info("No relevant semantic search results found.")
            return "I'm sorry, but I couldn't generate a response for this query."

        documents, scores, references = [], [], []

        for doc in results:
            score = doc.metadata.get("score", None)
            documents.append(doc)
            scores.append(score)
            references.append({
                "title": doc.metadata.get("subtopic", "Unknown Title"),
                "url": doc.metadata.get("source", "#"),
                "score": score
            })

        logger.info(f"Semantic search returned {len(documents)} documents.")

        context_docs = [doc for doc, score in zip(documents, scores) if score <= Config.SCORE_THRESHOLD]

        if not context_docs:
            logger.info("No sufficiently relevant documents found, bypassing similarity search.")
            context = None
        else:
            context = "\n\n".join([doc.page_content for doc in context_docs])
        prompt_type = "contextual" if context else "general"
        logger.info(f"Using {prompt_type} prompt for query: {query}")

        if mode == "chat":
            chat_response = chat_model.generate_response_with_history(query, context, session_id)
        else:
            chat_response = chat_model.generate_response(query, context)

        logger.info(f" AI Response:\n{chat_response}")

        return {"response_text": chat_response, "references": references}
        
    except Exception as e:
        logger.error(f"Error doing semantic search: {e}")
        return ""

def keyword_search(vector_store, text, filter, top_k):

    logger.info(f"Keyword Search text: {text}")

    where_filter = {"topic": {"$in": [reg.lower() for reg in filter]}} if filter else None

    documents = vector_store.get_documents(where=where_filter, where_document={"$contains": text}, top_k=top_k)

    logger.info(f"Total documents retrieved from vector store {documents}")

    if documents and isinstance(documents, dict) and "documents" in documents:
        retrieved_docs = documents["documents"]
        metadata_list = documents.get("metadatas", [{}] * len(retrieved_docs))

        if retrieved_docs:
            logger.info(f"### Found {len(retrieved_docs)} matching documents")
            results = []
            for i, (doc_content, metadata) in enumerate(zip(retrieved_docs, metadata_list), start=1):
                title = metadata.get("subtopic", "Unknown Title")
                url = metadata.get("source", "#")
                short_content = (
                    " ".join(doc_content.split()[:100]) + "..." 
                    if len(doc_content.split()) > 100 
                    else doc_content
                )
                results.append({
                    "title": title,
                    "url": url,
                    "short_content": short_content,
                })

            return results
        else:
            return "No documents found for the given search."
    else:
        return "No documents found for the given search."
   
def data_loading_test():
    """Demo function to load data from sitemap and store it in the vector store."""
    logger.info("Starting Sitemap Data Loading Demo")
    embedding_model = initialize_embedding_model(Config.DEFAULT_EMBEDDING_MODEL).model
    vector_store = initialize_vector_store(Config.DEFAULT_VECTOR_STORE, Config.CHROMA_DB_COLLECTION, Config.CHROMA_DB_PATH, embedding_model)
    load_data(Config.SITEMAP_URL, vector_store, Config.BATCH_SIZE, Config.FILTER_URLS, Config.FILTER_PATTERN)
    logger.info("Sitemap Data Loading Demo Completed")

def keyword_search_test():
    """Demo function to perform keyword search."""
    logger.info("Starting Keyword Search Demo")
    embedding_model = initialize_embedding_model(Config.DEFAULT_EMBEDDING_MODEL).model
    vector_store = initialize_vector_store(Config.DEFAULT_VECTOR_STORE, Config.CHROMA_DB_COLLECTION, Config.CHROMA_DB_PATH, embedding_model)
    text = "Innovation"
    filter_list = ["content"]
    response = keyword_search(vector_store, text, filter_list)
    logger.info(f"Keyword search response: {response}")
    logger.info("Keyword Search Demo Completed")

def semantic_search_search_mode_test():
    """Demo function to run semantic search in search-only mode."""
    logger.info("Starting Semantic Search (Search Mode) Demo")
    embedding_model = initialize_embedding_model(Config.DEFAULT_EMBEDDING_MODEL).model
    vector_store = initialize_vector_store(Config.DEFAULT_VECTOR_STORE, Config.CHROMA_DB_COLLECTION, Config.CHROMA_DB_PATH, embedding_model)
    chat_model = initialize_chat_model(Config.DEFAULT_CHAT_MODEL, Config.DEFAULT_TEMPERATURE, Config.DEFAULT_MAX_TOKENS)
    query = "What is the purpose of the Acquisition Excellence and Small Business Excellence Awards?"
    response = semantic_search(
        vector_store,
        chat_model,
        query=query,
        filter=None,
        session_id=None,
        mode="search"
    )
    logger.info(f"Semantic Search (Search Mode) Response:\n{response}")

def semantic_search_chat_mode_test():
    """Demo function to simulate a multi-turn chat conversation."""
    logger.info("Starting Multi-Turn Semantic Chat Demo")

    embedding_model = initialize_embedding_model(Config.DEFAULT_EMBEDDING_MODEL).model
    vector_store = initialize_vector_store(Config.DEFAULT_VECTOR_STORE, Config.CHROMA_DB_COLLECTION, Config.CHROMA_DB_PATH, embedding_model)
    chat_model = initialize_chat_model(Config.DEFAULT_CHAT_MODEL, Config.DEFAULT_TEMPERATURE, Config.DEFAULT_MAX_TOKENS)

    queries = [
        {"query": "What is the purpose of the Acquisition Excellence and Small Business Excellence Awards?", "session_id": "1234"},
        {"query": "What did I ask you?", "session_id": "1234"},
        {"query": "Who is eligible to be nominated for these awards?", "session_id": "5678"},
        {"query": "Who is eligible to be nominated for these awards?", "session_id": "1234"}
    ]

    for idx, query in enumerate(queries, 1):
        logger.info(f"User Query {idx}: {query["query"]}")
        response = semantic_search(
            vector_store,
            chat_model,
            query=query["query"],
            filter=None,
            session_id=query["session_id"],
            mode="chat"
        )
        logger.info(f"AI Response {idx}:\n{response}")

    logger.info("Multi-Turn Chat Demo Completed")

if __name__ == "__main__":
    data_loading_test()
    #keyword_search_test()
    #semantic_search_search_mode_test()
    #semantic_search_chat_mode_test()