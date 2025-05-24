from langchain_community.document_loaders import SitemapLoader
from bs4 import BeautifulSoup
from utils.logger import get_logger
from config.config import Config
import re
import requests

logger = get_logger(__name__)

class Sitemap:
    """Class for loading sitemaps."""
    def __init__(self, sitemap_url, vector_store, block_size, filter_urls, filter_pattern):
        self.sitemap_url = sitemap_url
        self.vector_store = vector_store
        self.block_size = block_size
        self.filter_urls = filter_urls
        self.filter_pattern = filter_pattern

    def load_records(self):
        """Retrieve sitemaps in batches and store them."""
        logger.info(f"Processing sitemap with url {self.sitemap_url}.")
        
        response = requests.get(self.sitemap_url)
        sitemap_content = response.content
        soup = BeautifulSoup(sitemap_content, "xml")

        loader_kwargs = {"web_path": self.sitemap_url}
        if self.filter_urls:
            loader_kwargs["filter_urls"] = [rf".*{self.filter_pattern}.*"]
            logger.info(loader_kwargs["filter_urls"])

        loader = SitemapLoader(**loader_kwargs)

        sitemap_elements = loader.parse_sitemap(soup)
        total_urls = len(sitemap_elements)
        logger.info(f"Found {total_urls} URLs in sitemap.")

        total_blocks = (total_urls + self.block_size - 1) // self.block_size
        logger.info(f"Total blocks: {total_blocks}.")

        for blocknum in range(total_blocks):
            logger.info(f"Processing block {blocknum + 1} of {total_blocks}.")

            block_loader_kwargs = {
                "web_path": self.sitemap_url,
                "blocksize": self.block_size,
                "blocknum": blocknum
            }
            if self.filter_urls:
                block_loader_kwargs["filter_urls"] = [rf".*{self.filter_pattern}.*"]

            loader = SitemapLoader(**block_loader_kwargs)
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} documents from block {blocknum + 1}.")

            for doc in docs: 
                soup = BeautifulSoup(doc.page_content, "html.parser")

                for tag in soup(["nav", "footer", "script", "style", "header"]):
                    tag.decompose()
                
                clean_text = soup.get_text(separator=" ", strip=True)

                clean_text = re.sub(r'\s+', ' ', clean_text)

                doc.page_content = clean_text
                
                source_url = doc.metadata.get("source", "")
                parts = source_url.replace(self.sitemap_url.replace('/sitemap.xml', "/"), "").split("/")
                topic = parts[0] if len(parts) > 0 else ""
                subtopic = parts[1] if len(parts) > 1 else ""
                doc.metadata["topic"] = topic
                doc.metadata["subtopic"] = subtopic

            for doc in docs[:5]:
                print(doc.metadata)
            
            self.vector_store.store_documents(docs)

        logger.info("All blocks processed successfully")


if __name__ == "__main__":

    from embedding.nomic import NomicEmbeddingGenerator
    from vectorstore.chroma import ChromaVectorStore

    embedding_model = NomicEmbeddingGenerator().model
    vector_store = ChromaVectorStore(collection_name=Config.CHROMA_DB_COLLECTION, persist_directory=Config.CHROMA_DB_PATH, embedding_model=embedding_model)
    sitemap_loader = Sitemap(sitemap_url=Config.SITEMAP_URL, vector_store=vector_store, block_size=Config.BATCH_SIZE, filter_urls=True, filter_pattern="content")
    sitemap_loader.load_records()