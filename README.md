## Sitemap RAG ##

Sitemap RAG is a retrieval-augmented generation (RAG) pipeline that enables question-answering and semantic search over a website’s content by leveraging the site’s sitemap. By crawling a website’s sitemap (an XML index of all pages) and embedding the page contents into a vector database, Sitemap RAG allows you to query or chat with an LLM using the website’s own content as context. This leads to more specific and grounded answers than a general search engine can provide, making it useful for building FAQ chatbots, documentation assistants, or support agents tailored to a particular site.

## Tech Stack Used ##

<table border="1">
  <tr>
    <th>Component</th>
    <th>Technology</th>
    <th>Purpose</th>
  </tr>
  <tr>
    <td>Local LLM Inference</td>
    <td>Ollama</td>
    <td>Runs large language models (LLMs) locally, avoiding dependence on external APIs</td>
  </tr>
  <tr>
    <td>LLM Model</td>
    <td>llama3.2:latest via Ollama</td>
    <td>Used to generate natural language answers from retrieved context chunks</td>
  </tr>
  <tr>
    <td>Embedding Model</td>
    <td>nomic-embed-text via Ollama</td>
    <td>Generates vector representations (embeddings) of document text for semantic similarity search</td>
  </tr>
  <tr>
    <td>Vector Database</td>
    <td>Chroma</td>
    <td>Stores and retrieves document embeddings efficiently during semantic search</td>
  </tr>
  <tr>
    <td>Orchestration Layer</td>
    <td>Langchain</td>
    <td>Manages chaining of tools: embeddings, vector store, retrievers, prompt templates, and LLMs</td>
  </tr>
  <tr>
    <td>User Interface</td>
    <td>Streamlit</td>
    <td>Provides an interactive and intuitive web-based UI to perform ingestion and query workflows</td>
  </tr>
</table>


1. Ollama serves as the backbone for running both the embedding model (nomic-embed-text) and the LLM (llama3.2:latest) locally. This approach removes reliance on cloud-based APIs like OpenAI, allowing full control over data privacy and cost.
2. Chroma Vector Database is used as the local vector store to persist and retrieve text embeddings. It offers fast approximate nearest-neighbor search and integrates well with LangChain retrievers. Chroma is preferred in this project for its minimal setup and local operation.
3. LangChain orchestrates the components of the RAG pipeline. It abstracts away low-level integration logic and provide
    - composable interfaces for:
    - Loading and parsing documents
    - Splitting and embedding text
    - Storing vectors in Chroma
    - Retrieving relevant chunks at query time
    - Constructing prompts and invoking the LLM via Ollama
4. Streamlit provides a clean web interface that enables users to:
    - Input the website sitemap URL
    - Trigger the ingestion and indexing pipeline
    - Ask questions about the indexed site content
    - View answers along with referenced source pages

## Architecture ##

![Architecture](./docs/sitemap-rag.drawio.svg) 