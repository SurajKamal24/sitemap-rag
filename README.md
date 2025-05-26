# Sitemap RAG #

Sitemap RAG is a retrieval-augmented generation (RAG) pipeline that enables question-answering and semantic search over a website’s content by leveraging the site’s sitemap. By crawling a website’s sitemap (an XML index of all pages) and embedding the page contents into a vector database, Sitemap RAG allows you to query or chat with an LLM using the website’s own content as context. This leads to more specific and grounded answers than a general search engine can provide, making it useful for building FAQ chatbots, documentation assistants, or support agents tailored to a particular site.

### Architecture Overview ###
