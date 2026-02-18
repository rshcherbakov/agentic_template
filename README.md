# agentic_template
first iteration of agentic template 

## Vector Store

The example configuration now uses **Qdrant** as the retrieval backend.  A
running Qdrant server is required (`http://localhost:6333` by default);
configuration lives in `RAGConfig.vector_store_url` or under the
`[tool.confluence-rag.vector_store]` section of `pyproject.toml`.
