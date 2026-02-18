import pytest

from example.rag_example import RAGConfig, MiniRAGSystem
from langchain.vectorstores import Qdrant


@pytest.fixture(autouse=True)
def patch_qdrant_client(monkeypatch):
    """Avoid actual network connections in tests by stubbing QdrantClient."""
    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        # methods that might be used by the wrapper
        def count(self, collection_name: str):
            class C:
                count = 0
            return C()

    monkeypatch.setattr("example.rag_example.QdrantClient", DummyClient)
    yield


def test_init_creates_qdrant_instance():
    cfg = RAGConfig()
    rag = MiniRAGSystem(cfg)
    assert isinstance(rag.vector_store, Qdrant)
    # client should be our dummy so that operations do not fail
    assert hasattr(rag, "qdrant_client")


def test_get_stats_uses_client():
    cfg = RAGConfig()
    rag = MiniRAGSystem(cfg)
    stats = rag.get_stats()
    assert stats["total_chunks_in_store"] == 0
    assert stats["vector_store_url"] == cfg.vector_store_url
