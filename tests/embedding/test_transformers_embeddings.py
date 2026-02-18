import pytest

from src.embedding.transformers_embeddings import TransformersEmbeddings


@pytest.mark.parametrize("model_name", [
    # use a tiny model for CI; the repository already depends on
    # sentence-transformers so this should download quickly.
    "sentence-transformers/all-MiniLM-L6-v2",
])
def test_embeddings_return_correct_shape(model_name: str):
    embedder = TransformersEmbeddings(model_name=model_name)
    texts = ["hello world", "another sentence"]
    vectors = embedder.embed(texts)

    assert isinstance(vectors, list)
    assert len(vectors) == len(texts)
    assert all(isinstance(v, list) for v in vectors)
    # all vectors should have the same dimensionality
    dims = {len(v) for v in vectors}
    assert len(dims) == 1
    assert dims.pop() > 0


def test_call_alias(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embedder = TransformersEmbeddings(model_name=model_name)
    single = embedder("one")
    multi = embedder(["one", "two"])
    assert len(single) == 1
    assert len(multi) == 2
