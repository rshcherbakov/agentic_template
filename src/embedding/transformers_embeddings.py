from __future__ import annotations

from typing import List, Union

import torch
from transformers import AutoModel, AutoTokenizer


class TransformersEmbeddings:
    """Lightweight wrapper around a HuggingFace ``transformers``
    embedding model. This class can be used as a drop-in replacement
    for ``langchain.embeddings.HuggingFaceEmbeddings`` when you want to
    run a model locally (`self-hosted`) and have full control over the
    pooling behaviour.

    It currently performs mean pooling over the last hidden state and
    applies L2-normalisation to match the behaviour of most embedding
    providers.
    """

    def __init__(self, model_name: str, device: str | torch.device = "cpu") -> None:
        self.model_name = model_name
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # ``AutoModel`` returns the base model (no LM head); most
        # embedding checkpoints are provided as base models.
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Encode a batch of texts and return a list of float vectors.

        Args:
            texts: a single string or a list of strings.

        Returns:
            List of embeddings (one per input string).
        """
        if isinstance(texts, str):
            texts = [texts]

        # tokenizer handles batching for us
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**encoded)
            # `last_hidden_state` has shape (batch, seq_len, hidden)
            hidden = output.last_hidden_state
            # mean pooling over the sequence dimension
            embeddings = hidden.mean(dim=1)
            # optional normalization to unit length
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().tolist()

    def __call__(self, texts: Union[str, List[str]]) -> List[List[float]]:
        return self.embed(texts)
