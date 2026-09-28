"""Dense retrieval: e5 embeddings of decoded chunks, cosine top-k.

The embedder is injected (anything with embed_passages / embed_queries), so tests
use a lookup table and never download a model. E5Embedder is the real one:
intfloat/multilingual-e5-small, with the "passage: " / "query: " prefixes the model
was trained with (it retrieves noticeably worse without them), mean pooling over the
attention mask and L2 normalisation, which is the recipe on the model card.
"""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

E5_MODEL = "intfloat/multilingual-e5-small"


class Embedder(Protocol):
    def embed_passages(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_queries(self, texts: Sequence[str]) -> np.ndarray: ...


def _l2_normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"embeddings must be 2-D (n, dim), got shape {x.shape}")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


class DenseIndex:
    """Normalised passage embeddings, one row per chunk. Normalising here as well as
    in the embedder means a fake (or any other) embedder still gets cosine, not dot
    product, so a long vector can't win on length alone."""

    def __init__(self, texts: Sequence[str], embedder: Embedder) -> None:
        if not texts:
            raise ValueError("DenseIndex needs at least one chunk")
        self.embedder = embedder
        self._emb = _l2_normalise(embedder.embed_passages(list(texts)))
        if self._emb.shape[0] != len(texts):
            raise ValueError(
                f"embedder returned {self._emb.shape[0]} rows for {len(texts)} texts"
            )

    @property
    def n(self) -> int:
        return self._emb.shape[0]

    def patch(self, i: int, text: str) -> None:
        """Re-embed chunk i only; every other row is left as it was."""
        if not 0 <= i < self.n:
            raise IndexError(f"chunk {i} out of range for {self.n} chunks")
        self._emb[i] = _l2_normalise(self.embedder.embed_passages([text]))[0]

    def search(self, query: str, k: int) -> list[int]:
        """Top-k chunk ids by cosine similarity; ties go to the earlier chunk."""
        q = _l2_normalise(self.embedder.embed_queries([query]))[0]
        sims = self._emb @ q
        k = min(k, self.n)
        idx = np.arange(self.n)
        order = np.lexsort((idx, -sims))
        return [int(i) for i in order[:k]]


class E5Embedder:
    """multilingual-e5-small via transformers. Loaded lazily by needle_eval only when
    a dense or fused condition runs, so BM25-only runs need no model at all."""

    def __init__(self, device: str = "cpu", batch_size: int = 64, max_length: int = 512,
                 model_name: str = E5_MODEL) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        out: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(
                    list(texts[i:i + self.batch_size]), padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                out.append(pooled.float().cpu().numpy())
        return np.concatenate(out, axis=0)

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed([f"passage: {t}" for t in texts])

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed([f"query: {t}" for t in texts])
