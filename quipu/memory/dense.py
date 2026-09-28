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
# Pinned so a later upload to the hub can't silently change the numbers; recorded in
# every needle_eval result file.
E5_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


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

    def _check(self, i: int) -> None:
        if not 0 <= i < self.n:
            raise IndexError(f"chunk {i} out of range for {self.n} chunks")

    def matrix(self) -> np.ndarray:
        """A copy of every stored row."""
        return self._emb.copy()

    def row(self, i: int) -> np.ndarray:
        """A copy of row i, exactly as stored, for set_row to put back later."""
        self._check(i)
        return self._emb[i].copy()

    def set_row(self, i: int, vec: np.ndarray) -> None:
        """Put back a row saved with row(). Stored as given, not re-normalised, so a
        restore is bit-exact. Re-embedding the old text instead would not be: on a GPU
        a text embedded alone doesn't reproduce the row it got inside a padded batch,
        and the index would drift a little with every trial."""
        self._check(i)
        vec = np.asarray(vec, dtype=np.float32)
        if vec.shape != self._emb[i].shape:
            raise ValueError(f"row must have shape {self._emb[i].shape}, got {vec.shape}")
        self._emb[i] = vec

    def patch(self, i: int, text: str) -> None:
        """Re-embed chunk i only; every other row is left as it was. The embedding is
        computed before anything is written, so a failing embedder changes nothing."""
        self._check(i)
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
    a dense or fused condition runs, so BM25-only runs need no model at all.

    `truncated` counts passages longer than max_length e5 tokens, which lose their
    tail silently; needle_eval records it so a truncated needle can't hide."""

    def __init__(self, device: str = "cpu", batch_size: int = 64, max_length: int = 512,
                 model_name: str = E5_MODEL, revision: str = E5_REVISION) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.model_name = model_name
        self.revision = revision
        self.truncated = 0
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModel.from_pretrained(model_name, revision=revision).to(device).eval()
        self.dim = int(self.model.config.hidden_size)

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        torch = self._torch
        out: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = list(texts[i:i + self.batch_size])
                lengths = self.tokenizer(batch_texts, truncation=False)["input_ids"]
                self.truncated += sum(1 for ids in lengths if len(ids) > self.max_length)
                batch = self.tokenizer(
                    batch_texts, padding=True, truncation=True,
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
