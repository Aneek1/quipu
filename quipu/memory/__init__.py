"""Approach A: retrieval memory for a 1,024-token model.

The model itself is untouched. A long document is cut into fixed chunks, the chunks
are indexed twice (BM25 over raw token ids for exact identifiers, dense e5
embeddings for meaning), the two rankings are fused, and the best chunks are packed
in front of the prompt. Whether that lets a 1,024-token model answer from 1M tokens
is what scripts/needle_eval.py measures.
"""
