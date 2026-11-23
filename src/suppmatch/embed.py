"""Embeddings: turn text into vectors so similar names/addresses are close.

- LocalEmbedder: the real small model (sentence-transformers), imported only when first used.
- FakeEmbedder: deterministic character n-gram hashing, for tests (no download, no GPU).
- CachedEmbedder: wraps either one; the same text is never embedded twice.
All vectors are L2-normalized, so cosine similarity is a plain dot product.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np

from . import config


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """One L2-normalized row per text."""


def _normalize_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class FakeEmbedder:
    """Character 3-grams hashed into a fixed-size vector (stable sha1, not Python's salted hash()).

    Strings that share most of their 3-grams get a high cosine, so "APEX TOOLS" and
    "APEKS TOOLS" score close, and unrelated names score low.
    """

    name = "fake-ngram-v1"

    def __init__(self, dim: int = 512, n: int = 3):
        self.dim, self.n = dim, n

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        padded = f"  {text}  "
        for i in range(len(padded) - self.n + 1):
            gram = padded[i:i + self.n].encode("utf-8")
            v[int.from_bytes(hashlib.sha1(gram).digest()[:4], "big") % self.dim] += 1.0
        return v

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize_rows(np.stack([self._vector(t) for t in texts]))


class LocalEmbedder:
    """sentence-transformers model (default all-MiniLM-L6-v2), loaded on first use."""

    def __init__(self, model_name: str | None = None):
        self.name = model_name or config.EMBED_MODEL
        self._model = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # lazy: heavy import
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed. Run: python -m pip install -e \".[embed]\" "
                    "(or use --embedder fake for a quick test run)") from exc
            self._model = SentenceTransformer(self.name)
        vecs = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float32)


def cache_key(model_name: str, text: str) -> str:
    return hashlib.sha256(f"{model_name}\n{text}".encode("utf-8")).hexdigest()


class CachedEmbedder:
    """Disk + memory cache in front of an embedder.

    Key = sha256(model name + text), one .npy file per key under cache_dir.
    `computed` counts texts actually sent to the model; `hits` counts cache answers.
    """

    def __init__(self, inner: Embedder, cache_dir: Path | None = None, use_cache: bool = True):
        self.inner = inner
        self.name = inner.name
        self.cache_dir = Path(cache_dir or config.CACHE_DIR / "embeddings")
        self.use_cache = use_cache
        self._memory: dict[str, np.ndarray] = {}
        self.computed = 0
        self.hits = 0

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.npy"

    def _lookup(self, key: str) -> np.ndarray | None:
        if key in self._memory:
            return self._memory[key]
        if self.use_cache:
            path = self._path(key)
            if path.exists():
                vec = np.load(path, allow_pickle=False)
                self._memory[key] = vec
                return vec
        return None

    def embed(self, texts: list[str]) -> np.ndarray:
        keys = [cache_key(self.name, t) for t in texts]
        missing: dict[str, str] = {}          # key -> text, each unique text once
        for key, text in zip(keys, texts):
            if key in missing:
                continue
            if self._lookup(key) is not None:
                self.hits += 1
            else:
                missing[key] = text
        if missing:
            vecs = self.inner.embed(list(missing.values()))
            self.computed += len(missing)
            for key, vec in zip(missing, vecs):
                self._memory[key] = vec
                if self.use_cache:
                    path = self._path(key)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, vec, allow_pickle=False)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([self._memory[k] for k in keys])


def get_embedder(kind: str, use_cache: bool = True, cache_dir: Path | None = None) -> CachedEmbedder:
    if kind == "fake":
        inner: Embedder = FakeEmbedder()
    elif kind == "local":
        inner = LocalEmbedder()
    else:
        raise ValueError(f"unknown embedder {kind!r} (use fake or local)")
    return CachedEmbedder(inner, cache_dir, use_cache)
