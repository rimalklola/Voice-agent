from __future__ import annotations

import os
import threading
from typing import Iterable, List

import numpy as np

_TEST_MODE = False
_TEST_DIM = 384

class _SingletonEmbedder:
    _lock = threading.Lock()
    _instance = None

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._test_mode = _TEST_MODE
        self._model = None
        self._device = os.getenv("EMBED_DEVICE", "auto")  # auto|cpu|cuda|mps

    def _load(self):
        if self._test_mode:
            return
        if self._model is None:
            # Lazy import to speed up startup when not used
            from sentence_transformers import SentenceTransformer
            # Prefer explicit device if provided; else default auto selection
            if self._device != "auto":
                self._model = SentenceTransformer(self.model_name, device=self._device)
            else:
                try:
                    import torch
                    if torch.cuda.is_available():
                        dev = "cuda"
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        dev = "mps"
                    else:
                        dev = "cpu"
                except Exception:
                    dev = "cpu"
                self._model = SentenceTransformer(self.model_name, device=dev)

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        if self._test_mode:
            return [self._fake_embed(t) for t in texts]
        self._load()
        vecs = self._model.encode(list(texts), batch_size=64, normalize_embeddings=True)
        if isinstance(vecs, np.ndarray):
            return vecs.tolist()
        return [list(v) for v in vecs]

    @staticmethod
    def _fake_embed(text: str) -> List[float]:
        # Deterministic embedding used in tests only
        dim = _TEST_DIM
        arr = np.zeros(dim, dtype=np.float32)
        for i, ch in enumerate(text.encode("utf-8")):
            arr[(i + ch) % dim] += (ch % 7 + 1) * 0.1
        norm = np.linalg.norm(arr)
        if norm == 0:
            return arr.tolist()
        return (arr / norm).tolist()


_embedder_singleton: _SingletonEmbedder | None = None


def get_embedder(model_name: str) -> _SingletonEmbedder:
    global _embedder_singleton
    if _embedder_singleton is None:
        with _SingletonEmbedder._lock:
            if _embedder_singleton is None:
                _embedder_singleton = _SingletonEmbedder(model_name)
                if os.getenv("EMBED_WARMUP", "0").lower() in {"1", "true", "yes", "on"}:
                    try:
                        _embedder_singleton.embed(["warmup"])  # loads model
                    except Exception:
                        pass
    return _embedder_singleton


def enable_test_mode(dim: int = 384) -> None:
    global _TEST_MODE, _TEST_DIM, _embedder_singleton
    _TEST_MODE = True
    _TEST_DIM = dim
    _embedder_singleton = None


def disable_test_mode() -> None:
    global _TEST_MODE, _embedder_singleton
    _TEST_MODE = False
    _embedder_singleton = None


__all__ = ["get_embedder", "enable_test_mode", "disable_test_mode"]
