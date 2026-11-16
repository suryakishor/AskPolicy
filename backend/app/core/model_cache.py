"""Point Hugging Face and Sentence Transformers at the project cache.

Imported before the model libraries so their cache environment variables are set
before they read them. Weights live only in ``backend/.cache``.
"""

from __future__ import annotations

import os

from app.core.config import BACKEND_DIR, settings


def configure_model_cache() -> None:
    """Create the cache directory and point model libraries at it."""
    cache_root = settings.model_cache_dir
    if not cache_root.is_absolute():
        cache_root = (BACKEND_DIR / cache_root).resolve()

    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "hub").mkdir(parents=True, exist_ok=True)
    (cache_root / "sentence_transformers").mkdir(parents=True, exist_ok=True)

    # ``setdefault`` keeps an explicit shell override while defaulting to the
    # project-managed cache.
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_root / "sentence_transformers"))
    # Older libraries still read TRANSFORMERS_CACHE; keeping it aligned avoids
    # a second copy of the weights.
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


configure_model_cache()
