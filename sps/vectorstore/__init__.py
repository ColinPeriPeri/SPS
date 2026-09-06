from .base import VectorStore, point_id_for
from .memory_store import InMemoryVectorStore

__all__ = [
    "VectorStore",
    "point_id_for",
    "InMemoryVectorStore",
    "QdrantVectorStore",
    "VectorStoreBusyError",
]


def __getattr__(name: str):
    # Deferred so importing the package does not require qdrant-client.
    if name in ("QdrantVectorStore", "VectorStoreBusyError"):
        from . import qdrant_store

        return getattr(qdrant_store, name)
    raise AttributeError(name)
