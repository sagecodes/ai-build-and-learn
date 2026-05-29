from fastembed import TextEmbedding
from config import EMBEDDING_MODEL

_model: TextEmbedding | None = None


def get_embedder() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(EMBEDDING_MODEL)
    return _model


def embed(text: str) -> list[float]:
    embedder = get_embedder()
    return list(embedder.embed([text]))[0].tolist()
