"""
Embedding utilities for AgroFlow AI.
"""

from sentence_transformers import SentenceTransformer

from chatbot.config import EMBEDDING_MODEL
from chatbot.logger import logger


_embedding_model = None


def load_embedding_model():

    """
    Load the embedding model once and reuse it
    throughout the application.
    """

    global _embedding_model

    if _embedding_model is not None:

        logger.info(
            "Embedding model already loaded."
        )

        return _embedding_model

    logger.info(
        "Loading embedding model: %s",
        EMBEDDING_MODEL
    )

    try:

        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        logger.info(
            "Embedding model loaded successfully."
        )

        return _embedding_model

    except Exception:

        logger.exception(
            "Failed to load embedding model."
        )

        raise


def create_query_embedding(
    embedding_model,
    question
):

    """
    Generate an embedding vector for a single question.
    """

    try:

        embedding = embedding_model.encode(
            question,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        logger.info(
            "Query embedding generated."
        )

        return embedding.tolist()

    except Exception:

        logger.exception(
            "Failed to generate query embedding."
        )

        raise


def create_query_embeddings(
    embedding_model,
    questions
):

    """
    Generate embeddings for multiple search queries.
    """

    try:

        embeddings = embedding_model.encode(
            questions,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        logger.info(
            "Generated embeddings for %d queries.",
            len(questions)
        )

        return embeddings.tolist()

    except Exception:

        logger.exception(
            "Failed to generate multiple embeddings."
        )

        raise