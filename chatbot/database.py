"""
Database utilities for AgroFlow AI.
"""

import os

from dotenv import load_dotenv
from supabase import create_client

from chatbot.config import TOP_K
from chatbot.logger import logger


load_dotenv()


SUPABASE_URL = os.getenv(
    "SUPABASE_URL"
)

SUPABASE_KEY = os.getenv(
    "SUPABASE_SERVICE_ROLE_KEY"
)


if not SUPABASE_URL:

    raise ValueError(
        "SUPABASE_URL is missing."
    )


if not SUPABASE_KEY:

    raise ValueError(
        "SUPABASE_SERVICE_ROLE_KEY is missing."
    )


def get_supabase():

    logger.info(
        "Connecting to Supabase."
    )

    return create_client(
        SUPABASE_URL,
        SUPABASE_KEY
    )


def retrieve_documents(
    supabase,
    embedding,
    match_count=TOP_K
):
    """
    Retrieve the most similar document chunks
    using pgvector.
    """

    try:

        result = supabase.rpc(
            "match_documents",
            {
                "query_embedding": embedding,
                "match_count": match_count
            }
        ).execute()

        documents = result.data or []

        if not documents:

            logger.info(
                "Vector search returned 0 documents."
            )

            return [], 0.0

        best_similarity = documents[0].get(
            "similarity",
            0.0
        )

        logger.info(
            "Vector search returned %d document(s). "
            "Best similarity: %.3f",
            len(documents),
            best_similarity
        )

        return documents, best_similarity

    except Exception:

        logger.exception(
            "Vector search failed."
        )

        return [], 0.0


def keyword_search(
    supabase,
    question,
    match_count=TOP_K
):
    """
    Perform PostgreSQL full-text search.
    """

    try:

        result = supabase.rpc(
            "keyword_search",
            {
                "search_query": question,
                "match_count": match_count
            }
        ).execute()

        documents = result.data or []

        logger.info(
            "Keyword search returned %d document(s).",
            len(documents)
        )

        return documents

    except Exception:

        logger.exception(
            "Keyword search failed."
        )

        return []