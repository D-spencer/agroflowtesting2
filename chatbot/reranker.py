"""
Cross-Encoder Reranker
"""

from sentence_transformers import CrossEncoder

from chatbot.config import (
    RERANKER_MODEL,
    ENABLE_RERANKING
)

from chatbot.logger import logger


MAX_RERANK_DOCUMENTS = 5

MIN_DOCUMENTS_TO_RERANK = 3

BATCH_SIZE = 16


reranker = None


def get_reranker():
    """
    Lazily load the CrossEncoder model and reuse it.
    """

    global reranker

    if not ENABLE_RERANKING:

        logger.info(
            "Cross-Encoder reranking is disabled."
        )

        return None

    if reranker is None:

        logger.info(
            "Loading Cross-Encoder reranker: %s",
            RERANKER_MODEL
        )

        reranker = CrossEncoder(
            RERANKER_MODEL
        )

        logger.info(
            "Cross-Encoder reranker loaded successfully."
        )

    return reranker


def rerank_documents(
    question,
    documents
):
    """
    Rerank retrieved documents using a CrossEncoder.
    """

    if not ENABLE_RERANKING:

        logger.info(
            "Reranking disabled. Returning documents "
            "without Cross-Encoder reranking."
        )

        return documents

    if not documents:

        logger.info(
            "Skipping reranking because no documents "
            "were retrieved."
        )

        return []

    if len(documents) <= MIN_DOCUMENTS_TO_RERANK:

        logger.info(
            "Skipping reranking (%d document(s) only).",
            len(documents)
        )

        return documents

    documents_to_rerank = documents[
        :MAX_RERANK_DOCUMENTS
    ]

    logger.info(
        "Reranking %d of %d retrieved document(s).",
        len(documents_to_rerank),
        len(documents)
    )

    model = get_reranker()

    if model is None:

        logger.info(
            "Reranker unavailable. Returning original "
            "document order."
        )

        return documents

    pairs = [

        (
            question,
            doc.get("content", "")
        )

        for doc in documents_to_rerank

    ]

    scores = model.predict(

        pairs,

        batch_size=BATCH_SIZE,

        show_progress_bar=False

    )

    for doc, score in zip(

        documents_to_rerank,

        scores

    ):

        doc["rerank_score"] = float(
            score
        )

    documents_to_rerank.sort(

        key=lambda x: x["rerank_score"],

        reverse=True

    )

    final_documents = (

        documents_to_rerank

        +

        documents[MAX_RERANK_DOCUMENTS:]

    )

    if documents_to_rerank:

        logger.info(
            "Best reranker score: %.3f",
            documents_to_rerank[0][
                "rerank_score"
            ]
        )

    logger.debug(
        "Top reranked documents:"
    )

    for index, doc in enumerate(

        documents_to_rerank,

        start=1

    ):

        logger.debug(

            "%d. %s | rerank_score=%.3f",

            index,

            doc.get(
                "source",
                "Unknown"
            ),

            doc["rerank_score"]

        )

    return final_documents