"""
Cross-Encoder Reranker

Uses a CrossEncoder model to rerank retrieved
documents according to their relevance to the
user's question.

Reranking can be enabled or disabled through
the AgroFlow configuration.
"""

from sentence_transformers import CrossEncoder

from chatbot.config import (
    RERANKER_MODEL,
    ENABLE_RERANKING
)

from chatbot.logger import logger


# ============================================================
# SETTINGS
# ============================================================

# Maximum number of documents that will be reranked.
MAX_RERANK_DOCUMENTS = 5

# Do not rerank when there are too few documents.
MIN_DOCUMENTS_TO_RERANK = 3

# Number of question-document pairs processed at once.
BATCH_SIZE = 16


# ============================================================
# GLOBAL RERANKER
# ============================================================

reranker = None


# ============================================================
# LOAD RERANKER
# ============================================================

def get_reranker():
    """
    Lazily load the CrossEncoder model.

    The model is loaded only once and then
    reused for subsequent requests.
    """

    global reranker

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if not ENABLE_RERANKING:

        logger.info(
            "Cross-Encoder reranking is disabled."
        )

        return None

    # --------------------------------------------------------
    # Load model only once
    # --------------------------------------------------------

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


# ============================================================
# RERANK DOCUMENTS
# ============================================================

def rerank_documents(
    question,
    documents
):
    """
    Rerank retrieved documents using a CrossEncoder.

    If ENABLE_RERANKING is False, the original
    document order is returned unchanged.

    Parameters
    ----------
    question : str
        User question.

    documents : list
        Retrieved documents.

    Returns
    -------
    list
        Reranked documents or original documents
        when reranking is disabled.
    """

    # --------------------------------------------------------
    # Check whether reranking is enabled
    # --------------------------------------------------------

    if not ENABLE_RERANKING:

        logger.info(
            "Reranking disabled. Returning documents "
            "without Cross-Encoder reranking."
        )

        return documents


    # --------------------------------------------------------
    # No documents
    # --------------------------------------------------------

    if not documents:

        logger.info(
            "Skipping reranking because no documents "
            "were retrieved."
        )

        return []


    # --------------------------------------------------------
    # Skip reranking if very few documents
    # --------------------------------------------------------

    if len(documents) <= MIN_DOCUMENTS_TO_RERANK:

        logger.info(
            "Skipping reranking (%d document(s) only).",
            len(documents)
        )

        return documents


    # --------------------------------------------------------
    # Only rerank the top retrieved documents
    # --------------------------------------------------------

    documents_to_rerank = documents[
        :MAX_RERANK_DOCUMENTS
    ]

    logger.info(
        "Reranking %d of %d retrieved document(s).",
        len(documents_to_rerank),
        len(documents)
    )


    # --------------------------------------------------------
    # Load reranker
    # --------------------------------------------------------

    model = get_reranker()

    if model is None:

        logger.info(
            "Reranker unavailable. Returning original "
            "document order."
        )

        return documents


    # --------------------------------------------------------
    # Build question-document pairs
    # --------------------------------------------------------

    pairs = [

        (
            question,
            doc.get("content", "")
        )

        for doc in documents_to_rerank

    ]


    # --------------------------------------------------------
    # Predict relevance scores
    # --------------------------------------------------------

    scores = model.predict(

        pairs,

        batch_size=BATCH_SIZE,

        show_progress_bar=False

    )


    # --------------------------------------------------------
    # Attach scores
    # --------------------------------------------------------

    for doc, score in zip(

        documents_to_rerank,

        scores

    ):

        doc["rerank_score"] = float(score)


    # --------------------------------------------------------
    # Sort reranked documents
    # --------------------------------------------------------

    documents_to_rerank.sort(

        key=lambda x: x["rerank_score"],

        reverse=True

    )


    # --------------------------------------------------------
    # Keep documents that were not reranked
    # --------------------------------------------------------

    final_documents = (

        documents_to_rerank

        +

        documents[MAX_RERANK_DOCUMENTS:]

    )


    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

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