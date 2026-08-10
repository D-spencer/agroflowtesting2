"""
AgroFlow AI Retrieval Pipeline

Responsible for:

- Optional query rewriting
- Multi-query generation
- Vector search
- Keyword search
- Reciprocal Rank Fusion
- Optional Cross-Encoder reranking
- Context compression
"""

from chatbot.config import (
    TOP_K,
    ENABLE_RERANKING,
    ENABLE_QUERY_REWRITE
)

from chatbot.query_rewriter import rewrite_query

from chatbot.multi_query import generate_multi_queries

from chatbot.embeddings import create_query_embedding

from chatbot.database import (
    retrieve_documents,
    keyword_search
)

from chatbot.rrf import reciprocal_rank_fusion

from chatbot.reranker import rerank_documents

from chatbot.context_compressor import compress_context

from chatbot.logger import logger


# ============================================================
# SETTINGS
# ============================================================

# Maximum number of search queries used by the retrieval system.
#
# A smaller number reduces:
#
# - LLM token usage
# - Embedding generation
# - Database queries
# - Retrieval latency
#
MAX_SEARCH_QUERIES = 3


# ============================================================
# REMOVE DUPLICATE QUERIES
# ============================================================

def remove_duplicate_queries(queries):
    """
    Remove duplicate search queries while preserving order.

    Comparison is case-insensitive.
    """

    unique_queries = []

    seen = set()

    for query in queries:

        if not query:
            continue

        query = query.strip()

        if not query:
            continue

        key = query.lower()

        if key in seen:
            continue

        seen.add(key)

        unique_queries.append(query)

    return unique_queries


# ============================================================
# RETRIEVE RELEVANT DOCUMENTS
# ============================================================

def retrieve_context(
    question,
    embedding_model,
    supabase,
    llm,
    match_count=TOP_K
):
    """
    Retrieve the most relevant agricultural documents.

    Pipeline:

        Question
            ↓
        Optional Query Rewrite
            ↓
        Multi-Query Generation
            ↓
        Vector Search
            ↓
        Keyword Search
            ↓
        Reciprocal Rank Fusion
            ↓
        Optional Reranking
            ↓
        Context Compression
            ↓
        Final Documents
    """

    logger.info(
        "Starting retrieval pipeline."
    )


    # ========================================================
    # QUERY REWRITE
    # ========================================================

    if ENABLE_QUERY_REWRITE:

        logger.info(
            "Query rewriting enabled."
        )

        try:

            search_query = rewrite_query(
                llm,
                question
            )

            if not search_query.strip():

                logger.warning(
                    "Query rewrite returned empty result. "
                    "Using original question."
                )

                search_query = question

        except Exception:

            logger.exception(
                "Query rewriting failed. "
                "Using original question."
            )

            search_query = question

    else:

        logger.info(
            "Query rewriting disabled. "
            "Using original question."
        )

        search_query = question


    logger.info(
        "Search query: %s",
        search_query
    )


    # ========================================================
    # GENERATE MULTIPLE SEARCH QUERIES
    # ========================================================

    try:

        search_queries = generate_multi_queries(
            llm,
            search_query
        )

        if not search_queries:

            search_queries = [
                search_query
            ]

    except Exception:

        logger.exception(
            "Multi-query generation failed. "
            "Using search query only."
        )

        search_queries = [
            search_query
        ]


    # ========================================================
    # REMOVE DUPLICATE QUERIES
    # ========================================================

    search_queries = remove_duplicate_queries(
        search_queries
    )


    # ========================================================
    # LIMIT NUMBER OF SEARCH QUERIES
    # ========================================================

    search_queries = search_queries[
        :MAX_SEARCH_QUERIES
    ]


    # ========================================================
    # SAFETY FALLBACK
    # ========================================================

    if not search_queries:

        search_queries = [
            search_query
        ]


    logger.info(
        "Using %d search quer%s.",
        len(search_queries),
        "y"
        if len(search_queries) == 1
        else "ies"
    )


    for index, query in enumerate(
        search_queries,
        start=1
    ):

        logger.debug(
            "Search Query %d: %s",
            index,
            query
        )


    # ========================================================
    # VECTOR SEARCH
    # ========================================================

    vector_rank_lists = []


    for query in search_queries:

        logger.info(
            "Vector search: %s",
            query
        )


        # ----------------------------------------------------
        # CREATE QUERY EMBEDDING
        # ----------------------------------------------------

        embedding = create_query_embedding(

            embedding_model,

            query

        )


        logger.info(
            "Query embedding generated."
        )


        # ----------------------------------------------------
        # RETRIEVE DOCUMENTS
        # ----------------------------------------------------

        docs, _ = retrieve_documents(

            supabase=supabase,

            embedding=embedding,

            match_count=match_count

        )


        if docs:

            vector_rank_lists.append(
                docs
            )

            logger.info(
                "Retrieved %d vector document(s).",
                len(docs)
            )

        else:

            logger.info(
                "No vector documents retrieved."
            )


    # ========================================================
    # KEYWORD SEARCH
    # ========================================================

    # Use the main search query for keyword search.
    #
    # This avoids running additional keyword searches
    # for every generated query.

    keyword_docs = keyword_search(

        supabase=supabase,

        question=search_query,

        match_count=match_count

    )


    logger.info(
        "Keyword search returned %d document(s).",
        len(keyword_docs)
    )


    # ========================================================
    # RECIPROCAL RANK FUSION
    # ========================================================

    rank_lists = vector_rank_lists.copy()


    if keyword_docs:

        rank_lists.append(
            keyword_docs
        )


    if rank_lists:

        documents = reciprocal_rank_fusion(
            rank_lists
        )

    else:

        documents = []


    logger.info(
        "RRF produced %d document(s).",
        len(documents)
    )


    # ========================================================
    # OPTIONAL CROSS-ENCODER RERANKING
    # ========================================================

    if ENABLE_RERANKING:

        logger.info(
            "Cross-Encoder reranking enabled."
        )

        documents = rerank_documents(

            question,

            documents

        )

        logger.info(
            "Reranker returned %d document(s).",
            len(documents)
        )

    else:

        logger.info(
            "Cross-Encoder reranking disabled."
        )


    # ========================================================
    # CONTEXT COMPRESSION
    # ========================================================

    documents = compress_context(
        documents
    )


    logger.info(
        "Context compressed to %d document(s).",
        len(documents)
    )


    # ========================================================
    # FINAL SCORE
    # ========================================================

    if documents:

        if ENABLE_RERANKING:

            best_score = documents[0].get(

                "rerank_score",

                documents[0].get(
                    "rrf_score",
                    0.0
                )

            )

        else:

            best_score = documents[0].get(

                "rrf_score",

                0.0

            )

    else:

        best_score = 0.0


    logger.info(
        "Best retrieval score: %.4f",
        best_score
    )


    # ========================================================
    # DEBUG LOGGING
    # ========================================================

    logger.debug(
        "Final retrieved documents:"
    )


    for index, doc in enumerate(

        documents,

        start=1

    ):

        logger.debug(

            "%d. %s | RRF=%.6f | Rerank=%.3f",

            index,

            doc.get(
                "source",
                "Unknown"
            ),

            doc.get(
                "rrf_score",
                0.0
            ),

            doc.get(
                "rerank_score",
                0.0
            )

        )


    # ========================================================
    # COMPLETE
    # ========================================================

    logger.info(
        "Retrieval pipeline completed successfully."
    )


    return documents, best_score