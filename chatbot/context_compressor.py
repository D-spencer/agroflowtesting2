"""
AgroFlow AI Context Compressor
"""

from chatbot.config import TOP_CONTEXT_CHUNKS

from chatbot.logger import logger


def remove_duplicate_documents(documents):

    unique_documents = []

    seen_content = set()

    duplicates_removed = 0

    for document in documents:

        content = document.get(
            "content",
            ""
        ).strip().lower()

        if content in seen_content:

            duplicates_removed += 1

            continue

        seen_content.add(
            content
        )

        unique_documents.append(
            document
        )

    logger.info(
        "Removed %d duplicate document(s).",
        duplicates_removed
    )

    return unique_documents


def sort_documents(documents):

    return sorted(

        documents,

        key=lambda document: (

            document.get(
                "rerank_score",
                document.get(
                    "rrf_score",
                    0.0
                )
            )

        ),

        reverse=True
    )


def compress_context(documents):

    if not documents:

        logger.info(
            "No documents available for context compression."
        )

        return []

    original_count = len(
        documents
    )

    documents = remove_duplicate_documents(
        documents
    )

    documents = sort_documents(
        documents
    )

    documents = documents[
        :TOP_CONTEXT_CHUNKS
    ]

    logger.info(
        "Context compressed from %d to %d document(s).",
        original_count,
        len(documents)
    )

    return documents