"""
Builds the context passed to the LLM
from retrieved knowledge base documents.
"""

import os

from chatbot.logger import logger


TOTAL_CONTEXT_BUDGET = 2000

MAX_PER_CHUNK = 1800

MIN_PER_CHUNK = 450


def clean_document_name(document):
    """
    Returns a clean document name using the title
    or filename as a fallback.
    """

    title = document.get("title")

    if (
        title
        and title.strip()
        and title.lower() != "unknown document"
    ):
        return title.strip()

    source = document.get(
        "source",
        "Unknown Source"
    )

    filename = os.path.basename(source)

    filename = os.path.splitext(filename)[0]

    filename = filename.replace("_", " ")

    filename = filename.replace("-", " ")

    filename = " ".join(filename.split())

    return filename


def get_chunk_limit(num_documents: int) -> int:
    """
    Dynamically allocate the total context budget
    across retrieved documents.
    """

    if num_documents <= 0:
        return MAX_PER_CHUNK

    limit = TOTAL_CONTEXT_BUDGET // num_documents

    limit = max(
        limit,
        MIN_PER_CHUNK
    )

    limit = min(
        limit,
        MAX_PER_CHUNK
    )

    return limit


def clean_content(
    text: str,
    max_length: int
) -> str:
    """
    Cleans retrieved content before sending it
    to the LLM.
    """

    if not text:
        return ""

    text = " ".join(
        text.split()
    )

    if len(text) <= max_length:
        return text

    shortened = text[:max_length]

    last_period = shortened.rfind(".")

    if last_period > max_length * 0.6:
        shortened = shortened[
            :last_period + 1
        ]

    return shortened.rstrip() + " ..."


def build_context(documents):
    """
    Builds a lightweight context for the LLM
    using the retrieved documents.
    """

    if not documents:

        logger.info(
            "No documents available."
        )

        return ""

    logger.info(
        "Building context from %d document(s).",
        len(documents)
    )

    chunk_limit = get_chunk_limit(
        len(documents)
    )

    logger.info(
        "Dynamic chunk limit: %d characters",
        chunk_limit
    )

    sections = []

    for doc in documents:

        source = clean_document_name(
            doc
        )

        content = clean_content(

            doc.get(
                "content",
                ""
            ),

            chunk_limit

        )

        sections.append(
            f"""Source: {source}

{content}"""
        )

    logger.info(
        "Context built successfully."
    )

    return "\n\n".join(
        sections
    )