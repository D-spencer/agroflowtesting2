# """
# Builds the context passed to the LLM
# from retrieved knowledge base documents.
# """

# import os

# from chatbot.logger import logger


# # ============================================================
# # CLEAN DOCUMENT NAME
# # ============================================================

# def clean_document_name(document):

#     """
#     Returns a clean document name.

#     Priority

#     1. title
#     2. filename
#     """

#     title = document.get("title")

#     if (

#         title

#         and

#         title.strip()

#         and

#         title.lower() != "unknown document"

#     ):

#         return title.strip()

#     source = document.get(

#         "source",

#         "Unknown Source"

#     )

#     filename = os.path.basename(source)

#     filename = os.path.splitext(filename)[0]

#     filename = filename.replace("_", " ")

#     filename = filename.replace("-", " ")

#     filename = " ".join(filename.split())

#     return filename


# # ============================================================
# # BUILD RAG CONTEXT
# # ============================================================

# def build_context(documents):

#     """
#     Builds a clean context for the LLM.

#     The model should understand where each
#     chunk came from without ever seeing
#     "Document 1", "Document 2", etc.
#     """

#     if not documents:

#         logger.info(

#             "No documents available."

#         )

#         return ""

#     logger.info(

#         "Building context from %d document(s).",

#         len(documents)

#     )

#     sections = []

#     for doc in documents:

#         source = clean_document_name(doc)

#         content = doc.get(

#             "content",

#             ""

#         ).strip()

#         sections.append(

# f"""
# Knowledge Source

# Source:
# {source}

# Content:
# {content}
# """
#         )

#     logger.info(

#         "Context built successfully."

#     )

#     return "\n\n".join(sections)









"""
Builds the context passed to the LLM
from retrieved knowledge base documents.
"""

import os

from chatbot.logger import logger


# ============================================================
# CONFIGURATION
# ============================================================

# Total amount of context (characters) that can be sent
# to the LLM regardless of the number of retrieved chunks.
TOTAL_CONTEXT_BUDGET = 2000

# Maximum characters allowed for a single chunk.
MAX_PER_CHUNK = 1800

# Minimum useful size for a chunk.
MIN_PER_CHUNK = 450


# ============================================================
# CLEAN DOCUMENT NAME
# ============================================================

def clean_document_name(document):
    """
    Returns a clean document name.

    Priority:
        1. title
        2. filename
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


# ============================================================
# CALCULATE DYNAMIC CHUNK LIMIT
# ============================================================

def get_chunk_limit(num_documents: int) -> int:
    """
    Dynamically allocate the total context budget
    across all retrieved documents.
    """

    if num_documents <= 0:
        return MAX_PER_CHUNK

    limit = TOTAL_CONTEXT_BUDGET // num_documents

    limit = max(limit, MIN_PER_CHUNK)

    limit = min(limit, MAX_PER_CHUNK)

    return limit


# ============================================================
# CLEAN CONTENT
# ============================================================

def clean_content(text: str, max_length: int) -> str:
    """
    Cleans retrieved content before sending
    it to the LLM.
    """

    if not text:
        return ""

    # Remove unnecessary whitespace
    text = " ".join(text.split())

    # Already short enough
    if len(text) <= max_length:
        return text

    shortened = text[:max_length]

    # Try to stop at the end of a sentence
    last_period = shortened.rfind(".")

    if last_period > max_length * 0.6:
        shortened = shortened[:last_period + 1]

    return shortened.rstrip() + " ..."


# ============================================================
# BUILD RAG CONTEXT
# ============================================================

def build_context(documents):
    """
    Builds a lightweight context for the LLM.

    Uses a dynamic context budget so that:
    - Few retrieved documents get more space.
    - Many retrieved documents each get a fair share.
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

    chunk_limit = get_chunk_limit(len(documents))

    logger.info(
        "Dynamic chunk limit: %d characters",
        chunk_limit
    )

    sections = []

    for doc in documents:

        source = clean_document_name(doc)

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

    return "\n\n".join(sections)