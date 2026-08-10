"""
Citation utilities for AgroFlow AI.

Builds a clean source list for the user.
"""

import os

from chatbot.logger import logger


# ============================================================
# CLEAN SOURCE NAME
# ============================================================

def clean_source_name(document):

    """
    Returns a user-friendly document name.

    Priority:

    1. title
    2. filename without extension
    3. Unknown Source
    """

    title = document.get("title")

    if title:

        title = title.strip()

        if (

            title

            and

            title.lower() != "unknown document"

        ):

            return title

    source = document.get(

        "source",

        "Unknown Source"

    )

    filename = os.path.basename(source)

    filename = os.path.splitext(filename)[0]

    filename = filename.replace("_", " ")

    filename = filename.replace("-", " ")

    filename = " ".join(filename.split())

    if filename:

        return filename

    return "Unknown Source"


# ============================================================
# REMOVE DUPLICATE SOURCES
# ============================================================

def remove_duplicate_sources(documents):

    unique = []

    seen = set()

    for doc in documents:

        name = clean_source_name(doc)

        if name in seen:

            continue

        seen.add(name)

        unique.append(doc)

    logger.info(

        "Unique sources: %d",

        len(unique)

    )

    return unique


# ============================================================
# SORT SOURCES
# ============================================================

def sort_sources(documents):

    return sorted(

        documents,

        key=lambda d: clean_source_name(d).lower()

    )


# ============================================================
# BUILD SOURCES
# ============================================================

def build_citations(documents):

    """
    Builds a clean list of source documents.

    Example

    AgroFlow Knowledge Base
    ----------------------------------------
    Sources

    1. Rice Production Manual

    2. Maize Production Training Manual
    """

    if not documents:

        logger.info(

            "No sources available."

        )

        return ""

    documents = remove_duplicate_sources(

        documents

    )

    documents = sort_sources(

        documents

    )

    lines = [

        "",

        "AgroFlow Knowledge Base",

        "-" * 40,

        "Sources",

        ""

    ]

    for index, document in enumerate(

        documents,

        start=1

    ):

        lines.append(

            f"{index}. {clean_source_name(document)}"

        )

        lines.append("")

    logger.info(

        "Built %d source(s).",

        len(documents)

    )

    return "\n".join(lines).rstrip()