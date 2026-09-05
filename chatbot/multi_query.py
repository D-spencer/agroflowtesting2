"""
Multi-Query Generator

Generates multiple agricultural search queries
from a user's original question to improve
retrieval coverage.
"""

import logging

from chatbot.llm import ask_llm


logger = logging.getLogger(__name__)


MULTI_QUERY_SYSTEM_PROMPT = """
You are an expert agricultural search assistant.

Your task is to generate multiple search queries
that would retrieve useful documents from an
agricultural knowledge base.

Rules:

1. Preserve the original meaning.
2. Use different wording for each query.
3. Expand abbreviations when appropriate.
4. Include agricultural terminology.
5. Keep each query concise.
6. Return ONLY the search queries.
7. One query per line.
8. Generate exactly 4 queries.

Do not number them.
Do not explain anything.
Do not answer the question.
"""


def generate_multi_queries(
    llm,
    question
):
    """
    Generate multiple search queries from a question.

    Duplicate queries are removed while preserving
    the original order.
    """

    messages = [

        {
            "role": "system",
            "content": MULTI_QUERY_SYSTEM_PROMPT
        },

        {
            "role": "user",
            "content": (
                "Generate four agricultural search queries for:\n\n"
                f"{question}"
            )
        }

    ]

    response = ask_llm(
        client=llm,
        messages=messages,
        temperature=0.2,
        max_tokens=200
    )

    queries = [
        line.strip()
        for line in response.splitlines()
        if line.strip()
    ]

    cleaned_queries = []

    for query in queries:

        if "." in query[:4]:

            query = query.split(
                ".",
                1
            )[1].strip()

        if ")" in query[:4]:

            query = query.split(
                ")",
                1
            )[1].strip()

        if query.startswith("-"):

            query = query[1:].strip()

        if not query:
            continue

        cleaned_queries.append(
            query
        )

    unique_queries = []

    seen = set()

    for query in cleaned_queries:

        key = " ".join(
            query.lower().split()
        )

        if key in seen:

            logger.debug(
                "Duplicate query removed: %s",
                query
            )

            continue

        seen.add(key)

        unique_queries.append(
            query
        )

    original_key = " ".join(
        question.lower().split()
    )

    if original_key not in seen:

        unique_queries.insert(
            0,
            question
        )

    unique_queries = unique_queries[:5]

    logger.info(
        "Multi-Query Generation"
    )

    logger.info(
        "Original Question: %s",
        question
    )

    logger.info(
        "Generated %d unique search quer%s.",
        len(unique_queries),
        "y"
        if len(unique_queries) == 1
        else "ies"
    )

    for i, query in enumerate(
        unique_queries,
        start=1
    ):

        logger.info(
            "Query %d: %s",
            i,
            query
        )

    return unique_queries