"""
AgroFlow AI Query Rewriter

Uses the LLM to rewrite user questions into
better search queries for the agricultural
knowledge base.

The rewriter is intentionally lightweight because
its purpose is retrieval, not answering the user.
"""

from chatbot.llm import ask_llm

from chatbot.logger import logger

from chatbot.config import (
    ENABLE_QUERY_REWRITE
)


# ============================================================
# QUERY REWRITE SYSTEM PROMPT
# ============================================================

QUERY_REWRITE_SYSTEM_PROMPT = """
You are an agricultural information retrieval expert.

Your task is to rewrite a user's agricultural question
into ONE concise search query for an agricultural
knowledge base.

Rules:

1. Preserve the user's original intent exactly.

2. Do not change the meaning of the question.

3. Expand important agricultural abbreviations when useful.

4. Include important agricultural terminology when it
   improves retrieval.

5. Remove conversational and unnecessary words.

6. Keep the query concise, preferably 5-15 words.

7. Do NOT answer the question.

8. Do NOT add information that is not implied by the
   user's question.

9. Return ONLY the rewritten search query.

No explanations.
No bullet points.
No numbering.
No quotation marks.
No labels such as "Query:".
"""


# ============================================================
# REWRITE QUERY
# ============================================================

def rewrite_query(
    llm,
    question
):
    """
    Rewrite the user's question into a concise
    agricultural search query.

    If query rewriting is disabled in config.py,
    the original question is returned immediately
    without making an LLM request.
    """

    # ========================================================
    # CHECK CONFIGURATION
    # ========================================================

    if not ENABLE_QUERY_REWRITE:

        logger.info(
            "Query rewriting disabled. "
            "Using original question."
        )

        return question.strip()


    # ========================================================
    # VALIDATE QUESTION
    # ========================================================

    if not question or not question.strip():

        logger.warning(
            "Empty question received for query rewriting."
        )

        return question


    question = question.strip()


    # ========================================================
    # LOG
    # ========================================================

    logger.info(
        "Rewriting search query."
    )


    # ========================================================
    # BUILD MESSAGES
    # ========================================================

    messages = [

        {
            "role": "system",
            "content": QUERY_REWRITE_SYSTEM_PROMPT
        },

        {
            "role": "user",
            "content":
                "Rewrite this agricultural question "
                "into a search query:\n\n"
                f"{question}"
        }

    ]


    # ========================================================
    # ASK LLM
    # ========================================================

    try:

        rewritten_query = ask_llm(

            client=llm,

            messages=messages,

            temperature=0.1,

            max_tokens=50

        )

    except Exception:

        logger.exception(
            "Query rewriting failed. "
            "Using original question."
        )

        return question


    # ========================================================
    # CLEAN OUTPUT
    # ========================================================

    if not rewritten_query:

        logger.warning(
            "Query rewrite returned an empty response. "
            "Using original question."
        )

        return question


    rewritten_query = rewritten_query.strip()


    # --------------------------------------------------------
    # Remove quotation marks
    # --------------------------------------------------------

    rewritten_query = rewritten_query.strip(
        "\"'"
    )


    # --------------------------------------------------------
    # Remove common labels
    # --------------------------------------------------------

    prefixes = [

        "Query:",

        "Search Query:",

        "Search query:",

        "Rewritten Query:",

        "Rewritten query:"

    ]


    for prefix in prefixes:

        if rewritten_query.startswith(prefix):

            rewritten_query = rewritten_query[
                len(prefix):
            ].strip()


    # --------------------------------------------------------
    # Keep only first line
    # --------------------------------------------------------

    if "\n" in rewritten_query:

        rewritten_query = rewritten_query.split(
            "\n",
            1
        )[0].strip()


    # --------------------------------------------------------
    # Remove quotation marks again
    # --------------------------------------------------------

    rewritten_query = rewritten_query.strip(
        "\"'"
    )


    # --------------------------------------------------------
    # Remove trailing punctuation
    # --------------------------------------------------------

    rewritten_query = rewritten_query.rstrip(
        "."
    ).strip()


    # ========================================================
    # FALLBACK
    # ========================================================

    if not rewritten_query:

        logger.warning(
            "Cleaned query rewrite is empty. "
            "Using original question."
        )

        rewritten_query = question


    # ========================================================
    # LOGGING
    # ========================================================

    logger.info(
        "Query rewritten successfully."
    )

    logger.debug(
        "Original Query: %s",
        question
    )

    logger.debug(
        "Rewritten Query: %s",
        rewritten_query
    )


    # ========================================================
    # CHECK WHETHER QUERY CHANGED
    # ========================================================

    if rewritten_query.lower() == question.lower():

        logger.debug(
            "Query remained unchanged."
        )


    # ========================================================
    # RETURN
    # ========================================================

    return rewritten_query