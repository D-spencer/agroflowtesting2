"""
History-Aware Question Rewriter

Converts follow-up questions into
standalone agricultural questions using
conversation history and the current topic.
"""

from chatbot.llm import ask_llm
from chatbot.logger import logger


# ============================================================
# SYSTEM PROMPT
# ============================================================

HISTORY_REWRITE_SYSTEM_PROMPT = """
You are an expert agricultural AI assistant.

Your task is to rewrite ONLY the user's latest question
into a complete standalone agricultural question.

You are given:

1. The current agricultural topic.
2. The recent conversation history.
3. The user's latest question.

Use conversation history to resolve references such as:

- it
- they
- them
- this
- that
- these
- those
- the crop
- the plant
- the disease
- the animal
- the fertilizer
- the treatment

Rules:

1. Preserve the user's original meaning.

2. Use conversation history as the primary source
   for resolving ambiguity.

3. Use the current topic when it clearly helps.

4. If the question is already standalone,
   return it unchanged.

5. Do not add new information.

6. Return ONLY the rewritten question.

Do not answer the question.

Do not explain your reasoning.
"""


# ============================================================
# BUILD HISTORY
# ============================================================

def build_history(history):
    """
    Convert conversation history into
    readable format for the LLM.
    """

    if not history:
        return ""

    return "\n".join(
        f"[{message.get('role', 'user').upper()}] "
        f"{message.get('content', '').strip()}"
        for message in history
        if message.get("content")
    )


# ============================================================
# REWRITE QUESTION
# ============================================================

def rewrite_with_history(
    llm,
    history,
    current_topic,
    question
):
    """
    Rewrite a follow-up question into
    a standalone agricultural question.
    """

    if not history and not current_topic:

        logger.info(
            "No history or topic available. Skipping history rewrite."
        )

        return question


    history_text = build_history(history)

    current_topic = current_topic or "No current topic available."


    messages = [

        {
            "role": "system",
            "content": HISTORY_REWRITE_SYSTEM_PROMPT
        },

        {
            "role": "system",
            "content": f"""
Current Agricultural Topic:

{current_topic}
"""
        },

        {
            "role": "user",
            "content": f"""
Conversation History:

{history_text}


Current Question:

{question}


Rewrite the current question into a standalone agricultural question.

Return ONLY the rewritten question.
"""
        }

    ]


    try:

        rewritten = ask_llm(
            client=llm,
            messages=messages,
            temperature=0,
            max_tokens=100
        ).strip()


    except Exception:

        logger.exception(
            "History-aware rewriting failed."
        )

        return question


    if not rewritten:

        logger.info(
            "History rewriter returned an empty response."
        )

        return question


    logger.info(
        f"History rewrite: '{question}' -> '{rewritten}'"
    )

    return rewritten