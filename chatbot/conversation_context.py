"""
Conversation Context

Responsible for:

- Maintaining conversation history
- Detecting simple conversational messages
- Supporting follow-up questions
- Helping determine when RAG can be skipped
"""

import re
from collections import deque

from chatbot.config import MAX_HISTORY


# ============================================================
# CONVERSATIONAL PHRASES
# ============================================================

CONVERSATIONAL_PHRASES = {
    "hi",
    "hello",
    "hey",
    "hiya",
    "wassup",

    "good morning",
    "good afternoon",
    "good evening",

    "thanks",
    "thank you",
    "thank you so much",
    "thanks a lot",

    "okay",
    "ok",
    "alright",
    "all right",

    "continue",
    "go on",

    "great",
    "nice",
    "perfect",
}


# ============================================================
# CONVERSATION HISTORY
# ============================================================

conversation_history = deque(
    maxlen=MAX_HISTORY
)


# ============================================================
# NORMALIZE MESSAGE
# ============================================================

def normalize_message(message: str) -> str:
    """
    Normalize a message for conversational
    intent detection.

    Examples:

        "Hello!"       -> "hello"
        "Good Morning" -> "good morning"
        "  Thanks! "   -> "thanks"
    """

    if not message:
        return ""

    message = message.lower().strip()

    # Remove punctuation
    message = re.sub(
        r"[^\w\s]",
        "",
        message
    )

    # Normalize whitespace
    message = " ".join(
        message.split()
    )

    return message


# ============================================================
# CHECK CONVERSATIONAL MESSAGE
# ============================================================

def is_conversational(message: str) -> bool:
    """
    Determine whether a message is a simple
    conversational input that does not require RAG.

    Examples:

        "Hi"
        "Hello"
        "Thanks"
        "Okay"
        "Continue"
        "Go on"

    Messages containing an actual agricultural
    question are not treated as conversational.

    Examples:

        "Hi, how do I plant maize?"
        "Thanks, what fertilizer should I use?"
    """

    normalized = normalize_message(
        message
    )

    if not normalized:
        return False

    # Exact conversational phrase
    if normalized in CONVERSATIONAL_PHRASES:
        return True

    return False


# ============================================================
# ADD USER MESSAGE
# ============================================================

def add_user_message(message: str):
    """
    Add a user message to conversation history.
    """

    if not message:
        return

    conversation_history.append({
        "role": "user",
        "content": message
    })


# ============================================================
# ADD ASSISTANT MESSAGE
# ============================================================

def add_assistant_message(message: str):
    """
    Add an assistant response to conversation history.
    """

    if not message:
        return

    conversation_history.append({
        "role": "assistant",
        "content": message
    })


# ============================================================
# GET HISTORY
# ============================================================

def get_history():
    """
    Return the current conversation history
    as a regular list.
    """

    return list(
        conversation_history
    )


# ============================================================
# CLEAR HISTORY
# ============================================================

def clear_history():
    """
    Clear the current conversation history.
    """

    conversation_history.clear()