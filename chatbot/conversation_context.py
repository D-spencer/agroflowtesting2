"""
AgroFlow AI Conversation Context
"""

import re
import uuid

from chatbot.config import MAX_HISTORY
from chatbot.logger import logger


CHAT_SESSION_TABLE = "ChatSession"

CHAT_MESSAGE_TABLE = "ChatMessage"


CONVERSATIONAL_PHRASES = {

    "hi",
    "hello",
    "hey",
    "hiya",
    "wassup",
    "sup",
    "hey there",
    "hi there",
    "hello there",
    "hey buddy",
    "hey friend",
    "hey pal",
    "hey mate",
    "hey dude",
    

    "good morning",
    "good afternoon",
    "good evening",
    "good night",
    "good day",

    "how are you",
    "how are you doing",
    "how's it going",
    "hows it going",
    "how is it going",
    "how's everything",
    "how's life",
    "how's your day",

    "thanks",
    "thank you",
    "thank you so much",
    "thanks a lot",
    "thank you very much",

    "okay",
    "ok",
    "alright",
    "all right",
    "sure",
    

    "continue",
    "go on",
    "explain more",
    "why",
    "how",
    "can you repeat that",
    "summarize",
    "give examples",
    "shorten this",
    "expand this",
    "next",

    "great",
    "nice",
    "perfect",

    "bye",
    "goodbye",
    "see you",
    "see you later",
    "talk to you later",
    "catch you later",
    "take care",
}


def _normalize_session_id(session_id):

    if session_id is None:
        return None

    session_id = str(
        session_id
    ).strip()

    if not session_id:
        return None

    return session_id


def _normalize_user_id(user_id):

    if user_id is None:
        return None

    user_id = str(
        user_id
    ).strip()

    if not user_id:
        return None

    return user_id


def normalize_message(
    message: str
) -> str:

    if not message:
        return ""

    message = message.lower().strip()

    message = re.sub(
        r"[^\w\s]",
        "",
        message
    )

    message = " ".join(
        message.split()
    )

    return message


def is_conversational(
    message: str
) -> bool:

    normalized = normalize_message(
        message
    )

    if not normalized:
        return False

    if normalized in CONVERSATIONAL_PHRASES:
        return True

    return False


def verify_session(
    supabase,
    session_id,
    user_id
):

    session_id = _normalize_session_id(
        session_id
    )

    user_id = _normalize_user_id(
        user_id
    )

    if not session_id:
        return False

    if not user_id:
        return False

    if supabase is None:

        logger.error(
            "Cannot verify session: Supabase client is unavailable."
        )

        return False

    try:

        response = (
            supabase
            .table(CHAT_SESSION_TABLE)
            .select("id")
            .eq("id", session_id)
            .eq("userId", user_id)
            .limit(1)
            .execute()
        )

        if response.data:
            return True

        logger.warning(
            "Session ownership verification failed "
            "for session=%s user=%s",
            session_id,
            user_id
        )

        return False

    except Exception:

        logger.exception(
            "Failed to verify ChatSession ownership."
        )

        return False


def get_history(
    supabase,
    session_id,
    user_id
):

    session_id = _normalize_session_id(
        session_id
    )

    user_id = _normalize_user_id(
        user_id
    )

    if not session_id:

        raise ValueError(
            "session_id is required when retrieving "
            "conversation history."
        )

    if not user_id:

        raise ValueError(
            "user_id is required when retrieving "
            "conversation history."
        )

    if supabase is None:

        logger.error(
            "Cannot retrieve conversation history: "
            "Supabase client is unavailable."
        )

        return []

    if not verify_session(
        supabase,
        session_id,
        user_id
    ):

        logger.warning(
            "Conversation history access denied "
            "for session=%s user=%s",
            session_id,
            user_id
        )

        return []

    try:

        response = (
            supabase
            .table(CHAT_MESSAGE_TABLE)
            .select(
                "id, sessionId, role, content, createdAt"
            )
            .eq(
                "sessionId",
                session_id
            )
            .order(
                "createdAt",
                desc=True
            )
            .limit(
                MAX_HISTORY
            )
            .execute()
        )

        messages = response.data or []

        messages.reverse()

        history = []

        for message in messages:

            role = message.get(
                "role"
            )

            content = message.get(
                "content"
            )

            if role not in {
                "user",
                "assistant"
            }:

                continue

            if not content:
                continue

            history.append({

                "role": role,

                "content": str(
                    content
                )

            })

        logger.info(
            "Loaded %d conversation message(s) "
            "for session %s.",
            len(history),
            session_id
        )

        return history

    except Exception:

        logger.exception(
            "Failed to retrieve conversation history "
            "for session %s.",
            session_id
        )

        return []


def add_message(
    supabase,
    session_id,
    user_id,
    role,
    content
):

    session_id = _normalize_session_id(
        session_id
    )

    user_id = _normalize_user_id(
        user_id
    )

    if not session_id:

        raise ValueError(
            "session_id is required."
        )

    if not user_id:

        raise ValueError(
            "user_id is required."
        )

    if not content:
        return False

    role = str(
        role
    ).strip().lower()

    if role not in {
        "user",
        "assistant"
    }:

        raise ValueError(
            "role must be either 'user' or 'assistant'."
        )

    if supabase is None:

        logger.error(
            "Cannot save conversation message: "
            "Supabase client is unavailable."
        )

        return False

    if not verify_session(
        supabase,
        session_id,
        user_id
    ):

        logger.warning(
            "Message rejected because session ownership "
            "could not be verified."
        )

        return False

    try:

        message_id = str(
            uuid.uuid4()
        )

        response = (
            supabase
            .table(CHAT_MESSAGE_TABLE)
            .insert({
                "id": message_id,
                "sessionId": session_id,
                "role": role,
                "content": str(content)
            })
            .execute()
        )

        if not response.data:

            logger.warning(
                "ChatMessage insert returned no data."
            )

            return False

        logger.info(
            "Saved %s message to ChatSession %s.",
            role,
            session_id
        )

        return True

    except Exception:

        logger.exception(
            "Failed to save %s message "
            "for session %s.",
            role,
            session_id
        )

        return False


def add_user_message(
    supabase,
    session_id,
    user_id,
    message
):

    if not message:
        return False

    return add_message(

        supabase=supabase,

        session_id=session_id,

        user_id=user_id,

        role="user",

        content=message

    )


def add_assistant_message(
    supabase,
    session_id,
    user_id,
    message
):

    if not message:
        return False

    return add_message(

        supabase=supabase,

        session_id=session_id,

        user_id=user_id,

        role="assistant",

        content=message

    )


def get_history_length(
    supabase,
    session_id,
    user_id
):

    history = get_history(

        supabase=supabase,

        session_id=session_id,

        user_id=user_id

    )

    return len(
        history
    )


def has_session(
    supabase,
    session_id,
    user_id
):

    return verify_session(

        supabase=supabase,

        session_id=session_id,

        user_id=user_id

    )


def clear_history(
    supabase,
    session_id,
    user_id
):

    session_id = _normalize_session_id(
        session_id
    )

    user_id = _normalize_user_id(
        user_id
    )

    if not session_id:
        return False

    if not user_id:
        return False

    if supabase is None:

        logger.error(
            "Cannot clear history: "
            "Supabase client is unavailable."
        )

        return False

    if not verify_session(
        supabase,
        session_id,
        user_id
    ):

        logger.warning(
            "History deletion denied for session %s.",
            session_id
        )

        return False

    try:

        (
            supabase
            .table(CHAT_MESSAGE_TABLE)
            .delete()
            .eq(
                "sessionId",
                session_id
            )
            .execute()
        )

        logger.info(
            "Conversation history cleared "
            "for session %s.",
            session_id
        )

        return True

    except Exception:

        logger.exception(
            "Failed to clear conversation history "
            "for session %s.",
            session_id
        )

        return False


def remove_session(
    supabase,
    session_id,
    user_id
):

    return clear_history(

        supabase=supabase,

        session_id=session_id,

        user_id=user_id

    )