"""
AgroFlow AI Topic Memory

Stores and retrieves the current topic for a ChatSession.
"""

from datetime import datetime

from chatbot.logger import logger


CHAT_SESSION_TABLE = "ChatSession"


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


def _normalize_topic(topic):

    if topic is None:
        return None

    topic = str(
        topic
    ).strip()

    if not topic:
        return None

    return topic


def set_topic(
    supabase,
    session_id,
    topic,
    user_id=None
):
    """
    Store the current topic for a ChatSession.
    """

    session_id = _normalize_session_id(
        session_id
    )

    topic = _normalize_topic(
        topic
    )

    user_id = _normalize_user_id(
        user_id
    )

    if not session_id:

        logger.warning(
            "Cannot set topic: session_id is missing."
        )

        return False

    if not topic:

        logger.warning(
            "Cannot set topic: topic is empty."
        )

        return False

    if supabase is None:

        logger.error(
            "Cannot set topic: Supabase client is unavailable."
        )

        return False

    logger.info(
        "Attempting to update topic: "
        "session_id=%s, user_id=%s, topic=%s",
        session_id,
        user_id,
        topic
    )

    try:

        update_data = {

            "currentTopic": topic,

            "updatedAt": datetime.utcnow().isoformat()

        }

        query = (

            supabase
            .table(CHAT_SESSION_TABLE)
            .update(update_data)
            .eq(
                "id",
                session_id
            )

        )

        if user_id:

            query = query.eq(
                "userId",
                user_id
            )

        response = (

            query
            .select(
                "id, userId, currentTopic, updatedAt"
            )
            .execute()

        )

        if not response.data:

            logger.error(
                "Topic update affected ZERO rows. "
                "Possible causes: session does not exist, "
                "userId does not match, or Supabase RLS "
                "is blocking the update. "
                "session_id=%s, user_id=%s",
                session_id,
                user_id
            )

            return False

        updated_row = response.data[0]

        saved_topic = updated_row.get(
            "currentTopic"
        )

        logger.info(
            "Topic successfully updated in Supabase. "
            "session_id=%s, topic=%s",
            session_id,
            saved_topic
        )

        logger.info(
            "Updated ChatSession row: %s",
            updated_row
        )

        return True

    except Exception:

        logger.exception(
            "Failed to save topic for session %s.",
            session_id
        )

        return False


def get_topic(
    supabase,
    session_id,
    user_id=None
):
    """
    Retrieve the current topic for a ChatSession.
    """

    session_id = _normalize_session_id(
        session_id
    )

    user_id = _normalize_user_id(
        user_id
    )

    if not session_id:

        logger.warning(
            "Cannot get topic: session_id is missing."
        )

        return None

    if supabase is None:

        logger.error(
            "Cannot get topic: Supabase client is unavailable."
        )

        return None

    try:

        query = (

            supabase
            .table(CHAT_SESSION_TABLE)
            .select(
                "id, userId, currentTopic"
            )
            .eq(
                "id",
                session_id
            )

        )

        if user_id:

            query = query.eq(
                "userId",
                user_id
            )

        response = (

            query
            .maybe_single()
            .execute()

        )

        if not response.data:

            logger.warning(
                "No ChatSession found while retrieving "
                "topic. session_id=%s, user_id=%s",
                session_id,
                user_id
            )

            return None

        topic = response.data.get(
            "currentTopic"
        )

        if topic is None:

            logger.info(
                "ChatSession %s currently has no topic.",
                session_id
            )

            return None

        topic = str(
            topic
        ).strip()

        if not topic:
            return None

        logger.info(
            "Retrieved topic for session %s: %s",
            session_id,
            topic
        )

        return topic

    except Exception:

        logger.exception(
            "Failed to retrieve topic for session %s.",
            session_id
        )

        return None


def clear_topic(
    supabase,
    session_id,
    user_id=None
):
    """
    Clear the current topic for a ChatSession.
    """

    session_id = _normalize_session_id(
        session_id
    )

    user_id = _normalize_user_id(
        user_id
    )

    if not session_id:

        logger.warning(
            "Cannot clear topic: session_id is missing."
        )

        return False

    if supabase is None:

        logger.error(
            "Cannot clear topic: Supabase client is unavailable."
        )

        return False

    try:

        update_data = {

            "currentTopic": None,

            "updatedAt": datetime.utcnow().isoformat()

        }

        query = (

            supabase
            .table(CHAT_SESSION_TABLE)
            .update(update_data)
            .eq(
                "id",
                session_id
            )

        )

        if user_id:

            query = query.eq(
                "userId",
                user_id
            )

        response = (

            query
            .select(
                "id, userId, currentTopic, updatedAt"
            )
            .execute()

        )

        if not response.data:

            logger.warning(
                "No ChatSession was updated while "
                "clearing topic. session_id=%s",
                session_id
            )

            return False

        logger.info(
            "Topic cleared for session %s.",
            session_id
        )

        return True

    except Exception:

        logger.exception(
            "Failed to clear topic for session %s.",
            session_id
        )

        return False