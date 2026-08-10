# ============================================================
# CONFIGURATION
# ============================================================

from chatbot.config import MAX_HISTORY

# ============================================================
# CONVERSATION MEMORY
# ============================================================

conversation_history = []


# ============================================================
# GET RECENT HISTORY
# ============================================================

def get_history():
    """
    Returns the most recent conversation history.
    """
    max_messages = MAX_HISTORY * 2
    return conversation_history[-max_messages:]

# ============================================================
# ADD USER MESSAGE
# ============================================================

def add_user_message(message):

    conversation_history.append(

        {
            "role": "user",
            "content": message
        }

    )

    trim_history()


# ============================================================
# ADD ASSISTANT MESSAGE
# ============================================================

def add_assistant_message(message):

    conversation_history.append(

        {
            "role": "assistant",
            "content": message
        }

    )

    trim_history()


# ============================================================
# ADD COMPLETE CONVERSATION
# ============================================================

def add_conversation(

    user_message,

    assistant_message

):

    add_user_message(user_message)

    add_assistant_message(assistant_message)


# ============================================================
# CLEAR MEMORY
# ============================================================

def clear_history():

    conversation_history.clear()


# ============================================================
# TRIM MEMORY
# ============================================================

def trim_history():

    max_messages = MAX_HISTORY * 2

    if len(conversation_history) > max_messages:

        del conversation_history[:-max_messages]