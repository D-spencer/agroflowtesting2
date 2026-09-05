"""
AgroFlow AI CLI

Terminal interface for testing the AgroFlow AI system.
"""

import uuid
from datetime import datetime

from chatbot.database import get_supabase
from chatbot.embeddings import load_embedding_model
from chatbot.llm import get_llm
from chatbot.chat_engine import chat
from chatbot.weather_location import get_default_location
from chatbot.logger import logger


CHAT_SESSION_TABLE = "ChatSession"
USER_TABLE = "User"

CLI_USER_ID = "cmtn4m5fy00018y56r3xeiedm"

CLI_SESSION_ID = str(
    uuid.uuid4()
)

CLI_SESSION_TITLE = "CLI Test Session(dunsin2)"

CLI_LATITUDE, CLI_LONGITUDE = get_default_location()


def create_test_session(
    supabase,
    user_id,
    session_id
):

    if not user_id:

        logger.error(
            "Cannot create CLI session: user_id is missing."
        )

        return False

    if (
        not session_id
        or not str(session_id).strip()
    ):

        logger.error(
            "Cannot create CLI session: session_id is missing."
        )

        return False

    updated_at = datetime.utcnow().isoformat()

    try:

        logger.info(
            "Creating CLI ChatSession: "
            "session_id=%s, user_id=%s",
            session_id,
            user_id
        )

        response = (
            supabase
            .table(CHAT_SESSION_TABLE)
            .insert({
                "id": session_id,
                "userId": user_id,
                "title": CLI_SESSION_TITLE,
                "updatedAt": updated_at
            })
            .execute()
        )

        if not response.data:

            logger.error(
                "ChatSession creation returned no data."
            )

            return False

        logger.info(
            "CLI ChatSession created successfully."
        )

        return True

    except Exception:

        logger.exception(
            "Failed to create CLI ChatSession."
        )

        return False


def verify_test_user(
    supabase,
    user_id
):

    if not user_id:

        logger.error(
            "CLI test user ID is missing."
        )

        return False

    try:

        logger.info(
            "Verifying CLI test user..."
        )

        response = (
            supabase
            .table(USER_TABLE)
            .select("id")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )

        if not response.data:

            logger.error(
                "CLI test user does not exist."
            )

            return False

        logger.info(
            "CLI test user verified."
        )

        return True

    except Exception:

        logger.exception(
            "Failed to verify CLI test user."
        )

        return False


def display_startup_information():

    print(
        "\n" + "=" * 70
    )

    print(
        "AgroFlow AI"
    )

    print(
        "CLI Test Mode"
    )

    print(
        "Type 'exit' to quit."
    )

    print(
        "=" * 70
    )

    print(
        f"\nTest User ID: {CLI_USER_ID}"
    )

    print(
        f"Session ID:   {CLI_SESSION_ID}"
    )

    print(
        f"Session Title: {CLI_SESSION_TITLE}"
    )

    print(
        f"Weather Location: {CLI_LATITUDE}, {CLI_LONGITUDE}"
    )

    print()


def initialize_agroflow():

    logger.info(
        "Initializing AgroFlow AI..."
    )

    try:

        supabase = get_supabase()

        logger.info(
            "Supabase connected."
        )

        if not verify_test_user(
            supabase=supabase,
            user_id=CLI_USER_ID
        ):

            print(
                "\nERROR: The configured CLI_USER_ID does "
                "not exist in the User table."
            )

            print(
                "\nCLI_USER_ID:"
            )

            print(
                CLI_USER_ID
            )

            print(
                "\nCreate a test user in your application "
                "first, then replace CLI_USER_ID with that "
                "user's ID."
            )

            return None

        session_created = create_test_session(
            supabase=supabase,
            user_id=CLI_USER_ID,
            session_id=CLI_SESSION_ID
        )

        if not session_created:

            print(
                "\nFailed to create CLI ChatSession."
            )

            print(
                "Check the logs and verify your "
                "ChatSession table structure."
            )

            return None

        logger.info(
            "CLI ChatSession ready."
        )

        embedding_model = load_embedding_model()

        logger.info(
            "Embedding model loaded."
        )

        llm = get_llm()

        logger.info(
            "LLM initialized."
        )

        logger.info(
            "AgroFlow AI is ready."
        )

        print(
            "\nAgroFlow AI is ready."
        )

        print(
            "This CLI is using a real Supabase ChatSession."
        )

        print(
            "Conversation history will be isolated to this "
            "session."
        )

        print(
            "You can start chatting."
        )

        return {
            "supabase": supabase,
            "embedding_model": embedding_model,
            "llm": llm
        }

    except Exception:

        logger.exception(
            "Failed to initialize AgroFlow AI."
        )

        print(
            "\nFailed to initialize AgroFlow AI."
        )

        print(
            "Check the logs for more information."
        )

        return None


def run_chat_loop(
    supabase,
    embedding_model,
    llm
):

    while True:

        try:

            question = input(
                "\nYou: "
            ).strip()

            if not question:
                continue

            if question.lower() in {
                "exit",
                "quit",
                "bye"
            }:

                logger.info(
                    "User ended the CLI session."
                )

                print(
                    "\nGoodbye!"
                )

                break

            logger.info(
                "User Question: %s",
                question
            )

            result = chat(
                question=question,
                supabase=supabase,
                embedding_model=embedding_model,
                llm=llm,
                session_id=CLI_SESSION_ID,
                user_id=CLI_USER_ID,
                latitude=CLI_LATITUDE,
                longitude=CLI_LONGITUDE
            )

            if not isinstance(
                result,
                dict
            ):

                logger.error(
                    "chat() returned an unexpected result: %r",
                    result
                )

                print(
                    "\nAgroFlow AI returned an unexpected response."
                )

                continue

            answer = result.get(
                "response",
                ""
            )

            topic = result.get(
                "topic"
            )

            logger.info(
                "Answer generated successfully."
            )

            print(
                "\n" + "=" * 70
            )

            print(
                "AgroFlow AI"
            )

            print(
                "=" * 70
            )

            print(
                answer
            )

            if topic:

                print(
                    f"\n[Current topic: {topic}]"
                )

            print(
                "=" * 70
            )

        except KeyboardInterrupt:

            logger.info(
                "CLI session interrupted by user."
            )

            print(
                "\n\nSession terminated."
            )

            break

        except Exception:

            logger.exception(
                "Unexpected error during chat session."
            )

            print(
                "\nAn unexpected error occurred."
            )

            print(
                "Please try again."
            )


def run_cli():

    display_startup_information()

    if (
        not CLI_USER_ID
        or CLI_USER_ID == "REPLACE_WITH_REAL_TEST_USER_ID"
    ):

        print(
            "\nERROR: CLI_USER_ID has not been configured."
        )

        print(
            "\nOpen cli.py and replace:"
        )

        print(
            'CLI_USER_ID = "REPLACE_WITH_REAL_TEST_USER_ID"'
        )

        print(
            "\nwith the ID of an existing test user "
            "from your Supabase User table."
        )

        return

    components = initialize_agroflow()

    if components is None:
        return

    supabase = components[
        "supabase"
    ]

    embedding_model = components[
        "embedding_model"
    ]

    llm = components[
        "llm"
    ]

    run_chat_loop(
        supabase=supabase,
        embedding_model=embedding_model,
        llm=llm
    )


if __name__ == "__main__":

    run_cli()