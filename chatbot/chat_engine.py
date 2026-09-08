"""
AgroFlow AI Chat Engine
"""

from chatbot.config import (
    RELEVANCE_SCORE_THRESHOLD
)

from chatbot.retrieval import (
    retrieve_context
)

from chatbot.context_builder import (
    build_context
)

from chatbot.history_rewriter import (
    rewrite_with_history
)

from chatbot.topic_extractor import (
    extract_topic
)

from chatbot.topic_memory import (
    set_topic,
    get_topic
)

from chatbot.prompts import (
    build_rag_prompt,
    build_general_prompt
)

from chatbot.citations import (
    build_citations
)

from chatbot.llm import (
    ask_llm,
    format_markdown_response
)

from chatbot.messages import (
    build_messages
)

from chatbot.conversation_context import (
    is_conversational,
    add_user_message,
    add_assistant_message,
    get_history
)

from chatbot.weather import (
    get_formatted_weather
)

from chatbot.weather_context import (
    build_weather_context
)

from chatbot.logger import logger


WEATHER_KEYWORDS = {
    "weather",
    "rain",
    "rainfall",
    "forecast",
    "temperature",
    "humidity",
    "wind",
    "windy",
    "storm",
    "thunderstorm",
    "drizzle",
    "sunny",
    "cloudy",
    "overcast",
    "precipitation",
    "shower",
    "showers",
    "drought",
    "dry",
    "wet",
    "heat",
    "cold",
    "hot",
    "cool",
    "climate",
}


WEATHER_AGRICULTURAL_ACTIVITIES = {
    "plant",
    "planting",
    "planted",
    "sow",
    "sowing",
    "spray",
    "spraying",
    "sprayed",
    "irrigate",
    "irrigation",
    "fertilizer",
    "fertilize",
    "fertilizing",
    "harvest",
    "harvesting",
    "dry",
    "drying",
    "weed",
    "weeding",
    "transplant",
    "transplanting",
    "cultivate",
    "cultivation",
}


WEATHER_TIME_INDICATORS = {
    "today",
    "tomorrow",
    "tonight",
    "morning",
    "afternoon",
    "evening",
    "yesterday",
    "now",
    "currently",
    "current",
    "this",
    "next",
}


LOCATION_INDICATORS = {
    "location",
    "area",
    "place",
    "here",
    "nearby",
    "region",
    "farm",
    "field",
}


def is_weather_question(question: str) -> bool:

    if not question:
        return False

    normalized = (
        question
        .lower()
        .strip()
    )

    for character in (
        "?",
        ",",
        ".",
        "!",
        ";",
        ":",
        "(",
        ")",
        "'",
        '"'
    ):

        normalized = normalized.replace(
            character,
            " "
        )

    words = set(
        normalized.split()
    )

    if words.intersection(
        WEATHER_KEYWORDS
    ):

        return True

    has_agricultural_activity = bool(
        words.intersection(
            WEATHER_AGRICULTURAL_ACTIVITIES
        )
    )

    has_time_indicator = bool(
        words.intersection(
            WEATHER_TIME_INDICATORS
        )
    )

    has_location_indicator = bool(
        words.intersection(
            LOCATION_INDICATORS
        )
    )

    weather_intent_phrases = (
        "based on my location",
        "based on the weather",
        "based on weather",
        "according to the weather",
        "according to weather",
        "weather conditions",
        "weather forecast",
        "based on today's weather",
        "based on tomorrow's weather",
        "in my area",
        "in my location",
        "at my location",
        "where i am",
        "where i live",
        "can i plant",
        "should i plant",
        "can i spray",
        "should i spray",
        "can i irrigate",
        "should i irrigate",
        "can i harvest",
        "should i harvest",
        "can i sow",
        "should i sow",
        "can i dry",
        "should i dry",
        "can i weed",
        "should i weed",
        "can i transplant",
        "should i transplant",
    )

    for phrase in weather_intent_phrases:

        if phrase in normalized:
            return True

    if (
        has_agricultural_activity
        and
        has_time_indicator
    ):

        return True

    if (
        has_agricultural_activity
        and
        has_location_indicator
    ):

        return True

    return False


def build_chat_response(
    answer,
    supabase,
    session_id,
    user_id=None
):

    try:

        current_topic = get_topic(
            supabase=supabase,
            session_id=session_id,
            user_id=user_id
        )

    except Exception:

        logger.exception(
            "Failed to retrieve current topic "
            "for session %s.",
            session_id
        )

        current_topic = None

    return {
        "response": format_markdown_response(answer),
        "topic": current_topic
    }


def build_conversational_prompt(
    question,
    history=None
):

    history_text = ""

    if history:

        history_lines = []

        for message in history:

            role = message.get(
                "role",
                ""
            )

            content = message.get(
                "content",
                ""
            )

            if not content:
                continue

            history_lines.append(
                f"{role}: {content}"
            )

        if history_lines:

            history_text = (
                "\nPrevious conversation:\n"
                + "\n".join(history_lines)
            )

    return f"""
You are AgroFlow AI, a friendly agricultural assistant.

The user sent a simple conversational message.

Respond naturally and briefly.

Use valid Markdown. Keep a one-sentence reply as a plain paragraph; for a
longer reply, use `##` headings and Markdown lists with blank lines between
blocks. Do not use Unicode bullets, raw HTML, or decorative separator lines.

Rules:

- If the user greets you, greet them back warmly.
- If appropriate, mention that you are ready to help with agriculture or farming.
- If the user thanks you, respond naturally and briefly.
- If the user says okay, alright, great, nice, or perfect, acknowledge them naturally.
- If the user says continue, next, or go on, respond naturally based on the conversation when possible.
- If the user asks you to repeat, summarize, expand, shorten, translate, or rewrite something, follow the request using the conversation context when available.
- Do not force an agricultural explanation.
- Do not give unnecessary information.
- Do not mention the knowledge base.
- Do not mention retrieval.
- Do not mention system operations.

Previous conversation context:

{history_text}

User message:

{question}

Respond naturally.
"""


def save_conversation_exchange(
    supabase,
    session_id,
    user_id,
    question,
    answer
):

    try:

        user_saved = add_user_message(
            supabase=supabase,
            session_id=session_id,
            user_id=user_id,
            message=question
        )

        if not user_saved:

            logger.warning(
                "User message was not saved "
                "for session %s.",
                session_id
            )

    except Exception:

        logger.exception(
            "Failed to save user message "
            "for session %s.",
            session_id
        )

    try:

        assistant_saved = add_assistant_message(
            supabase=supabase,
            session_id=session_id,
            user_id=user_id,
            message=answer
        )

        if not assistant_saved:

            logger.warning(
                "Assistant message was not saved "
                "for session %s.",
                session_id
            )

    except Exception:

        logger.exception(
            "Failed to save assistant message "
            "for session %s.",
            session_id
        )

    logger.info(
        "Conversation exchange processed for session %s.",
        session_id
    )


def handle_conversational_message(
    session_id,
    question,
    llm,
    supabase,
    user_id=None
):

    logger.info(
        "Conversational message detected for session %s. "
        "Skipping topic extraction and RAG.",
        session_id
    )

    history = get_history(
        supabase=supabase,
        session_id=session_id,
        user_id=user_id
    )

    user_prompt = build_conversational_prompt(
        question=question,
        history=history
    )

    messages = build_messages(
        user_prompt
    )

    logger.info(
        "Sending conversational response to LLM..."
    )

    try:

        answer = ask_llm(
            client=llm,
            messages=messages
        )

    except Exception:

        logger.exception(
            "Conversational LLM request failed."
        )

        answer = (
            "I'm here to help. "
            "What would you like to know?"
        )

    logger.info(
        "Conversational response generated."
    )

    save_conversation_exchange(
        supabase=supabase,
        session_id=session_id,
        user_id=user_id,
        question=question,
        answer=answer
    )

    current_history = get_history(
        supabase=supabase,
        session_id=session_id,
        user_id=user_id
    )

    logger.info(
        "Session %s conversation now contains "
        "%d message(s).",
        session_id,
        len(current_history)
    )

    return build_chat_response(
        answer=answer,
        supabase=supabase,
        session_id=session_id,
        user_id=user_id
    )


def build_weather_prompt(
    question,
    weather_context,
    agricultural_context=""
):

    return f"""
You are AgroFlow AI, an intelligent agricultural assistant.

The user is asking a question that requires current or
forecast weather information.

Use the supplied weather information to answer the question
accurately and practically.

IMPORTANT RULES:

1. Use the supplied weather information as the PRIMARY source
   for current and forecast weather conditions.

2. Never invent weather conditions, temperatures, rainfall,
   humidity, wind speeds, precipitation probabilities, or
   forecast dates.

3. Never claim that a forecast is certain.

4. Clearly distinguish between:
   - what the weather data says
   - what the weather data suggests for the agricultural activity

5. If the supplied weather information does not contain
   enough information to answer the question confidently,
   say so.

6. When the question involves an agricultural activity such
   as planting, sowing, spraying, irrigation, harvesting,
   drying, weeding, transplanting, cultivation, or fertilizer
   application, explain how the available weather conditions
   affect that activity.

7. Give a practical recommendation when the available
   weather information supports one.

8. Consider multiple relevant weather variables rather than
   relying on only one variable.

9. For planting or sowing questions, consider:
   - rainfall
   - rain probability
   - temperature
   - excessive wetness
   - expected dry periods
   - suitability for field operations

10. For spraying questions, pay particular attention to:
    - rain probability
    - precipitation
    - wind speed
    - temperature
    - humidity when available
    - timing

11. For irrigation questions, consider:
    - expected rainfall
    - rainfall probability
    - current conditions
    - upcoming precipitation

12. For harvesting and drying questions, consider:
    - rainfall
    - rain probability
    - humidity when available
    - consecutive wet conditions
    - expected dry periods

13. Do not invent pesticide-specific, fertilizer-specific,
    or crop-specific requirements.

14. If product-specific information is required, tell the user
    to check the product label or relevant agricultural guidance.

15. Do not make a final agricultural decision using weather
    alone when important information is missing, such as soil
    condition, crop growth stage, product label requirements,
    or local field conditions.

16. If the question asks about a specific future date, use the
    forecast information corresponding to that date.

17. If the requested date is outside the supplied forecast
    period, clearly say that the supplied forecast does not
    cover that date.

18. Use simple English.

19. Give the user a direct answer first.

20. Keep the answer concise and practical.

21. Do not mention:
    - Open-Meteo
    - APIs
    - retrieval
    - vector search
    - the knowledge base
    - internal systems
    - weather service implementation

22. Return valid Markdown. Use `##` headings for distinct sections, `- ` for
    bullets, and `1. ` for ordered steps. Leave blank lines between blocks;
    do not use Unicode bullets, raw HTML, or decorative separator lines.

WEATHER INFORMATION
-------------------

{weather_context}

ADDITIONAL AGRICULTURAL INFORMATION
-----------------------------------

{agricultural_context}

USER QUESTION
-------------

{question}

Answer directly and practically.
"""


def get_weather_context(
    latitude,
    longitude
):

    if (
        latitude is None
        or
        longitude is None
    ):

        logger.warning(
            "Weather requested but user coordinates "
            "were not supplied."
        )

        return ""

    logger.info(
        "Using coordinates supplied to chatbot: "
        "latitude=%s, longitude=%s",
        latitude,
        longitude
    )

    logger.info(
        "Fetching weather for "
        "latitude=%s, longitude=%s",
        latitude,
        longitude
    )

    try:

        weather_data = get_formatted_weather(
            latitude=latitude,
            longitude=longitude,
            forecast_days=7
        )

    except Exception:

        logger.exception(
            "Weather service failed."
        )

        return ""

    if not weather_data:

        logger.warning(
            "Weather service returned no data."
        )

        return ""

    try:

        weather_context = build_weather_context(
            weather_data
        )

    except Exception:

        logger.exception(
            "Failed to build weather context."
        )

        return ""

    if not weather_context:

        logger.warning(
            "Weather context is empty."
        )

        return ""

    logger.info(
        "Weather context generated successfully."
    )

    return weather_context


def handle_weather_question(
    question,
    weather_context,
    llm,
    agricultural_context=""
):

    logger.info(
        "Building weather-aware response."
    )

    user_prompt = build_weather_prompt(
        question=question,
        weather_context=weather_context,
        agricultural_context=agricultural_context
    )

    messages = build_messages(
        user_prompt
    )

    logger.info(
        "Sending weather-aware request to LLM..."
    )

    try:

        answer = ask_llm(
            client=llm,
            messages=messages
        )

    except Exception:

        logger.exception(
            "Weather-aware LLM request failed."
        )

        return (
            "I was able to retrieve the weather information, "
            "but I could not generate the weather-based answer "
            "right now."
        )

    logger.info(
        "Weather-aware response received."
    )

    return answer


def run_rag_pipeline(
    question,
    supabase,
    embedding_model,
    llm
):

    try:

        documents, best_relevance_score = retrieve_context(
            question=question,
            embedding_model=embedding_model,
            supabase=supabase,
            llm=llm
        )

    except Exception:

        logger.exception(
            "Retrieval pipeline failed."
        )

        return [], 0.0

    if not documents:

        logger.info(
            "No documents were retrieved."
        )

        return [], best_relevance_score

    logger.info(
        "Retrieved %d document(s).",
        len(documents)
    )

    return (
        documents,
        best_relevance_score
    )


def generate_rag_response(
    question,
    documents,
    llm
):

    context = build_context(
        documents
    )

    user_prompt = build_rag_prompt(
        context,
        question
    )

    messages = build_messages(
        user_prompt
    )

    logger.info(
        "Sending RAG request to LLM..."
    )

    answer = ask_llm(
        client=llm,
        messages=messages
    )

    logger.info(
        "RAG response generated."
    )

    citations = build_citations(
        documents
    )

    if citations:

        answer += (
            f"\n\n{citations}"
        )

        logger.info(
            "Citations appended."
        )

    return answer


def generate_general_response(
    question,
    llm
):

    logger.info(
        "Using general agricultural LLM knowledge."
    )

    user_prompt = build_general_prompt(
        question
    )

    messages = build_messages(
        user_prompt
    )

    logger.info(
        "Sending general agricultural request to LLM..."
    )

    answer = ask_llm(
        client=llm,
        messages=messages
    )

    logger.info(
        "General agricultural response generated."
    )

    return answer


def chat(
    question,
    supabase,
    embedding_model,
    llm,
    session_id,
    user_id=None,
    latitude=None,
    longitude=None
):

    if not question or not question.strip():

        logger.warning(
            "Empty user question received."
        )

        return {
            "response": "Please enter a question.",
            "topic": None
        }

    question = question.strip()

    if not session_id or not str(session_id).strip():

        logger.error(
            "Chat request received without session_id."
        )

        return {
            "response": (
                "A valid conversation session is required."
            ),
            "topic": None
        }

    session_id = str(
        session_id
    ).strip()

    if not user_id or not str(user_id).strip():

        logger.error(
            "Chat request received without user_id."
        )

        return {
            "response": (
                "A valid user ID is required."
            ),
            "topic": None
        }

    user_id = str(
        user_id
    ).strip()

    logger.info(
        "Processing chat request: "
        "user_id=%s, session_id=%s",
        user_id,
        session_id
    )

    if is_conversational(question):

        return handle_conversational_message(
            session_id=session_id,
            question=question,
            llm=llm,
            supabase=supabase,
            user_id=user_id
        )

    history = get_history(
        supabase=supabase,
        session_id=session_id,
        user_id=user_id
    )

    logger.info(
        "Session %s conversation history contains "
        "%d message(s).",
        session_id,
        len(history)
    )

    try:

        current_topic = get_topic(
            supabase=supabase,
            session_id=session_id,
            user_id=user_id
        )

    except Exception:

        logger.exception(
            "Failed to retrieve current topic "
            "before topic extraction."
        )

        current_topic = None

    logger.info(
        "Current topic before extraction for session %s: %s",
        session_id,
        current_topic
    )

    try:

        topic = extract_topic(
            llm=llm,
            question=question,
            history=history,
            current_topic=current_topic
        )

        if (
            topic
            and
            topic.strip()
            and
            topic.upper() != "UNKNOWN"
        ):

            topic = topic.strip()

            set_topic(
                supabase=supabase,
                session_id=session_id,
                topic=topic,
                user_id=user_id
            )

            logger.info(
                "Topic updated for session %s: %s",
                session_id,
                topic
            )

        else:

            logger.info(
                "Topic extraction returned UNKNOWN. "
                "Keeping existing topic for session %s: %s",
                session_id,
                current_topic
            )

    except Exception:

        logger.exception(
            "Topic extraction failed. "
            "Keeping existing topic for session %s: %s",
            session_id,
            current_topic
        )

    try:

        current_topic = get_topic(
            supabase=supabase,
            session_id=session_id,
            user_id=user_id
        )

    except Exception:

        logger.exception(
            "Failed to retrieve current topic "
            "after topic extraction."
        )

        current_topic = current_topic

    logger.info(
        "Current topic for session %s: %s",
        session_id,
        current_topic
    )

    if history or current_topic:

        try:

            standalone_question = rewrite_with_history(
                llm=llm,
                history=history,
                current_topic=current_topic,
                question=question
            )

            if not standalone_question:

                standalone_question = question

        except Exception:

            logger.exception(
                "History-based question rewriting failed. "
                "Using original question."
            )

            standalone_question = question

    else:

        logger.info(
            "No history or topic available. "
            "Skipping history rewrite."
        )

        standalone_question = question

    standalone_question = (
        standalone_question.strip()
        if standalone_question
        else question
    )

    logger.info(
        "Standalone question: %s",
        standalone_question
    )

    user_message_saved = add_user_message(
        supabase=supabase,
        session_id=session_id,
        user_id=user_id,
        message=question
    )

    if not user_message_saved:

        logger.warning(
            "Current user message could not be saved "
            "for session %s.",
            session_id
        )

    weather_request = is_weather_question(
        standalone_question
    )

    logger.info(
        "Weather question for session %s: %s",
        session_id,
        weather_request
    )

    if weather_request:

        logger.info(
            "Weather-related question detected "
            "for session %s.",
            session_id
        )

        weather_context = get_weather_context(
            latitude=latitude,
            longitude=longitude
        )

        if weather_context:

            answer = handle_weather_question(
                question=standalone_question,
                weather_context=weather_context,
                llm=llm
            )

            assistant_message_saved = add_assistant_message(
                supabase=supabase,
                session_id=session_id,
                user_id=user_id,
                message=answer
            )

            if not assistant_message_saved:

                logger.warning(
                    "Weather response could not be saved "
                    "for session %s.",
                    session_id
                )

            logger.info(
                "Weather response completed successfully "
                "for session %s.",
                session_id
            )

            return build_chat_response(
                answer=answer,
                supabase=supabase,
                session_id=session_id,
                user_id=user_id
            )

        logger.warning(
            "Weather was requested but no weather context "
            "could be generated. Falling back to RAG."
        )

    logger.info(
        "Starting retrieval pipeline."
    )

    documents, best_relevance_score = run_rag_pipeline(
        question=standalone_question,
        supabase=supabase,
        embedding_model=embedding_model,
        llm=llm
    )

    logger.info(
        "Final retrieval result for session %s: "
        "%d document(s), best relevance score: %.4f",
        session_id,
        len(documents),
        best_relevance_score
    )

    using_knowledge_base = (
        len(documents) > 0
        and
        best_relevance_score >= RELEVANCE_SCORE_THRESHOLD
    )

    if using_knowledge_base:

        logger.info(
            "Using Knowledge Base "
            "(Relevance Score: %.4f >= Threshold %.4f) "
            "for session %s.",
            best_relevance_score,
            RELEVANCE_SCORE_THRESHOLD,
            session_id
        )

        try:

            answer = generate_rag_response(
                question=standalone_question,
                documents=documents,
                llm=llm
            )

        except Exception:

            logger.exception(
                "RAG response generation failed. "
                "Falling back to general LLM."
            )

            answer = generate_general_response(
                question=standalone_question,
                llm=llm
            )

    else:

        logger.info(
            "Knowledge Base skipped "
            "(Retrieved documents: %d, "
            "Relevance Score: %.4f, "
            "Threshold: %.4f) "
            "for session %s.",
            len(documents),
            best_relevance_score,
            RELEVANCE_SCORE_THRESHOLD,
            session_id
        )

        answer = generate_general_response(
            question=standalone_question,
            llm=llm
        )

    assistant_message_saved = add_assistant_message(
        supabase=supabase,
        session_id=session_id,
        user_id=user_id,
        message=answer
    )

    if not assistant_message_saved:

        logger.warning(
            "Assistant response could not be saved "
            "for session %s.",
            session_id
        )

    current_history = get_history(
        supabase=supabase,
        session_id=session_id,
        user_id=user_id
    )

    logger.info(
        "Session %s conversation now contains "
        "%d message(s).",
        session_id,
        len(current_history)
    )

    return build_chat_response(
        answer=answer,
        supabase=supabase,
        session_id=session_id,
        user_id=user_id
    )
