"""
AgroFlow AI Chat Engine

Responsible for:

- Detecting simple conversational messages
- Maintaining conversation context
- Extracting conversation topics
- Rewriting follow-up questions
- Running the RAG retrieval pipeline
- Building prompts
- Generating LLM responses
- Adding citations
- Saving conversation history
"""

from chatbot.config import (
    RRF_SCORE_THRESHOLD
)

from chatbot.retrieval import (
    retrieve_context
)

from chatbot.context_builder import (
    build_context
)

from chatbot.context_compressor import (
    compress_context
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
    ask_llm
)

from chatbot.memory import (
    add_conversation
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

from chatbot.logger import logger


# ============================================================
# CONVERSATIONAL RESPONSE
# ============================================================

def build_conversational_prompt(question):
    """
    Build a short prompt for simple conversational messages.

    These messages do not require:

        - Topic extraction
        - History rewriting
        - Query rewriting
        - RAG retrieval
        - Vector search
        - Keyword search
        - RRF
        - Reranking
    """

    return f"""
You are AgroFlow AI, a friendly agricultural assistant.

The user sent a simple conversational message.

Respond naturally and briefly.

Rules:

- If the user greets you, greet them back warmly.
- If appropriate, mention that you are ready to help with agriculture or farming.
- If the user thanks you, respond naturally and briefly.
- If the user says okay, alright, great, nice, or perfect, acknowledge them naturally.
- If the user says continue or go on, respond naturally based on the ongoing conversation when possible.
- Do not force an agricultural explanation.
- Do not give unnecessary information.
- Do not mention the knowledge base.
- Do not mention retrieval.
- Do not mention system operations.

User message:

{question}

Respond naturally.
"""


# ============================================================
# SAVE CONVERSATION EXCHANGE
# ============================================================

def save_conversation_exchange(
    question,
    answer
):
    """
    Save the user message and assistant response
    to both conversation contexts.
    """

    add_conversation(
        question,
        answer
    )

    add_assistant_message(
        answer
    )

    logger.info(
        "Conversation exchange saved."
    )


# ============================================================
# HANDLE SIMPLE CONVERSATIONAL MESSAGE
# ============================================================

def handle_conversational_message(
    question,
    llm
):
    """
    Handle greetings, thanks, acknowledgements,
    and other simple conversational messages.

    The expensive RAG pipeline is completely skipped.
    """

    logger.info(
        "Conversational message detected. "
        "Skipping RAG pipeline."
    )

    # --------------------------------------------------------
    # Save user message
    # --------------------------------------------------------

    add_user_message(
        question
    )

    # --------------------------------------------------------
    # Build conversational prompt
    # --------------------------------------------------------

    user_prompt = build_conversational_prompt(
        question
    )

    # --------------------------------------------------------
    # Build messages
    # --------------------------------------------------------

    messages = build_messages(
        user_prompt
    )

    # --------------------------------------------------------
    # Ask LLM
    # --------------------------------------------------------

    logger.info(
        "Sending conversational response to LLM..."
    )

    answer = ask_llm(
        client=llm,
        messages=messages
    )

    logger.info(
        "Conversational response generated."
    )

    # --------------------------------------------------------
    # Save exchange
    # --------------------------------------------------------

    save_conversation_exchange(
        question,
        answer
    )

    return answer


# ============================================================
# ASK CHATBOT
# ============================================================

def chat(
    question,
    supabase,
    embedding_model,
    llm
):
    """
    Main AgroFlow AI chat function.

    Normal questions:

        User Question
             ↓
        Conversation History
             ↓
        Topic Extraction
             ↓
        History Rewriting
             ↓
        Retrieval Pipeline
             ↓
        Query Rewrite
             ↓
        Multi-Query Generation
             ↓
        Vector Search
             ↓
        Keyword Search
             ↓
        RRF
             ↓
        Optional Reranking
             ↓
        Context Compression
             ↓
        Prompt Construction
             ↓
        LLM
             ↓
        Citations
             ↓
        Conversation Memory


    Simple conversational messages:

        User Message
             ↓
        Conversational Detection
             ↓
        LLM
             ↓
        Conversation Memory
    """

    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    if not question or not question.strip():

        logger.warning(
            "Empty user question received."
        )

        return "Please enter a question."

    question = question.strip()


    # ========================================================
    # SIMPLE CONVERSATIONAL MESSAGE
    # ========================================================

    if is_conversational(question):

        return handle_conversational_message(
            question=question,
            llm=llm
        )


    # ========================================================
    # GET PREVIOUS CONVERSATION HISTORY
    # ========================================================

    history = get_history()

    logger.info(
        "Conversation history contains %d message(s).",
        len(history)
    )


    # ========================================================
    # EXTRACT CURRENT TOPIC
    # ========================================================

    try:

        topic = extract_topic(
            llm,
            question
        )

        if topic and topic.upper() != "UNKNOWN":

            set_topic(
                topic
            )

    except Exception:

        logger.exception(
            "Topic extraction failed."
        )

    logger.info(
        "Current topic: %s",
        get_topic()
    )


    # ========================================================
    # REWRITE QUESTION USING CONVERSATION HISTORY
    # ========================================================

    try:

        standalone_question = rewrite_with_history(

            llm=llm,

            history=history,

            current_topic=get_topic(),

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


    logger.info(
        "Standalone question: %s",
        standalone_question
    )


    # ========================================================
    # SAVE CURRENT USER MESSAGE
    # ========================================================

    add_user_message(
        question
    )


    # ========================================================
    # RETRIEVE DOCUMENTS
    # ========================================================

    try:

        documents, best_rrf_score = retrieve_context(

            question=standalone_question,

            embedding_model=embedding_model,

            supabase=supabase,

            llm=llm

        )

    except Exception:

        logger.exception(
            "Retrieval pipeline failed. "
            "Falling back to general agricultural knowledge."
        )

        documents = []

        best_rrf_score = 0.0


    logger.info(
        "Retrieved %d document(s).",
        len(documents)
    )


    # ========================================================
    # COMPRESS RETRIEVED CONTEXT
    # ========================================================

    if documents:

        documents = compress_context(
            documents
        )

    else:

        documents = []


    logger.info(
        "Context compressed to %d chunk(s).",
        len(documents)
    )


    # ========================================================
    # DECIDE WHETHER TO USE KNOWLEDGE BASE
    # ========================================================

    using_knowledge_base = (

        len(documents) > 0

        and

        best_rrf_score >= RRF_SCORE_THRESHOLD

    )


    # ========================================================
    # BUILD PROMPT
    # ========================================================

    if using_knowledge_base:

        logger.info(
            "Using Knowledge Base "
            "(Retrieval Score: %.4f)",
            best_rrf_score
        )

        # ----------------------------------------------------
        # Build retrieved context
        # ----------------------------------------------------

        context = build_context(
            documents
        )

        # ----------------------------------------------------
        # Build RAG prompt
        # ----------------------------------------------------

        user_prompt = build_rag_prompt(

            context,

            question

        )

    else:

        logger.info(
            "Knowledge Base skipped "
            "(Retrieval Score: %.4f < Threshold %.4f)",
            best_rrf_score,
            RRF_SCORE_THRESHOLD
        )

        # ----------------------------------------------------
        # No sufficiently relevant documents
        # ----------------------------------------------------

        user_prompt = build_general_prompt(
            question
        )


    # ========================================================
    # BUILD CONVERSATION MESSAGES
    # ========================================================

    messages = build_messages(
        user_prompt
    )


    # ========================================================
    # ASK LLM
    # ========================================================

    logger.info(
        "Sending request to LLM..."
    )

    answer = ask_llm(

        client=llm,

        messages=messages

    )


    logger.info(
        "LLM response received."
    )


    # ========================================================
    # APPEND CITATIONS
    # ========================================================

    if using_knowledge_base:

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


    # ========================================================
    # SAVE CONVERSATION
    # ========================================================

    save_conversation_exchange(

        question,

        answer

    )


    # ========================================================
    # CONVERSATION SUMMARY
    # ========================================================

    logger.info(
        "Conversation now contains %d message(s).",
        len(get_history())
    )


    # ========================================================
    # RETURN FINAL ANSWER
    # ========================================================

    return answer