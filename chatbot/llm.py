# """
# LLM utilities for AgroFlow AI.

# Responsible for:

# - Creating the Groq client
# - Sending prompts to the LLM
# - Returning generated responses
# """

# import os

# from dotenv import load_dotenv

# from groq import Groq

# from chatbot.logger import logger

# from chatbot.config import (

#     LLM_MODEL,

#     TEMPERATURE,

#     MAX_TOKENS

# )

# load_dotenv()


# # ============================================================
# # GLOBAL CLIENT CACHE
# # ============================================================

# _llm_client = None


# # ============================================================
# # CREATE GROQ CLIENT
# # ============================================================

# def get_llm():

#     """
#     Create the Groq client once and
#     reuse it throughout the application.
#     """

#     global _llm_client

#     if _llm_client is not None:

#         logger.info(

#             "Groq client already initialized."

#         )

#         return _llm_client

#     api_key = os.getenv(

#         "GROQ_API_KEY"

#     )

#     if not api_key:

#         raise ValueError(

#             "GROQ_API_KEY is missing."

#         )

#     try:

#         _llm_client = Groq(

#             api_key=api_key

#         )

#         logger.info(

#             "Groq client initialized."

#         )

#         return _llm_client

#     except Exception:

#         logger.exception(

#             "Failed to initialize Groq client."

#         )

#         raise


# # ============================================================
# # GENERATE RESPONSE
# # ============================================================

# def ask_llm(

#     client,

#     messages,

#     temperature=TEMPERATURE,

#     max_tokens=MAX_TOKENS

# ):

#     """
#     Send a chat completion request
#     to Groq.
#     """

#     try:

#         response = client.chat.completions.create(

#             model=LLM_MODEL,

#             messages=messages,

#             temperature=temperature,

#             max_tokens=max_tokens

#         )

#         answer = response.choices[0].message.content

#         logger.info(

#             f"Groq response generated using model '{LLM_MODEL}'."

#         )

#         return answer

#     except Exception:

#         logger.exception(

#             "Groq request failed."

#         )

#         raise














































"""
LLM utilities for AgroFlow AI.

Responsible for:

- Creating and caching the Groq client
- Sending prompts to the LLM
- Automatic model fallback
- Rate-limit handling
- Model cooldown management
- Retry handling for temporary failures
- Returning generated responses
"""

import os
import re
import time
import random

from dotenv import load_dotenv

from groq import Groq
from groq import RateLimitError

from chatbot.logger import logger

from chatbot.config import (
    LLM_MODEL,
    LLM_FALLBACK_MODELS,
    ENABLE_MODEL_FALLBACK,
    MAX_MODEL_RETRIES,
    TEMPERATURE,
    MAX_TOKENS,
)

load_dotenv()


# ============================================================
# GLOBAL CLIENT CACHE
# ============================================================

_llm_client = None


# ============================================================
# MODEL COOLDOWN CACHE
# ============================================================

"""
Stores the timestamp at which a rate-limited model
can be tried again.

Example:

{
    "llama-3.3-70b-versatile": 1754598212.5
}
"""

_model_cooldowns = {}


# ============================================================
# CREATE GROQ CLIENT
# ============================================================

def get_llm():
    """
    Create the Groq client once and reuse it
    throughout the application.
    """

    global _llm_client

    # --------------------------------------------------------
    # Return existing client
    # --------------------------------------------------------

    if _llm_client is not None:

        logger.info(
            "Groq client already initialized."
        )

        return _llm_client

    # --------------------------------------------------------
    # Get API key
    # --------------------------------------------------------

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY is missing."
        )

    # --------------------------------------------------------
    # Create client
    # --------------------------------------------------------

    try:

        _llm_client = Groq(
            api_key=api_key
        )

        logger.info(
            "Groq client initialized successfully."
        )

        return _llm_client

    except Exception:

        logger.exception(
            "Failed to initialize Groq client."
        )

        raise


# ============================================================
# GET CONFIGURED MODELS
# ============================================================

def get_models():
    """
    Return the models in priority order.

    Primary model is always first.

    Fallback models are added afterward.
    """

    models = [
        LLM_MODEL
    ]

    if ENABLE_MODEL_FALLBACK:

        for model in LLM_FALLBACK_MODELS:

            if model not in models:

                models.append(model)

    return models


# ============================================================
# CHECK MODEL AVAILABILITY
# ============================================================

def is_model_available(model):
    """
    Check whether a model is currently available.

    A model becomes unavailable when Groq reports
    that its rate limit has been reached.
    """

    cooldown_until = _model_cooldowns.get(
        model
    )

    # No cooldown registered
    if cooldown_until is None:

        return True

    current_time = time.time()

    # --------------------------------------------------------
    # Cooldown expired
    # --------------------------------------------------------

    if current_time >= cooldown_until:

        del _model_cooldowns[model]

        logger.info(
            "Model '%s' cooldown expired. "
            "It will be available again.",
            model
        )

        return True

    # --------------------------------------------------------
    # Still cooling down
    # --------------------------------------------------------

    remaining = cooldown_until - current_time

    logger.info(
        "Model '%s' is cooling down. "
        "%.0f seconds remaining.",
        model,
        remaining
    )

    return False


# ============================================================
# EXTRACT RATE LIMIT WAIT TIME
# ============================================================

def extract_retry_seconds(error):
    """
    Extract the retry duration from a Groq
    rate-limit error.

    Example Groq message:

        Please try again in 58m26.976s

    Returns:
        Number of seconds.
    """

    message = str(error)

    # --------------------------------------------------------
    # Match minutes + seconds
    # --------------------------------------------------------

    match = re.search(
        r"(\d+)m([\d.]+)s",
        message
    )

    if match:

        minutes = int(
            match.group(1)
        )

        seconds = float(
            match.group(2)
        )

        return (
            minutes * 60
            + seconds
        )

    # --------------------------------------------------------
    # Match seconds only
    # --------------------------------------------------------

    match = re.search(
        r"([\d.]+)s",
        message
    )

    if match:

        return float(
            match.group(1)
        )

    # --------------------------------------------------------
    # Match "retry-after"
    # --------------------------------------------------------

    match = re.search(
        r"retry[- ]after[:\s]+([\d.]+)",
        message,
        re.IGNORECASE
    )

    if match:

        return float(
            match.group(1)
        )

    # --------------------------------------------------------
    # Safe default
    # --------------------------------------------------------

    return 60.0


# ============================================================
# REGISTER MODEL COOLDOWN
# ============================================================

def set_model_cooldown(
    model,
    error
):
    """
    Put a rate-limited model into cooldown.
    """

    retry_seconds = extract_retry_seconds(
        error
    )

    # Add a small safety buffer so we don't
    # immediately hit the same limit again.
    retry_seconds += 2

    cooldown_until = (
        time.time()
        + retry_seconds
    )

    _model_cooldowns[model] = (
        cooldown_until
    )

    logger.warning(
        "Model '%s' is rate limited. "
        "Putting model into cooldown for %.1f seconds.",
        model,
        retry_seconds
    )


# ============================================================
# CALCULATE BACKOFF
# ============================================================

def calculate_backoff(attempt):
    """
    Calculate exponential backoff with jitter.

    Example:

        attempt 0 -> ~0.5s
        attempt 1 -> ~1.0s
        attempt 2 -> ~2.0s

    Random jitter prevents multiple requests
    from retrying at exactly the same time.
    """

    base_delay = 0.5

    max_delay = 8.0

    delay = min(
        base_delay * (2 ** attempt),
        max_delay
    )

    jitter = random.uniform(
        0,
        0.25
    )

    return delay + jitter


# ============================================================
# CHECK TRANSIENT ERROR
# ============================================================

def is_transient_error(error):
    """
    Determine whether an exception is likely
    temporary and worth retrying.
    """

    status_code = getattr(
        error,
        "status_code",
        None
    )

    # --------------------------------------------------------
    # HTTP status codes that are usually temporary
    # --------------------------------------------------------

    if status_code in (
        500,
        502,
        503,
        504,
    ):

        return True

    # --------------------------------------------------------
    # Network-related errors
    # --------------------------------------------------------

    error_name = type(
        error
    ).__name__.lower()

    error_message = str(
        error
    ).lower()

    network_keywords = [

        "timeout",

        "timed out",

        "connection",

        "temporarily unavailable",

        "server error",

        "service unavailable",

        "bad gateway",

        "gateway timeout",

    ]

    if any(
        keyword in error_message
        for keyword in network_keywords
    ):

        return True

    if "connection" in error_name:

        return True

    if "timeout" in error_name:

        return True

    return False


# ============================================================
# SEND SINGLE REQUEST
# ============================================================

def _request_model(
    client,
    model,
    messages,
    temperature,
    max_tokens,
):
    """
    Send one request to one specific model.
    """

    return client.chat.completions.create(

        model=model,

        messages=messages,

        temperature=temperature,

        max_tokens=max_tokens,

    )


# ============================================================
# ASK LLM
# ============================================================

def ask_llm(
    client,
    messages,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
):
    """
    Send a chat completion request to Groq.

    Model selection strategy:

    1. Try the primary model.
    2. Skip models currently in cooldown.
    3. If a model hits a rate limit, put it
       into cooldown.
    4. Move to the next available model.
    5. Retry temporary server/network failures.
    6. Automatically return to the primary model
       after its cooldown expires.
    """

    models = get_models()

    last_exception = None

    # ========================================================
    # LOOP THROUGH MODELS
    # ========================================================

    for model in models:

        # ----------------------------------------------------
        # Skip rate-limited models
        # ----------------------------------------------------

        if not is_model_available(
            model
        ):

            logger.info(
                "Skipping model '%s' "
                "because it is still cooling down.",
                model
            )

            continue

        logger.info(
            "Attempting LLM request with model '%s'.",
            model
        )

        # ----------------------------------------------------
        # Retry count
        # ----------------------------------------------------

        attempts = MAX_MODEL_RETRIES + 1

        # ====================================================
        # RETRY CURRENT MODEL
        # ====================================================

        for attempt in range(
            attempts
        ):

            try:

                response = _request_model(

                    client=client,

                    model=model,

                    messages=messages,

                    temperature=temperature,

                    max_tokens=max_tokens,

                )

                # ------------------------------------------------
                # Extract response
                # ------------------------------------------------

                answer = (
                    response
                    .choices[0]
                    .message
                    .content
                )

                if not answer:

                    raise ValueError(
                        f"Model '{model}' returned an empty response."
                    )

                logger.info(
                    "LLM response generated successfully "
                    "using model '%s'.",
                    model
                )

                return answer

            # ====================================================
            # RATE LIMIT
            # ====================================================

            except RateLimitError as error:

                last_exception = error

                logger.warning(
                    "Rate limit reached for model '%s'.",
                    model
                )

                # ------------------------------------------------
                # Remember cooldown
                # ------------------------------------------------

                set_model_cooldown(

                    model=model,

                    error=error

                )

                # ------------------------------------------------
                # IMPORTANT:
                #
                # Do NOT retry this model.
                #
                # The API already told us that the
                # model is unavailable.
                # ------------------------------------------------

                logger.info(
                    "Moving to next available model."
                )

                break

            # ====================================================
            # TEMPORARY SERVER / NETWORK ERROR
            # ====================================================

            except Exception as error:

                last_exception = error

                # ------------------------------------------------
                # Check if retryable
                # ------------------------------------------------

                if is_transient_error(
                    error
                ):

                    if attempt < attempts - 1:

                        delay = calculate_backoff(
                            attempt
                        )

                        logger.warning(

                            "Temporary error from model '%s'. "
                            "Retrying in %.2f seconds "
                            "(attempt %d/%d).",

                            model,

                            delay,

                            attempt + 1,

                            attempts

                        )

                        time.sleep(
                            delay
                        )

                        continue

                # ------------------------------------------------
                # Non-retryable error
                # ------------------------------------------------

                logger.exception(
                    "Model '%s' failed with a non-retryable error.",
                    model
                )

                break

        logger.info(
            "Model '%s' was unsuccessful. "
            "Checking next model.",
            model
        )

    # ========================================================
    # ALL MODELS FAILED
    # ========================================================

    logger.error(
        "All configured LLM models failed."
    )

    if last_exception is not None:

        raise last_exception

    raise RuntimeError(
        "No available LLM model could generate a response."
    )