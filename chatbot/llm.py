"""
LLM utilities for AgroFlow AI.
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


_llm_client = None
_model_cooldowns = {}


def get_llm():
    """Create and cache the Groq client."""

    global _llm_client

    if _llm_client is not None:
        logger.info("Groq client already initialized.")
        return _llm_client

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing.")

    try:
        _llm_client = Groq(api_key=api_key)

        logger.info(
            "Groq client initialized successfully."
        )

        return _llm_client

    except Exception:
        logger.exception(
            "Failed to initialize Groq client."
        )
        raise


def get_models():
    """Return configured models in priority order."""

    models = [LLM_MODEL]

    if ENABLE_MODEL_FALLBACK:
        for model in LLM_FALLBACK_MODELS:
            if model not in models:
                models.append(model)

    return models


def is_model_available(model):
    """Check whether a model is outside its cooldown period."""

    cooldown_until = _model_cooldowns.get(model)

    if cooldown_until is None:
        return True

    current_time = time.time()

    if current_time >= cooldown_until:
        del _model_cooldowns[model]

        logger.info(
            "Model '%s' cooldown expired.",
            model
        )

        return True

    remaining = cooldown_until - current_time

    logger.info(
        "Model '%s' is cooling down. "
        "%.0f seconds remaining.",
        model,
        remaining
    )

    return False


def extract_retry_seconds(error):
    """Extract retry duration from a rate-limit error."""

    message = str(error)

    match = re.search(
        r"(\d+)m([\d.]+)s",
        message
    )

    if match:
        minutes = int(match.group(1))
        seconds = float(match.group(2))

        return minutes * 60 + seconds

    match = re.search(
        r"([\d.]+)s",
        message
    )

    if match:
        return float(match.group(1))

    match = re.search(
        r"retry[- ]after[:\s]+([\d.]+)",
        message,
        re.IGNORECASE
    )

    if match:
        return float(match.group(1))

    return 60.0


def set_model_cooldown(model, error):
    """Put a rate-limited model into cooldown."""

    retry_seconds = extract_retry_seconds(error) + 2

    cooldown_until = time.time() + retry_seconds

    _model_cooldowns[model] = cooldown_until

    logger.warning(
        "Model '%s' is rate limited. "
        "Cooldown: %.1f seconds.",
        model,
        retry_seconds
    )


def calculate_backoff(attempt):
    """Calculate exponential backoff with jitter."""

    base_delay = 0.5
    max_delay = 8.0

    delay = min(
        base_delay * (2 ** attempt),
        max_delay
    )

    jitter = random.uniform(0, 0.25)

    return delay + jitter


def is_transient_error(error):
    """Determine whether an error is temporary."""

    status_code = getattr(
        error,
        "status_code",
        None
    )

    if status_code in (500, 502, 503, 504):
        return True

    error_name = type(error).__name__.lower()
    error_message = str(error).lower()

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


def clean_llm_text(text):
    """Remove leaked reasoning markers from model output."""

    if not text:
        return text

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    text = re.sub(
        r"</?think>",
        "",
        text,
        flags=re.IGNORECASE
    )

    return text.strip()


def format_markdown_response(text):
    """Normalize common model formatting variations into display-safe Markdown.

    Prompt instructions provide the structure; this light normalization keeps
    the API contract consistent without rewriting the model's content.
    """

    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    normalized_lines = []
    in_code_fence = False

    for line in text.split("\n"):
        stripped_line = line.lstrip()

        if stripped_line.startswith(("```", "~~~")):
            in_code_fence = not in_code_fence
            normalized_lines.append(line.rstrip())
            continue

        if not in_code_fence:
            # Models sometimes use typographic bullets or ``1)`` despite the
            # prompt. Do not change examples inside fenced code blocks.
            line = re.sub(r"^[ \t]*[•‣◦][ \t]+", "- ", line)
            line = re.sub(r"^([ \t]*\d+)\)[ \t]+", r"\1. ", line)

        normalized_lines.append(line.rstrip())

    # Never return trailing whitespace, which can unintentionally create hard
    # line breaks in Markdown renderers.
    return "\n".join(normalized_lines).strip()


def get_reasoning_settings(model, reasoning_effort):
    """Return reasoning settings compatible with the selected model."""

    if model.startswith("qwen/qwen3."):
        if reasoning_effort not in {
            "none",
            "default",
            "low",
            "medium",
            "high",
        }:
            reasoning_effort = "none"

        return {
            "reasoning_effort": reasoning_effort,
            "reasoning_format": "hidden",
        }

    if model in {
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    }:
        if reasoning_effort == "none":
            reasoning_effort = "low"

        if reasoning_effort not in {
            "low",
            "medium",
            "high",
        }:
            reasoning_effort = "low"

        return {
            "reasoning_effort": reasoning_effort,
            "reasoning_format": "hidden",
        }

    return {}


def _request_model(
    client,
    model,
    messages,
    temperature,
    max_tokens,
    reasoning_effort="none",
):
    """Send one request to one model."""

    reasoning_settings = get_reasoning_settings(
        model,
        reasoning_effort
    )

    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **reasoning_settings,
    )


def ask_llm(
    client,
    messages,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    reasoning_effort="none",
):
    """Send a chat completion request with model fallback."""

    models = get_models()

    last_exception = None

    for model in models:

        if not is_model_available(model):
            logger.info(
                "Skipping model '%s' because it is cooling down.",
                model
            )
            continue

        logger.info(
            "Attempting LLM request with model '%s'.",
            model
        )

        attempts = MAX_MODEL_RETRIES + 1

        for attempt in range(attempts):

            try:

                response = _request_model(
                    client=client,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                )

                answer = (
                    response
                    .choices[0]
                    .message
                    .content
                )

                answer = clean_llm_text(answer)

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

            except RateLimitError as error:

                last_exception = error

                logger.warning(
                    "Rate limit reached for model '%s'.",
                    model
                )

                set_model_cooldown(
                    model=model,
                    error=error
                )

                logger.info(
                    "Moving to next available model."
                )

                break

            except Exception as error:

                last_exception = error

                if is_transient_error(error):

                    if attempt < attempts - 1:

                        delay = calculate_backoff(attempt)

                        logger.warning(
                            "Temporary error from model '%s'. "
                            "Retrying in %.2f seconds "
                            "(attempt %d/%d).",
                            model,
                            delay,
                            attempt + 1,
                            attempts
                        )

                        time.sleep(delay)

                        continue

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

    logger.error(
        "All configured LLM models failed."
    )

    if last_exception is not None:
        raise last_exception

    raise RuntimeError(
        "No available LLM model could generate a response."
    )
