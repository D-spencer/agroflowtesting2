"""
AgroFlow AI Topic Extractor
"""

from chatbot.llm import ask_llm
from chatbot.logger import logger


TOPIC_EXTRACTION_SYSTEM_PROMPT = """
You are an agricultural conversation topic extractor.

Your task is to identify the SINGLE primary topic currently
being discussed in the user's message.

The topic represents the actual subject of the conversation,
not necessarily a predefined agricultural category.

Examples of valid topics include:

Farming
Botany
Agronomy
Plant Physiology
Maize Cultivation
Rice Production
Soil Fertility
Soil pH
Irrigation
Fertilizer Application
Pest Management
Plant Diseases
Livestock Farming
Poultry Farming
Weather
Farm Equipment
Weed Management
Post-Harvest Handling
Farm Management
Agricultural Economics

These are examples only. They are NOT a fixed list.

If the user's message clearly introduces a different
agricultural subject, return the new subject.

If the user's message is a continuation of the existing
conversation topic, keep the existing topic.

If the user's message is a follow-up question whose meaning
depends on the previous conversation, use the conversation
context to determine the topic.

Rules:

1. Return exactly ONE topic.

2. Return the actual subject being discussed.

3. Keep the topic concise, usually between 1 and 4 words.

4. Prefer a specific topic when it clearly represents the
   user's subject.

5. Do not force a question into a broad category when a more
   accurate topic exists.

6. For example:
   "What is botany?" -> Botany
   "What are the branches of botany?" -> Botany
   "Is botany a good course to study?" -> Botany
   "How do I grow maize?" -> Maize Cultivation
   "What fertilizer should I use for maize?" -> Maize Fertilization
   "What is soil pH?" -> Soil pH
   "How can I control weeds?" -> Weed Management

7. If the user clearly changes from one agricultural subject
   to another, replace the previous topic with the new topic.

8. Do not include explanations.

9. Do not answer the user's question.

10. Do not return multiple topics.

11. If the message is not related to agriculture or an
    agriculture-related scientific subject, return:

UNKNOWN
"""


def extract_topic(
    llm,
    question,
    history=None,
    current_topic=None
):
    logger.info("Extracting conversation topic.")

    context = ""

    if current_topic:
        context += f"\nCurrent topic: {current_topic}"

    if history:
        context += "\nRecent conversation:\n"

        for message in history[-6:]:
            role = message.get("role", "")
            content = message.get("content", "")

            if content:
                context += f"{role}: {content}\n"

    user_content = f"""
{context}

Latest user message:
{question}

Identify the single primary topic currently being discussed.
Return only the topic.
"""

    messages = [
        {
            "role": "system",
            "content": TOPIC_EXTRACTION_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_content
        }
    ]

    try:
        topic = ask_llm(
            client=llm,
            messages=messages,
            temperature=0,
            max_tokens=20
        )

        topic = topic.strip()

    except Exception:
        logger.exception("Topic extraction failed.")
        return "UNKNOWN"

    if not topic:
        logger.warning("Topic extractor returned an empty response.")
        return "UNKNOWN"

    if topic.upper() == "UNKNOWN":
        logger.info("No agricultural topic detected.")
        return "UNKNOWN"

    topic = topic.strip("`\"'.")

    logger.info("Topic extracted successfully.")
    logger.debug("Question: %s", question)
    logger.debug("Current Topic: %s", current_topic)
    logger.debug("Extracted Topic: %s", topic)

    return topic