"""
AgroFlow AI Flask API
"""

import os
from datetime import datetime, timezone

from flask import Flask, request, jsonify
from flask_cors import CORS

from chatbot.database import get_supabase
from chatbot.embeddings import load_embedding_model
from chatbot.llm import get_llm
from chatbot.chat_engine import chat
from chatbot.logger import logger


app = Flask(__name__)

CORS(app)


logger.info(
    "Initializing AgroFlow AI API..."
)


supabase = None
embedding_model = None
llm = None


try:

    supabase = get_supabase()

    logger.info(
        "Supabase connected."
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
        "AgroFlow AI API is ready."
    )


except Exception:

    logger.exception(
        "Failed to initialize AgroFlow AI API."
    )

    supabase = None
    embedding_model = None
    llm = None


@app.route(
    "/chat",
    methods=["POST"]
)
def chat_endpoint():

    try:

        data = request.get_json(
            silent=True
        )


        if not data:

            return jsonify({
                "success": False,
                "error":
                    "Request body must contain valid JSON."
            }), 400


        user_message = data.get(
            "message"
        )


        session_id = data.get(
            "session_id"
        )


        user_id = data.get(
            "user_id"
        )


        latitude = data.get(
            "latitude"
        )


        longitude = data.get(
            "longitude"
        )


        if (
            not isinstance(
                user_message,
                str
            )
            or
            not user_message.strip()
        ):

            return jsonify({
                "success": False,
                "error":
                    "No message provided."
            }), 400


        user_message = user_message.strip()


        if (
            not isinstance(
                session_id,
                str
            )
            or
            not session_id.strip()
        ):

            return jsonify({
                "success": False,
                "error":
                    "session_id is required."
            }), 400


        session_id = session_id.strip()


        if (
            not isinstance(
                user_id,
                str
            )
            or
            not user_id.strip()
        ):

            return jsonify({
                "success": False,
                "error":
                    "user_id is required."
            }), 400


        user_id = user_id.strip()


        if latitude is not None:

            try:

                latitude = float(
                    latitude
                )

            except (
                TypeError,
                ValueError
            ):

                return jsonify({
                    "success": False,
                    "error":
                        "latitude must be a valid number."
                }), 400


        if longitude is not None:

            try:

                longitude = float(
                    longitude
                )

            except (
                TypeError,
                ValueError
            ):

                return jsonify({
                    "success": False,
                    "error":
                        "longitude must be a valid number."
                }), 400


        if (
            (
                latitude is None
                and
                longitude is not None
            )
            or
            (
                latitude is not None
                and
                longitude is None
            )
        ):

            return jsonify({
                "success": False,
                "error":
                    (
                        "Both latitude and longitude "
                        "must be provided together."
                    )
            }), 400


        if latitude is not None:

            if not -90 <= latitude <= 90:

                return jsonify({
                    "success": False,
                    "error":
                        "Invalid latitude."
                }), 400


        if longitude is not None:

            if not -180 <= longitude <= 180:

                return jsonify({
                    "success": False,
                    "error":
                        "Invalid longitude."
                }), 400


        unavailable_components = []


        if supabase is None:

            unavailable_components.append(
                "supabase"
            )


        if embedding_model is None:

            unavailable_components.append(
                "embedding_model"
            )


        if llm is None:

            unavailable_components.append(
                "llm"
            )


        if unavailable_components:

            logger.error(
                "AI service unavailable. "
                "Missing components: %s",
                ", ".join(
                    unavailable_components
                )
            )

            return jsonify({
                "success": False,
                "error":
                    "AI service is not properly initialized.",
                "components":
                    unavailable_components
            }), 503


        logger.info(
            "Chat request received "
            "(session_id=%s, user_id=%s)",
            session_id,
            user_id
        )


        logger.info(
            "User Question: %s",
            user_message
        )


        result = chat(

            question=user_message,

            supabase=supabase,

            embedding_model=embedding_model,

            llm=llm,

            session_id=session_id,

            user_id=user_id,

            latitude=latitude,

            longitude=longitude

        )


        if not isinstance(
            result,
            dict
        ):

            logger.error(
                "chat() returned an unexpected result: %r",
                result
            )

            return jsonify({
                "success": False,
                "error":
                    "AI returned an unexpected response."
            }), 500


        answer = result.get(
            "response",
            ""
        )


        topic = result.get(
            "topic"
        )


        if not isinstance(
            answer,
            str
        ):

            answer = str(
                answer
            ) if answer is not None else ""


        answer = answer.strip()


        if not answer:

            logger.error(
                "Chat engine returned an empty response."
            )

            return jsonify({
                "success": False,
                "error":
                    "AI returned an empty response."
            }), 500


        if topic is not None:

            topic = str(
                topic
            ).strip()

            if not topic:

                topic = None


        logger.info(
            "Answer generated successfully "
            "(session_id=%s, topic=%s)",
            session_id,
            topic
        )


        return jsonify({

            "success": True,

            "response": answer,

            "session_id": session_id,

            "topic": topic

        }), 200


    except Exception:

        logger.exception(
            "Chat request failed."
        )

        return jsonify({
            "success": False,
            "error":
                (
                    "An error occurred while processing "
                    "the chat request."
                )
        }), 500


@app.route(
    "/health",
    methods=["GET"]
)
def health():

    status = "healthy"

    details = {}


    if supabase is None:

        status = "degraded"

        details["supabase"] = (
            "not connected"
        )

    else:

        details["supabase"] = (
            "connected"
        )


    if embedding_model is None:

        status = "degraded"

        details["embedding_model"] = (
            "not loaded"
        )

    else:

        details["embedding_model"] = (
            "loaded"
        )


    if llm is None:

        status = "degraded"

        details["llm"] = (
            "not initialized"
        )

    else:

        details["llm"] = (
            "initialized"
        )


    return jsonify({

        "status": status,

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "details": details

    }), 200


@app.route(
    "/",
    methods=["GET"]
)
def root():

    return jsonify({

        "service":
            "AgroFlow AI",

        "version":
            "1.0.0",

        "status":
            "running",

        "endpoints": {

            "/chat":
                "POST - Send a message to AgroFlow AI",

            "/health":
                "GET - Check AI service health"

        }

    }), 200


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )


    logger.info(
        "Starting AgroFlow AI API "
        "on port %s",
        port
    )


    app.run(

        host="0.0.0.0",

        port=port

    )