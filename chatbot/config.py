"""
AgroFlow AI Configuration
"""

TOP_K = 3

RELEVANCE_SCORE_THRESHOLD = 0.10

RRF_SCORE_THRESHOLD = RELEVANCE_SCORE_THRESHOLD

TOP_CONTEXT_CHUNKS = 2

ENABLE_RERANKING = True

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

LLM_MODEL = "qwen/qwen3.6-27b"

LLM_FALLBACK_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
]

ENABLE_MODEL_FALLBACK = True

MAX_MODEL_RETRIES = 1

TEMPERATURE = 0.2

MAX_TOKENS = 350

MAX_HISTORY = 6

ENABLE_MEMORY = True

ENABLE_QUERY_REWRITE = True

ENABLE_RAG = True

ENABLE_CITATIONS = True

ENABLE_HYBRID_SEARCH = False