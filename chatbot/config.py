"""
AgroFlow AI Configuration
"""


TOP_K = 3

RRF_SCORE_THRESHOLD = 0.10

TOP_CONTEXT_CHUNKS = 2

ENABLE_RERANKING = True

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

LLM_MODEL = "llama-3.3-70b-versatile"


LLM_FALLBACK_MODELS = [


    # "openai/gpt-oss-120b",

    "llama-3.1-8b-instant",

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